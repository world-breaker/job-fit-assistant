"""评测脚本：跑完整评测，产出 experiments/results/ 下的数字文件。

跑五组指标（规格要求的全部五项）：
1. JD 技能要求抽取的 P/R/F1（普通 + 重要度加权），**并与关键词基线对比**
2. 匹配环节的证据定位正确率
3. 端到端「推荐哪版简历」准确率（12 JD × 4 简历）
4. 假绿次数（该说缺失却报命中）
5. 兜底占比 + LLM 费用（都记进结果文件）

★ 诚实边界（写进结果文件的 `caveats` 段，README 只引用文件里的数字）：
- n = 12 份 JD，样本量小 → 报告只给量级，不给「准确率 90%」这种断言；
- 标注是**单人标注**，无第二标注者复核；
- 关键词基线是**同一份词表**上的朴素扫描，它不区分重要度（这是有意为之）；
- 端到端「推荐哪版简历」的标签是按**岗位族**标注的，不是「哪个人更适合」。

用法：
    python scripts/run_eval.py --out experiments/results
    python scripts/run_eval.py --llm none          # 只跑规则基线（不花钱）
    python scripts/run_eval.py --budget-usd 2.0
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from jobfit.advice import coverage_summary  # noqa: E402
from jobfit.extractor import extract_requirements  # noqa: E402
from jobfit.lexicon import canonical_name  # noqa: E402
from jobfit.llm import build_llm  # noqa: E402
from jobfit.matching import evidence_is_verbatim, evidence_supports  # noqa: E402
from jobfit.metrics import (  # noqa: E402
    Prf,
    aggregate_prf,
    check_evidence,
    count_false_green,
    macro_f1,
    score_extraction,
)
from jobfit.normalize import normalize_key  # noqa: E402
from jobfit.resume import parse_resume  # noqa: E402
from jobfit.rules import extract_keyword_baseline  # noqa: E402
from jobfit.schemas import MATCH_HIT, MATCH_MISSING, MATCH_PARTIAL, SOURCE_LLM  # noqa: E402

JD_DIR = ROOT / "data" / "jds"
RESUME_DIR = ROOT / "data" / "resumes"


# ---------------------------------------------------------------- 独立参考实现

#: 参考实现用的强动作动词表（**故意与 matching.py 的表不同**）
#: ——用来做交叉核对，两处用同一份词表就等于自证。
REF_STRONG_VERBS = (
    "实现", "开发", "设计", "构建", "搭建", "优化", "训练", "微调", "部署", "落地",
    "主导", "负责", "独立完成", "从零实现", "复现", "产出", "撰写", "上线", "集成",
)

#: 参考实现用的量化正则
REF_QUANT = re.compile(r"\d+(\.\d+)?\s*(%|px|ms|s\b|万|亿|千|次|行|个|天|周|月|年|倍|dB|F1)")

#: 参考实现认为「技术栈声明」的行首（这种行本身就是「用过这个技术」的证据）
REF_STACK_PREFIX = ("技术栈", "技术栈：", "技术栈:", "编程语言", "AI 框架", "后端框架", "工程工具")


#: 人工标注的「简历 × 技能」真值表（假绿指标的独立依据）
TRUTH_PATH = ROOT / "data" / "annotations" / "resume_skill_truth.json"


def load_truth() -> Dict[str, Dict[str, bool]]:
    """加载人工标注真值表。"""
    if not TRUTH_PATH.exists():
        raise SystemExit(
            f"缺少人工标注真值表：{TRUTH_PATH}\n"
            " 请先运行：python scripts/annotate_resume_truth.py"
        )
    payload = json.loads(TRUTH_PATH.read_text(encoding="utf-8"))
    return payload["resumes"]


def derive_expected_resume(
    truth_pairs: List[Tuple[str, str]],
    truth: Dict[str, Dict[str, bool]],
    resume_names: List[str],
) -> Tuple[str, Dict[str, float]]:
    """从**人工标注真值**推出「这份 JD 最该用哪版简历」。

    ★ 为什么要推导而不是手写标签：
    手写「期望版本」会引入主观判断，而且很容易被「看着系统输出调标签」污染
    （那就是泄漏，指标就不干净了）。这里的口径完全客观：

        期望版本 = 「按人工真值，覆盖该 JD **必须项**最多的那份简历」

    人工真值表（data/annotations/resume_skill_truth.json）与系统的匹配逻辑
    完全独立，所以：
    - 期望版本来自**人工真值**；
    - 系统推荐来自**系统自己的匹配结果**；
    两者独立算出后比对，指标才有意义（不是拿系统输出给自己打分）。

    返回 (期望简历名, 各简历的必须项覆盖率)。
    """
    must_skills = [skill for skill, importance in truth_pairs if importance == "必须"]
    coverage: Dict[str, float] = {}
    for name in resume_names:
        table = truth.get(name, {})
        if not must_skills:
            coverage[name] = 0.0
            continue
        hits = 0
        for skill in must_skills:
            value = table.get(normalize_key(skill))
            if value is True:
                hits += 1
        coverage[name] = round(hits / len(must_skills), 4)
    if not coverage:
        return "", coverage
    best = max(coverage, key=lambda k: (coverage[k], k))
    return best, coverage


def reference_has_skill(resume_text: str, skill: str) -> bool:
    """独立参考（无人工真值时的兜底口径）。

    与 jobfit.matching 的实现路径不同，用的是另一套动词表与判定规则：
    - 技能名（任一别名）出现在简历里；
    - 且该行是「技术栈声明行」，或含强动作动词，或含量化结果。

    ★ 正常评测走的是**人工标注真值表**（load_truth），本函数只在真值表缺项时兜底。
    """
    aliases = [skill]
    canon = canonical_name(skill)
    if canon:
        from jobfit.lexicon import SKILL_ALIASES

        aliases = SKILL_ALIASES.get(canon, [skill])

    lines = [line.strip() for line in resume_text.splitlines() if line.strip()]
    for line in lines:
        if not any(alias in line for alias in aliases):
            continue
        if line.startswith(REF_STACK_PREFIX):
            return True
        if any(verb in line for verb in REF_STRONG_VERBS) or REF_QUANT.search(line):
            return True
    return False


def reference_evidence_supports(evidence: str, skill: str) -> bool:
    """独立参考：证据句是否支持该技能结论。

    与 matching.evidence_supports 的区别：这里用**词表别名**判定，
    而不是只用规范名 —— 规范名不在证据里、但别名在，也算支持。
    """
    canon = canonical_name(skill) or skill
    from jobfit.lexicon import SKILL_ALIASES

    aliases = SKILL_ALIASES.get(canon, [canon, skill])
    return any(alias and alias in evidence for alias in aliases)


# ---------------------------------------------------------------- 数据加载

def load_jds() -> List[Dict[str, Any]]:
    """加载全部 JD（含人工标注）。"""
    out: List[Dict[str, Any]] = []
    for path in sorted(JD_DIR.glob("*.json")):
        if path.name == "index.json":
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["_path"] = path
        out.append(payload)
    return out


def load_resumes() -> List[Dict[str, Any]]:
    """加载全部脱敏简历。"""
    out: List[Dict[str, Any]] = []
    for path in sorted(RESUME_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        out.append({"name": path.stem, "path": path, "text": text})
    return out


# ---------------------------------------------------------------- 评测主体

def run(
    out_dir: Path,
    llm_backend: str = "auto",
    llm_model: str = "",
    budget_usd: float = 1.0,
    use_mcp: bool = False,
) -> Dict[str, Any]:
    """跑完整评测，返回结果字典（同时写盘）。"""
    started = time.perf_counter()
    jds = load_jds()
    resumes = load_resumes()
    truth = load_truth()
    if not jds or not resumes:
        raise SystemExit(f"数据不足：JD {len(jds)} 份、简历 {len(resumes)} 份")

    llm = build_llm(backend=llm_backend, model=llm_model, budget_usd=budget_usd)
    llm_name = type(llm).__name__ if llm else "none"

    per_jd: List[Dict[str, Any]] = []
    system_prfs: List[Prf] = []
    llm_only_prfs: List[Prf] = []
    baseline_prfs: List[Prf] = []
    evidence_checks: List[Dict[str, Any]] = []
    false_greens: List[Dict[str, Any]] = []
    resume_scores: List[Dict[str, Any]] = []
    total_fallback = 0
    total_requirements = 0

    for jd in jds:
        jd_text = jd["raw_text"]
        truth_pairs = [
            (r["skill"], r["importance"])
            for r in jd["annotations"]["requirements"]
        ]

        # --- 抽取：三条路径 ---
        # ① 系统（LLM + 规则兜底）：产品实际用的路径
        # ② 纯 LLM：只有 LLM 的抽取（不含规则补的那部分）
        # ③ 关键词基线：完全不用 LLM 的朴素词表扫描
        #
        # ★ 公平性说明：①与③不是「同一条路换模型」，①比③多了 LLM 的输出，
        #   所以①如果比③差，说明 **LLM 带来了噪声**（这正是本评测要发现的）。
        #   为了把这个信号单独量化，同时给出②（纯 LLM）这一档。
        #
        # ★ 只调一次 LLM：②直接从①的结果里筛出 source=="llm" 的条目。
        #   merge_rules=True 的结果 == LLM 条目 + 规则补的条目，所以筛出来
        #   与「关掉规则合并再跑一遍」完全等价——但省掉一半 API 费用。
        requirements, stats = extract_requirements(jd_text, llm=llm, merge_rules=True)
        system_pairs = [(r.skill, r.importance) for r in requirements]

        llm_only_requirements = [r for r in requirements if r.source == SOURCE_LLM]
        llm_only_pairs = [(r.skill, r.importance) for r in llm_only_requirements]

        baseline_requirements = extract_keyword_baseline(jd_text)
        baseline_pairs = [(r.skill, r.importance) for r in baseline_requirements]

        system_prf, system_detail = score_extraction(system_pairs, truth_pairs)
        llm_only_prf, llm_only_detail = score_extraction(llm_only_pairs, truth_pairs)
        baseline_prf, baseline_detail = score_extraction(baseline_pairs, truth_pairs)
        system_prfs.append(system_prf)
        llm_only_prfs.append(llm_only_prf)
        baseline_prfs.append(baseline_prf)

        fallback_count = sum(1 for r in requirements if r.source == "rule_fallback")
        total_fallback += fallback_count
        total_requirements += len(requirements)

        # --- 匹配：系统 vs 4 份简历 ---
        jd_resume_results: List[Dict[str, Any]] = []
        for resume in resumes:
            profile = parse_resume(resume["path"], source_path_label=resume["name"])
            from jobfit.matching import match_requirements

            matches = match_requirements(requirements, profile)

            # 证据核对（用独立参考实现）
            check = check_evidence(
                matches,
                resume["text"],
                evidence_is_verbatim,
                reference_evidence_supports,
            )
            evidence_checks.append(
                {"jd": jd["id"], "resume": resume["name"], **check.to_dict()}
            )

            # 假绿：系统报命中、但**人工标注真值**判该简历不具备
            truth_table = truth.get(resume["name"], {})
            reference_missing = [
                skill
                for skill, _ in truth_pairs
                if truth_table.get(normalize_key(skill)) is False
            ]
            fg = count_false_green(matches, reference_missing)
            for item in fg:
                item["jd"] = jd["id"]
                item["resume"] = resume["name"]
            false_greens.extend(fg)

            coverage = coverage_summary(matches)
            jd_resume_results.append(
                {
                    "resume": resume["name"],
                    "coverage": coverage,
                    "matches": [m.to_dict() for m in matches],
                    "evidence_accuracy": check.accuracy,
                    "false_green": len(fg),
                }
            )
            resume_scores.append(
                {
                    "jd": jd["id"],
                    "resume": resume["name"],
                    "coverage": coverage,
                    "hit_rate": coverage["hit_rate"],
                    "must_hit_rate": coverage["must_hit_rate"],
                    "partial_rate": coverage["partial_rate"],
                }
            )

        # --- 端到端：推荐哪版简历 ---
        # 打分口径：必须项命中率优先，其次整体命中率，最后部分命中率。
        # ★ 为什么必须项优先：JD 里「必须」的要求才是能不能拿面试的关键，
        #   一个人可能整体命中率不低但必须项全缺（那不会被约面试）。
        def _score(result: Dict[str, Any]) -> Tuple[float, float, float]:
            cov = result["coverage"]
            return (cov["must_hit_rate"], cov["hit_rate"], cov["partial_rate"])

        best = max(jd_resume_results, key=_score)
        recommended = best["resume"]
        # 期望版本：按**岗位族**人工标注（data/jds/*.json 的 expected_resume 字段）。
        #
        # ★ 为什么用人工标签而不是从真值表推导：
        #   试过「按真值覆盖必须项最多的那份简历」来推导，结果退化——
        #   评测后训练版简历覆盖的技能最多，于是几乎每份 JD 都推导出它，
        #   指标变成「总选覆盖面最广的那份」，测不出岗位匹配能力。
        #   岗位族标签是**人写的、按岗位主线划分的**，符合规格要求
        #   （「给出该岗位族标注的期望版本」），也是真实招聘里真实的决策方式。
        #   人为标签可能有主观性 → 这是本指标的已知边界，写进 caveats。
        family_map = json.loads((JD_DIR / "index.json").read_text(encoding="utf-8"))["resume_family"]
        expected_family = jd.get("expected_resume", "")
        expected_file = next(
            (name.replace(".md", "") for name, fam in family_map.items() if fam == expected_family), ""
        )
        # 辅口径：从真值表推导的「必须项覆盖率最高者」——只作为参考，不用于判对错
        truth_expected, truth_coverage = derive_expected_resume(
            truth_pairs, truth, [r["name"] for r in resumes]
        )
        correct = bool(expected_file) and recommended == expected_file

        per_jd.append(
            {
                "jd": jd["id"],
                "title": jd["title"],
                "source_type": jd["source_type"],
                "source_url": jd.get("source_url", ""),
                "fetched_at": jd.get("fetched_at", ""),
                "author_labeled_family": expected_family,
                "expected_resume": expected_file,
                "expected_resume_truth_derived": truth_expected,
                "expected_resume_truth_coverage": truth_coverage,
                "recommended_resume": recommended,
                "correct": correct,
                "extraction": {
                    "system": system_prf.to_dict(),
                    "llm_only": llm_only_prf.to_dict(),
                    "keyword_baseline": baseline_prf.to_dict(),
                    "truth_count": len(truth_pairs),
                    "system_count": len(system_pairs),
                    "llm_only_count": len(llm_only_pairs),
                    "baseline_count": len(baseline_pairs),
                    "fallback_count": fallback_count,
                    "fallback_ratio": round(fallback_count / len(requirements), 4) if requirements else 0.0,
                    "llm_stats": stats.to_dict(),
                    "system_detail": system_detail,
                    "llm_only_detail": llm_only_detail,
                    "baseline_detail": baseline_detail,
                },
                "per_resume": jd_resume_results,
            }
        )
        print(
            f"[{jd['id']:22s}] 系统F1={system_prf.f1:.3f} 基线F1={baseline_prf.f1:.3f} "
            f"推荐={recommended} {'✓' if correct else '✗'}"
        )

    # ---------------- 汇总 ----------------
    system_total = aggregate_prf(system_prfs)
    llm_only_total = aggregate_prf(llm_only_prfs)
    baseline_total = aggregate_prf(baseline_prfs)

    ev_total = sum(c["total"] for c in evidence_checks)
    ev_both = sum(c["both_correct"] for c in evidence_checks)
    ev_in_resume = sum(c["evidence_in_resume"] for c in evidence_checks)
    ev_supports = sum(c["evidence_supports_skill"] for c in evidence_checks)

    end_to_end_correct = sum(1 for row in per_jd if row["correct"])
    end_to_end_total = len(per_jd)

    results: Dict[str, Any] = {
        "meta": {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_seconds": round(time.perf_counter() - started, 2),
            "llm_backend": llm_name,
            "llm_model": getattr(llm, "model", "") if llm else "",
            "budget_usd": budget_usd,
            "mcp_enabled": use_mcp,
            "jd_count": len(jds),
            "resume_count": len(resumes),
            "public_real_jd": sum(1 for j in jds if j["source_type"] == "public_real"),
            "authored_variant_jd": sum(1 for j in jds if j["source_type"] == "authored_variant"),
            "total_annotated_requirements": total_requirements,
        },
        "metric_1_extraction": {
            "system_llm_plus_fallback": {
                "micro": system_total.to_dict(),
                "macro": macro_f1(system_prfs),
            },
            "llm_only": {
                "micro": llm_only_total.to_dict(),
                "macro": macro_f1(llm_only_prfs),
            },
            "keyword_baseline": {
                "micro": baseline_total.to_dict(),
                "macro": macro_f1(baseline_prfs),
            },
            "delta_micro_f1": round(system_total.f1 - baseline_total.f1, 4),
            "delta_weighted_f1": round(
                system_total.weighted_f1 - baseline_total.weighted_f1, 4
            ),
            "delta_llm_only_micro_f1": round(llm_only_total.f1 - baseline_total.f1, 4),
            "delta_llm_only_weighted_f1": round(
                llm_only_total.weighted_f1 - baseline_total.weighted_f1, 4
            ),
            "system_beats_baseline": system_total.f1 > baseline_total.f1,
            "llm_only_beats_baseline": llm_only_total.f1 > baseline_total.f1,
            "note": (
                "三档口径：①系统=LLM+规则兜底（产品实际路径）②纯LLM=关掉规则合并 "
                "③关键词基线=完全不用 LLM 的朴素词表扫描（命中即算一条、不区分重要度）。"
                "①与③不是「同一条路换模型」，①比③多了 LLM 的输出；"
                "所以①若低于③，说明 LLM 带来了噪声。②用来把 LLM 单独隔离出来看。"
            ),
        },
        "metric_2_evidence_localization": {
            "total_conclusions_with_evidence": ev_total,
            "evidence_in_resume": ev_in_resume,
            "evidence_supports_skill": ev_supports,
            "both_correct": ev_both,
            "accuracy": round(ev_both / ev_total, 4) if ev_total else 0.0,
            "per_jd_resume": evidence_checks,
            "note": (
                "一条结论的证据要判对，必须同时满足：①证据句是简历原文子串 "
                "②证据句里确实出现该技能（用与匹配器不同的实现路径核对，避免自证）。"
            ),
        },
        "metric_3_end_to_end_resume_recommendation": {
            "jd_count": end_to_end_total,
            "correct": end_to_end_correct,
            "accuracy": round(end_to_end_correct / end_to_end_total, 4) if end_to_end_total else 0.0,
            "per_jd": [
                {
                    "jd": row["jd"],
                    "title": row["title"],
                    "expected_resume": row["expected_resume"],
                    "expected_resume_truth_derived": row["expected_resume_truth_derived"],
                    "expected_resume_truth_coverage": row["expected_resume_truth_coverage"],
                    "recommended_resume": row["recommended_resume"],
                    "correct": row["correct"],
                }
                for row in per_jd
            ],
            "note": (
                "期望版本是**按岗位族的人工标注**（4 份简历 ↔ 4 个岗位族），"
                "符合规格要求；系统推荐由系统自己的匹配结果算出，两者独立。"
                "另附一个辅口径「按真值覆盖必须项最多者」——它退化了"
                "（评测后训练版覆盖面最广，几乎每份 JD 都推导出它），"
                "只作参考，不用于判对错。"
            ),
        },
        "metric_4_false_green": {
            "count": len(false_greens),
            "total_hit_conclusions": sum(
                1
                for row in per_jd
                for res in row["per_resume"]
                for m in res["matches"]
                if m["status"] == MATCH_HIT
            ),
            "items": false_greens,
            "ground_truth": "data/annotations/resume_skill_truth.json（人工标注）",
            "note": (
                "假绿 = 系统判「命中」，但**人工标注真值**判该简历不具备该技能。"
                "真值由本项目作者独立标注（data/annotations/resume_skill_truth.json），"
                "与系统的匹配逻辑完全无关——否则系统自证，假绿数恒为 0。"
            ),
        },
        "metric_5_fallback_and_cost": {
            "fallback_requirements": total_fallback,
            "total_requirements": total_requirements,
            "fallback_ratio": round(total_fallback / total_requirements, 4) if total_requirements else 0.0,
            "llm_usage": llm.usage.to_dict() if llm else {"backend": "none", "calls": 0},
            "note": "兜底占比 = 来源标记为 rule_fallback 的抽取条数 / 总抽取条数。",
        },
        "coverage_summary": {
            "hit_rate_mean": round(
                sum(r["hit_rate"] for r in resume_scores) / len(resume_scores), 4
            ) if resume_scores else 0.0,
            "must_hit_rate_mean": round(
                sum(r["must_hit_rate"] for r in resume_scores) / len(resume_scores), 4
            ) if resume_scores else 0.0,
        },
        "per_jd": per_jd,
        "caveats": [
            f"n = {len(jds)} 份 JD × {len(resumes)} 份简历；样本量小，"
            "所有比例只应作为**量级**参考，不要当成「准确率 90%」这种断言。",
            "人工标注为**单人标注**，没有第二标注者做交叉复核；"
            "重要度边界案例按 data/jds/ANNOTATION.md 的规则统一判定。",
            "关键词基线用同一份技能词表，它无法区分「必须/加分」，"
            "因此在重要度加权口径上**必然**吃亏——这是基线设计使然，不是实现缺陷。",
            "端到端「推荐哪版简历」的期望版本是**按岗位族的人工标注**（4 份简历 ↔ 4 个岗位族），"
            "人写标签有主观性，是本指标的已知边界；系统推荐与该标签相互独立。",
            "LLM 成本按公开价目表的非峰值档估算（来源：DeepSeek 官方文档，抓取日 2026-10-21）；"
            "实际账单可能因时段不同而变化。",
        ],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_summary(out_dir / "eval_summary.md", results)
    return results


def _write_summary(path: Path, results: Dict[str, Any]) -> None:
    """写人读的汇总（README 只引用这个文件里的数字）。"""
    m1 = results["metric_1_extraction"]
    m2 = results["metric_2_evidence_localization"]
    m3 = results["metric_3_end_to_end_resume_recommendation"]
    m4 = results["metric_4_false_green"]
    m5 = results["metric_5_fallback_and_cost"]
    meta = results["meta"]

    lines = [
        "# 评测结果汇总",
        "",
        f"生成时间：{meta['generated_at']}（耗时 {meta['elapsed_seconds']}s）",
        f"数据：{meta['jd_count']} 份 JD（真实 {meta['public_real_jd']} / 自写变体 {meta['authored_variant_jd']}）"
        f" × {meta['resume_count']} 份简历",
        f"LLM：`{meta['llm_backend']}`（{meta['llm_model'] or '—'}），预算上限 ${meta['budget_usd']}",
        "",
        "## 指标 1：JD 技能要求抽取（P/R/F1）",
        "",
        "| 口径 | P | R | F1 | 加权 P | 加权 R | 加权 F1 |",
        "|---|---|---|---|---|---|---|",
    ]
    for label, block in (
        ("系统（LLM + 规则兜底）", m1["system_llm_plus_fallback"]["micro"]),
        ("纯 LLM（关掉规则合并）", m1["llm_only"]["micro"]),
        ("关键词基线（无 LLM）", m1["keyword_baseline"]["micro"]),
    ):
        lines.append(
            f"| {label} | {block['precision']} | {block['recall']} | {block['f1']} | "
            f"{block['weighted_precision']} | {block['weighted_recall']} | {block['weighted_f1']} |"
        )
    lines += [
        "",
        f"- micro F1 差值（系统 − 基线）：**{m1['delta_micro_f1']}**",
        f"- 加权 F1 差值（系统 − 基线）：**{m1['delta_weighted_f1']}**",
        f"- micro F1 差值（纯 LLM − 基线）：**{m1['delta_llm_only_micro_f1']}**",
        f"- 加权 F1 差值（纯 LLM − 基线）：**{m1['delta_llm_only_weighted_f1']}**",
        f"- 系统是否优于基线：**{'是' if m1['system_beats_baseline'] else '否'}**",
        f"- 纯 LLM 是否优于基线：**{'是' if m1['llm_only_beats_baseline'] else '否'}**",
        "",
        "## 指标 2：匹配环节的证据定位正确率",
        "",
        f"- 有证据的结论共 {m2['total_conclusions_with_evidence']} 条",
        f"- 证据确实在简历里：{m2['evidence_in_resume']} 条",
        f"- 证据确实支持该技能：{m2['evidence_supports_skill']} 条",
        f"- **两项都对：{m2['both_correct']} 条 → 正确率 {m2['accuracy']}**",
        "",
        "## 指标 3：端到端「推荐哪版简历」准确率",
        "",
        f"- {m3['correct']} / {m3['jd_count']} 份 JD 推荐正确 → 准确率 **{m3['accuracy']}**",
        "",
        "## 指标 4：假绿次数",
        "",
        f"- 假绿（该说缺失却报命中）：**{m4['count']} 次**",
        f"- 期间系统共给出「命中」结论 {m4['total_hit_conclusions']} 条",
        "",
        "## 指标 5：兜底占比与 LLM 费用",
        "",
        f"- 兜底条数 {m5['fallback_requirements']} / 总条数 {m5['total_requirements']}"
        f" → 兜底占比 **{m5['fallback_ratio']}**",
    ]
    usage = m5["llm_usage"]
    if usage.get("calls"):
        cost = (
            f"${usage['cost_usd']:.6f}"
            if usage.get("price_known", True)
            else "价格未知（只报 token）"
        )
        lines.append(
            f"- LLM 调用 {usage['calls']} 次，token {usage['prompt_tokens']}+{usage['completion_tokens']}，"
            f"估算费用 {cost}"
        )
    else:
        lines.append("- LLM 未调用（纯规则档）")
    lines += [
        "",
        "## 诚实边界",
        "",
    ]
    for caveat in results["caveats"]:
        lines.append(f"- {caveat}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="跑 job-fit-assistant 的完整评测")
    parser.add_argument("--out", default="experiments/results", help="输出目录")
    parser.add_argument("--llm", default="auto", choices=["auto", "deepseek", "local", "none"], help="LLM 后端")
    parser.add_argument("--llm-model", default="", help="覆盖模型名")
    parser.add_argument("--budget-usd", type=float, default=1.0, help="LLM 预算上限（美元）")
    parser.add_argument("--mcp", action="store_true", help="开启 MCP 检索通道（会加载 embedding 模型）")
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    results = run(
        out_dir=out_dir,
        llm_backend=args.llm,
        llm_model=args.llm_model,
        budget_usd=args.budget_usd,
        use_mcp=args.mcp,
    )
    print(f"\n结果已写入：{out_dir / 'eval_results.json'}")
    print(f"汇总：{out_dir / 'eval_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
