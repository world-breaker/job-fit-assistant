"""核对 README 里引用的数字与 experiments/results/ 下的实际文件是否一致。

★ 为什么要有这个脚本：README 的数字是「引用」结果文件的。
如果结果文件更新了而 README 没改，README 就在说谎——而且没人会发现。
这个脚本把两者对上，对不上就报错。

用法：
    python scripts/check_doc_numbers.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "experiments" / "results" / "eval_results.json"
SUMMARY = ROOT / "experiments" / "results" / "eval_summary.md"
README = ROOT / "README.md"

#: README 里必须出现的关键数字（键 -> 期望出现在 README 里的字符串）
#:
#: ★ 这些值**不是抄来的**，而是从 eval_results.json 里算出来后写进 README 的；
#:   本脚本反过来验证「README 里的值 == 结果文件里的值」。
def annotated_skill_count() -> Tuple[str, str]:
    """从 data/jds/*.json 的 annotations.requirements **当场数**出技能标注条数。

    ★ 为什么必须当场数：2026-10-21 实测，README 写「共 130 条人工标注」，
    而真实条数是 119（必须 108 / 加分 11）—— 数字得从数据里来，不能凭印象写。
    """
    jd_dir = ROOT / "data" / "jds"
    total = must = 0
    for f in sorted(jd_dir.glob("*.json")):
        if f.name == "index.json":
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        for r in d.get("annotations", {}).get("requirements", []):
            total += 1
            if r.get("importance") == "必须":
                must += 1
    return str(total), f"必须 {must} / 加分 {total - must}"


def expected_numbers() -> List[Tuple[str, str]]:
    if not RESULTS.exists():
        return []
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    m1 = data["metric_1_extraction"]
    m2 = data["metric_2_evidence_localization"]
    m3 = data["metric_3_end_to_end_resume_recommendation"]
    m4 = data["metric_4_false_green"]
    m5 = data["metric_5_fallback_and_cost"]

    usage = m5.get("llm_usage") or {}
    cost = f"{usage.get('cost_usd', 0.0):.4f}"

    return [
        ("技能标注条数", annotated_skill_count()[0]),
        ("技能标注构成", annotated_skill_count()[1]),
        ("系统 micro F1", f"{m1['system_llm_plus_fallback']['micro']['f1']:.4f}"),
        ("纯LLM micro F1", f"{m1['llm_only']['micro']['f1']:.4f}"),
        ("基线 micro F1", f"{m1['keyword_baseline']['micro']['f1']:.4f}"),
        ("系统 加权F1", f"{m1['system_llm_plus_fallback']['micro']['weighted_f1']:.4f}"),
        ("基线 加权F1", f"{m1['keyword_baseline']['micro']['weighted_f1']:.4f}"),
        ("F1 差值", f"{m1['delta_micro_f1']:+.4f}"),
        ("证据总数", str(m2["total_conclusions_with_evidence"])),
        ("证据正确数", str(m2["both_correct"])),
        ("端到端正确", f"{m3['correct']} / {m3['jd_count']}"),
        ("假绿次数", str(m4["count"])),
        ("命中结论总数", str(m4["total_hit_conclusions"])),
        ("兜底条数", str(m5["fallback_requirements"])),
        ("总条数", str(m5["total_requirements"])),
        ("兜底占比", f"{m5['fallback_ratio'] * 100:.1f}%"),
        ("LLM 费用", f"${cost}"),
        ("LLM token 输入", str(usage.get("prompt_tokens", 0))),
        ("LLM token 输出", str(usage.get("completion_tokens", 0))),
    ]


#: **带语境的关键断言**：只验「README 里含有这个数」是不够的 ——
#:   2026-10-21 实测踩过：README 写「124 条「命中」结论」，而结果文件是 123；
#:   旧校验只查「123 是否出现在 README 某处」（它在表格里出现过 → 蒙混过关），
#:   于是**错误数字被放行**。所以对这几个关键数字改成**带语境的正则**：
#:   数字必须出现在它该在的那句话里。
CONTEXT_ASSERTIONS = [
    ("命中结论总数", r"(\d+)\s*条「?命中」?结论"),
    ("假绿次数", r"假绿[^。；\n]{0,6}?(\d+)\s*次"),
    # README 原文：`- **9 / 12** 份 JD 推荐正确（准确率 0.75）`（数字带 markdown 粗体）
    ("端到端正确", r"\*\*(\d+)\s*/\s*(\d+)\*\*\s*份\s*JD\s*推荐正确"),
    # README 原文：`- 两项都对：**151** 条 → **正确率 100%**`
    ("证据正确数", r"两项都对：\*\*(\d+)\*\*\s*条"),
    # README 原文：`| 系统（LLM + 规则兜底） | 0.7829 | 0.9182 | **0.8452** | ...`
    #   → 表格里第 3 列才是 F1（第 1 列 P、第 2 列 R），所以显式跳过前两列再捕获第 3 列
    ("系统 micro F1", r"系统（LLM \+ 规则兜底）\s*\|\s*[0-9.]+\s*\|\s*[0-9.]+\s*\|\s*\*{0,2}([0-9.]+)"),
    # ★ 三个条数口径：README「指标 1」下的那张表，逐格校验（防止 119/110/129 再打架）
    #   README 原文：`| **人工标注原始条数** | **119** | ... |`
    ("人工标注原始条数", r"\*\*人工标注原始条数\*\*\s*\|\s*\*\*(\d+)\*\*"),
    ("计分用真值条数", r"\*\*计分用真值条数\*\*\s*\|\s*\*\*(\d+)\*\*"),
    ("系统抽取条数", r"\*\*系统抽取条数\*\*\s*\|\s*\*\*(\d+)\*\*"),
    # README 原文：`- 兜底条数 **39** / 系统抽取总条数 **129** → 兜底占比 **30.2%**`
    ("兜底条数与分母", r"兜底条数\s*\*\*(\d+)\*\*\s*/\s*系统抽取总条数\s*\*\*(\d+)\*\*"),
]


def context_failures(readme: str) -> List[str]:
    """带语境的数字断言：把 README 里对应句子里的数字抓出来，和结果文件比。"""
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    m1 = data["metric_1_extraction"]
    m2 = data["metric_2_evidence_localization"]
    m3 = data["metric_3_end_to_end_resume_recommendation"]
    m4 = data["metric_4_false_green"]
    m5 = data["metric_5_fallback_and_cost"]
    annotated_live, _ = annotated_skill_count()
    want = {
        "命中结论总数": [(str(m4["total_hit_conclusions"]),)],
        "假绿次数": [(str(m4["count"]),)],
        "端到端正确": [(str(m3["correct"]), str(m3["jd_count"]))],
        "证据正确数": [(str(m2["both_correct"]),)],
        "系统 micro F1": [(f"{m1['system_llm_plus_fallback']['micro']['f1']:.4f}",)],
        # ★ 三个条数口径（119 / 110 / 129）
        "人工标注原始条数": [(annotated_live,)],
        "计分用真值条数": [(str(m1["system_llm_plus_fallback"]["micro"]["tp"]
                              + m1["system_llm_plus_fallback"]["micro"]["fn"]),)],
        "系统抽取条数": [(str(m5["total_requirements"]),)],
        "兜底条数与分母": [(str(m5["fallback_requirements"]), str(m5["total_requirements"]))],
    }
    bad: List[str] = []
    for label, pattern in CONTEXT_ASSERTIONS:
        m = re.search(pattern, readme)
        if not m:
            bad.append(f"{label}: README 里找不到带语境的断言（正则 {pattern}）")
            continue
        got = tuple(g for g in m.groups() if g is not None)
        for expect in want.get(label, []):
            for e, g in zip(expect, got):
                if e != g:
                    bad.append(f"{label}: README 语境里写的是 {g}，结果文件是 {e}")
    # 顺便：系统 micro F1 的语境校验已由 CONTEXT_ASSERTIONS 里的同名词条覆盖
    #   （早期版本在这里另写了一份只抓 P 列的检查，会把 P 当成 F1 报错 → 已删除）
    return bad


def meta_guards() -> List[str]:
    """meta 里的三个条数口径必须自洽——防止再出现「129 条标注」和「119 条标注」并存。

    ★ 为什么要有这个守卫：2026-10-22 实测发现，`eval_results.json` 的
    `meta.total_annotated_requirements` 里装的是**系统抽取**条数（129），
    而 `data/jds` 里**人工标注**只有 119 条——同一个字段名，说的却是两件事，
    于是仓库里同时流通「129 条标注」和「119 条标注」两种说法，面试对不上。

    现在三个数各用各的名字，并且互相加得上：

    | 字段 | 值 | 交叉校验 |
    |---|---|---|
    | `total_annotated_requirements` | 119 | == data/jds 现场点出来的条数 |
    | `total_scored_truth_requirements` | 110 | == 指标 1 的 tp + fn（归一化去重后） |
    | `total_extracted_requirements` | 129 | == 指标 5 的 total_requirements == 指标 1 的 tp + fp |
    """
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    meta = data.get("meta", {})
    m1 = data["metric_1_extraction"]["system_llm_plus_fallback"]["micro"]
    m5 = data["metric_5_fallback_and_cost"]

    annotated_live, _ = annotated_skill_count()
    annotated = meta.get("total_annotated_requirements")
    extracted = meta.get("total_extracted_requirements")
    scored = meta.get("total_scored_truth_requirements")

    bad: List[str] = []
    if annotated != int(annotated_live):
        bad.append(
            f"meta.total_annotated_requirements={annotated}，"
            f"但 data/jds 里当场点出来的是 {annotated_live} 条"
        )
    if extracted != m5.get("total_requirements"):
        bad.append(
            f"meta.total_extracted_requirements={extracted}，"
            f"metric_5.total_requirements={m5.get('total_requirements')}"
        )
    if extracted != m1["tp"] + m1["fp"]:
        bad.append(
            f"meta.total_extracted_requirements={extracted}，"
            f"但指标 1 的 tp+fp={m1['tp'] + m1['fp']}（系统抽取条数对不上）"
        )
    if scored != m1["tp"] + m1["fn"]:
        bad.append(
            f"meta.total_scored_truth_requirements={scored}，"
            f"但指标 1 的 tp+fn={m1['tp'] + m1['fn']}（计分用真值条数对不上）"
        )
    if annotated is not None and scored is not None and int(annotated) < int(scored):
        bad.append(
            f"meta 里计分用真值 {scored} 条 > 人工标注 {annotated} 条——归一化去重不会变多，逻辑反了"
        )
    return bad


def demo_numbers() -> List[Tuple[str, str]]:
    """Demo 报告里的数字（README 的 demo 表格引用它们）。

    README 表格里用的是**简称**（如「AI工程」），所以这里也用简称比对——
    简称的唯一来源是文件名去掉「简历202607-」前缀，避免两处各写一套映射。
    """
    demo = ROOT / "experiments" / "results" / "demo" / "demo_run.json"
    if not demo.exists():
        return []
    data = json.loads(demo.read_text(encoding="utf-8"))
    out: List[Tuple[str, str]] = []
    for row in data.get("results", []):
        full = row["resume"]
        short = full.replace("简历202607-", "")
        out.append(
            (
                f"demo {short} 命中",
                f"| {short} | {row['hit']} | {row['partial']} | {row['missing']} | "
                f"{row['gaps']} | {row['rewrites']} | {row['questions']} |",
            )
        )
    return out


def main() -> int:
    if not README.exists():
        print("[错误] 找不到 README.md", file=sys.stderr)
        return 1
    readme = README.read_text(encoding="utf-8")
    numbers = expected_numbers() + demo_numbers()
    if not numbers:
        print("[错误] 找不到 eval_results.json，先跑 scripts/run_eval.py", file=sys.stderr)
        return 1

    missing: List[str] = []
    for label, value in numbers:
        if value not in readme:
            missing.append(f"{label}: {value}")

    # 带语境的关键断言（数字必须出现在它该在的那句话里）
    ctx_bad = context_failures(readme)

    # meta 三个条数口径的自洽性（119 / 110 / 129 别再打架）
    meta_bad = meta_guards()

    print(f"核对 {len(numbers)} 个数字 + {len(CONTEXT_ASSERTIONS)} 条语境断言"
          f" + 6 项 meta 自洽检查（来源：experiments/results/）")
    if missing or ctx_bad or meta_bad:
        if missing:
            print("\n❌ README 里没有这些值（或值已过期）：")
            for item in missing:
                print(f"   - {item}")
        if ctx_bad:
            print("\n❌ 带语境的数字对不上（同一个数出现在别处不算数）：")
            for item in ctx_bad:
                print(f"   - {item}")
        if meta_bad:
            print("\n❌ eval_results.json 的 meta 里三个条数口径不自洽：")
            for item in meta_bad:
                print(f"   - {item}")
        return 1

    print("✅ README 的数字与结果文件一致（含语境断言 + meta 三个条数口径自洽）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
