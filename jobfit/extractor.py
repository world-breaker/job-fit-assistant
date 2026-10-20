"""步骤 1：JD 技能要求抽取。

产物：`{技能, 类别(语言/框架/工程/算法/业务), 重要度(必须/加分), 原文证据}`。

抽取有两条路：
- **LLM 抽取**（主）：一次调用拿结构化列表，随后**逐条校验**：
  技能名归一化到词表、类别校验、重要度校验、**原文证据必须是 JD 原文的子串**
  （拿不到原句的要求直接丢弃——宁可少抽，也不要编证据）。
- **规则兜底**（备）：LLM 不可用/超时/解析失败/校验全灭时启用，产出的每条都带
  `source="rule_fallback"`，报告里单独统计占比。

**兜底不许混作一谈**：`source` 字段一路带到报告与 JSON 输出，评测端按它分组统计。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from jobfit.filters import is_non_skill
from jobfit.lexicon import CATEGORIES, SKILL_CATEGORY, canonical_name
from jobfit.llm import BaseLlm, LlmError, extract_json_array
from jobfit.normalize import canonical, clean, split_compound
from jobfit.rules import extract_rule_fallback, merge_rule_into_llm, split_sentences
from jobfit.schemas import (
    IMPORTANCE_MUST,
    IMPORTANCE_PLUS,
    SOURCE_LLM,
    SOURCE_RULE_FALLBACK,
    Requirement,
)

#: LLM 的抽取指令。强调三件事：只抽原文出现的、必须带原句证据、重要度只分两档。
EXTRACT_SYSTEM = (
    "你是资深技术招聘分析师。你的任务是从职位描述（JD）中抽取「技能要求」。"
    "严格遵守：\n"
    "1. 只抽取 JD 原文中明确出现的技能/技术要求，不要脑补、不要泛化；\n"
    "2. ★ **skill 字段必须从给定的候选技能清单里选**（原样使用清单里的名称）。"
    "只有当 JD 提到了清单里完全没有的具体技术时，才可以用清单外的名称；\n"
    "3. ★ **不要抽这些**：软素质（沟通/责任心/学习能力/抗压）、学历与年限要求、"
    "会议与期刊名（ICML/NeurIPS/KDD…）、泛泛的名词（工程能力/计算机基础/平台/体系）、"
    "纯业务描述；\n"
    "4. ★ 每条要求只对应**一个**技术名词，不要写复合短语"
    "（如「计算机基础、数据结构与算法」应写成「数据结构与算法」）；\n"
    "5. 每一条必须给出 evidence：JD 原文里**一字不差**的完整句子（可含开头的编号）；\n"
    "6. importance 只能取「必须」或「加分」：出现「必须/精通/熟练/掌握/具备/深入理解」"
    "等硬性表述 → 必须；只出现「优先/加分/更佳/了解/接触过」 → 加分；\n"
    "7. category 只能取：语言、框架、工程、算法、业务；\n"
    "8. 只输出 JSON 数组，不要任何解释文字。"
)

EXTRACT_PROMPT = """下面是职位描述原文。请抽取其中的技能要求，输出 JSON 数组，每项形如：
{{"skill": "单个技术名词（优先从候选清单里选）", "category": "语言|框架|工程|算法|业务", "importance": "必须|加分", "evidence": "JD 原文原句", "detail": "该要求的补充说明（可选）"}}

候选技能清单（skill 字段优先从这里选，原样使用清单里的名称）：
{candidates}

职位描述原文：
---
{jd_text}
---

只输出 JSON 数组。若原文没有任何技能要求，输出 []。"""


class ExtractionStats:
    """抽取过程的统计——报告要能看到「LLM 抽了多少、被校验丢了多少、兜底补了多少」。"""

    def __init__(self) -> None:
        self.llm_returned = 0
        self.llm_kept = 0
        self.rejected_unknown_skill = 0
        self.rejected_bad_importance = 0
        self.rejected_no_evidence = 0
        self.rejected_evidence_not_in_text = 0
        self.rejected_non_skill = 0
        self.fallback_used = False
        self.fallback_reason = ""
        self.merged_from_rules = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "llm_returned": self.llm_returned,
            "llm_kept": self.llm_kept,
            "rejected_unknown_skill": self.rejected_unknown_skill,
            "rejected_bad_importance": self.rejected_bad_importance,
            "rejected_no_evidence": self.rejected_no_evidence,
            "rejected_evidence_not_in_text": self.rejected_evidence_not_in_text,
            "rejected_non_skill": self.rejected_non_skill,
            "fallback_used": self.fallback_used,
            "fallback_reason": self.fallback_reason,
            "merged_from_rules": self.merged_from_rules,
        }


def _normalize_evidence(raw: str) -> str:
    """归一化证据文本：去掉首尾空白与编号前缀，便于子串校验。"""
    text = (raw or "").strip()
    text = re.sub(r"^\s*(?:\d+[.、)）]|[（(]\d+[）)]|[·•\-*])\s*", "", text)
    return text.strip()


def _evidence_in_text(evidence: str, jd_text: str) -> bool:
    """校验证据确实是 JD 原文的子串。

    先按归一化后的精确子串比；再试「去空白」比对（模型可能改了空格/换行）。
    两步都不通过 → 判为编造，丢弃。
    """
    if not evidence:
        return False
    if evidence in jd_text:
        return True
    squeeze = lambda s: re.sub(r"\s+", "", s)  # noqa: E731 —— 一行 lambda，图省事
    return squeeze(evidence) in squeeze(jd_text)


def _validate_items(
    items: List[Dict[str, Any]],
    jd_text: str,
    stats: ExtractionStats,
) -> List[Requirement]:
    """逐条校验 LLM 输出。返回保留下来的要求。"""
    kept: List[Requirement] = []
    for item in items:
        stats.llm_returned += 1
        raw_skill = str(item.get("skill") or "").strip()
        if not raw_skill:
            stats.rejected_unknown_skill += 1
            continue

        # ★ 归一化：把「计算机基础、数据结构与算法」这类复合短语拆成单个技术名词，
        #   并映射到词表规范名。不做这一步，评测端就没法把「LLM 抽的」和
        #   「人工标的」对上同一个东西（P/R/F1 会变成字符串巧合的统计）。
        parts = split_compound(raw_skill)
        resolved: List[str] = []
        for part in parts:
            canon = canonical(part)
            if canon:
                if canon not in resolved:
                    resolved.append(canon)
            elif part not in resolved:
                resolved.append(part)
        if not resolved:
            resolved = [clean(raw_skill)]

        category = str(item.get("category") or "").strip()
        if category not in CATEGORIES:
            fallback_cat = SKILL_CATEGORY.get(resolved[0])
            if fallback_cat is None:
                stats.rejected_unknown_skill += 1
                continue
            category = fallback_cat

        importance = str(item.get("importance") or "").strip()
        if importance not in (IMPORTANCE_MUST, IMPORTANCE_PLUS):
            stats.rejected_bad_importance += 1
            continue

        evidence = _normalize_evidence(str(item.get("evidence") or ""))
        if not evidence:
            stats.rejected_no_evidence += 1
            continue
        if not _evidence_in_text(evidence, jd_text):
            stats.rejected_evidence_not_in_text += 1
            continue

        detail = str(item.get("detail") or "").strip()
        for skill in resolved:
            # ★ 非技能过滤：LLM 会把「逻辑思维能力」「AAAI 会议发表」这类当成技能，
            #   不过滤会让 P/R/F1 的分母虚高、精度虚低（见 jobfit/filters.py）。
            if is_non_skill(skill):
                stats.rejected_non_skill += 1
                continue
            kept.append(
                Requirement(
                    skill=skill,
                    category=category,
                    importance=importance,
                    evidence=evidence,
                    source=SOURCE_LLM,
                    raw=raw_skill,
                    detail=detail,
                )
            )
            stats.llm_kept += 1

    # 同一技能多条 → 保留证据最长的那条（信息量最大）
    deduped: Dict[str, Requirement] = {}
    for req in kept:
        current = deduped.get(req.skill)
        if current is None or len(req.evidence) > len(current.evidence):
            deduped[req.skill] = req
    result = list(deduped.values())
    stats.llm_kept = len(result)
    return result


def candidate_skill_list() -> str:
    """给 LLM 的候选技能清单（按类别分组）。

    ★ 为什么要给清单：不给清单时，LLM 会自造大量词表外的名字
    （「AI Agent」「上下文管理」「监控告警」…），评测时这些要么对不上标注
    （误报），要么各说各话（同一个东西两个名字）。
    给了清单后，LLM 的任务变成「**在受控词表里做选择 + 判重要度**」——
    这既符合真实产品的做法（输出要能被下游匹配），也让 LLM 与基线的
    对比变得公平（同一份词表）。
    """
    from jobfit.lexicon import SKILL_CATEGORY

    by_category: Dict[str, List[str]] = {}
    for name, category in SKILL_CATEGORY.items():
        by_category.setdefault(category, []).append(name)
    lines: List[str] = []
    for category in ("语言", "框架", "工程", "算法", "业务"):
        names = sorted(by_category.get(category, []))
        if names:
            lines.append(f"- {category}：" + "、".join(names))
    return "\n".join(lines)


def extract_requirements(
    jd_text: str,
    llm: Optional[BaseLlm] = None,
    merge_rules: bool = True,
) -> Tuple[List[Requirement], ExtractionStats]:
    """抽取 JD 的技能要求。返回 (要求列表, 统计)。

    `merge_rules=True` 时，LLM 结果会与规则结果合并（LLM 漏掉的由规则补，
    补进来的标 `source="rule_fallback"`）。这条路径让「兜底占比」有意义：
    占比 = rule_fallback 条数 / 总条数。
    """
    stats = ExtractionStats()
    jd_text = jd_text or ""
    if not jd_text.strip():
        stats.fallback_used = True
        stats.fallback_reason = "JD 正文为空"
        return [], stats

    llm_requirements: List[Requirement] = []
    if llm is not None:
        try:
            prompt = EXTRACT_PROMPT.format(
                jd_text=jd_text, candidates=candidate_skill_list()
            )
            response = llm.complete(prompt, system=EXTRACT_SYSTEM)
            items = extract_json_array(response.text)
            llm_requirements = _validate_items(items, jd_text, stats)
        except LlmError as exc:
            stats.fallback_used = True
            stats.fallback_reason = f"LLM 抽取失败：{exc}"
        except Exception as exc:  # noqa: BLE001 —— 任何异常都必须落到兜底，不能让整条流水线断掉
            stats.fallback_used = True
            stats.fallback_reason = f"LLM 抽取异常：{type(exc).__name__}: {exc}"

    if not llm_requirements:
        # LLM 没给出任何可用结果（不可用 / 全被校验丢掉）→ 全量走规则兜底
        stats.fallback_used = True
        if not stats.fallback_reason:
            stats.fallback_reason = "LLM 未产出可用要求（空结果或全部未通过校验）"
        rule_only = extract_rule_fallback(jd_text)
        stats.merged_from_rules = len(rule_only)
        return rule_only, stats

    if merge_rules:
        rule_only = extract_rule_fallback(jd_text)
        before = len(llm_requirements)
        merged = merge_rule_into_llm(llm_requirements, rule_only)
        stats.merged_from_rules = len(merged) - before
        return merged, stats

    return llm_requirements, stats


def fallback_ratio(requirements: List[Requirement]) -> float:
    """兜底占比：来源为 rule_fallback 的条数 / 总条数。"""
    if not requirements:
        return 0.0
    fallback = sum(1 for r in requirements if r.source == SOURCE_RULE_FALLBACK)
    return round(fallback / len(requirements), 4)


def jd_sentences(text: str) -> List[str]:
    """暴露句子切分（供 JD 条款检索复用）。"""
    return split_sentences(text)
