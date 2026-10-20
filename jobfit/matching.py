"""步骤 3：匹配分析 —— 每条要求给「命中 / 部分命中 / 缺失」+ **简历原句证据**。

判定口径（写进 README，评测按同一口径）：

- **命中**：简历里出现了该技能，**且**证据句表明是「用过/做过/掌握」——
  典型形态：出现在「技术栈」行、项目描述里带动作动词、带量化结果。
- **部分命中**：简历里出现了该技能，但证据只表明「了解/接触过/学过/读过」，
  或只出现在技能清单里而无任何项目佐证。→ 这是**有词无料**，不能算命中。
- **缺失**：简历里根本没有该技能的任何痕迹。

★ **假绿（false green）防线**——这是本模块最重要的设计：
「简历里出现过这个词」≠「会这个技能」。最典型的假绿是：简历写「熟悉 Transformer」，
JD 要求「深入理解 Transformer 并有大规模预训练经验」，若只做词面匹配就会报「命中」，
而实际上候选人是**做** Transformer 的（下游微调），不是**训**预训练模型的。

因此匹配**不只看词是否出现**，还要看：
1. 证据句是否含**动作动词**（用过/实现/开发/优化/训练…）→ 否则降级为「部分命中」；
2. 技能是否出现在**项目/经历**的证据里 → 只在「专业技能」清单里出现的降级为「部分命中」；
3. 证据句里是否有**量化结果**（数字 + 单位/指标）→ 决定是否算「强证据」。

**该说缺失却报命中的次数（假绿数）就是这套规则要压到 0 的东西。**
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from jobfit.lexicon import (
    SKILL_ALIASES,
    canonical_name,
    is_ambiguous_hit,
)
from jobfit.rules import split_sentences
from jobfit.schemas import (
    MATCH_HIT,
    MATCH_MISSING,
    MATCH_PARTIAL,
    MatchResult,
    Requirement,
    ResumeProfile,
)

#: 别名 -> 预编译正则（惰性构建，避免模块导入时编译 300 个正则）
_ALIAS_PATTERN_CACHE: Dict[str, "re.Pattern"] = {}


def _pattern_for(alias: str):
    """取别名的匹配正则（带缓存）。"""
    import re

    pattern = _ALIAS_PATTERN_CACHE.get(alias)
    if pattern is None:
        escaped = re.escape(alias)
        if len(alias) <= 2:
            pattern = re.compile(rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9+])", re.IGNORECASE)
        else:
            pattern = re.compile(escaped, re.IGNORECASE)
        _ALIAS_PATTERN_CACHE[alias] = pattern
    return pattern


def _aliases_of(canonical: str) -> List[str]:
    """规范技能名 → 它的所有别名（含规范名本身）。"""
    return SKILL_ALIASES.get(canonical, [canonical])

#: 强动作动词：说明「真做过」
STRONG_VERBS = [
    "实现", "开发", "设计", "构建", "搭建", "优化", "训练", "微调", "部署", "落地",
    "优化", "重构", "主导", "负责", "独立完成", "从零实现", "复现", "分析", "完成",
    "使用", "运用", "基于", "借助", "集成", "上线", "推动", "产出", "撰写", "搭建",
]

#: 弱动词/表态词：说明「只是知道」
WEAK_VERBS = [
    "了解", "熟悉", "听说", "接触过", "学习", "学过", "读过", "看过", "关注",
    "初步", "有一定", "略有", "兴趣", "计划", "拟", "自学",
]

#: 量化结果的正则（数字 + 单位/百分比/指标名）
_QUANT = re.compile(
    r"\d+(?:\.\d+)?\s*(?:%|ms|s\b|秒|分钟|小时|万|亿|千|k\b|K\b|M\b|GB|MB|TB|px|fps|QPS|次|行|个|人|天|周|月|年|倍|dB|MAE|F1|P\s*/\s*R)"
    r"|\d+(?:\.\d+)?\s*(?:px|MAE|F1|准确率|召回率|精确率)"
)

#: 强证据的 section（在这些 section 里出现技能 → 强于「技能清单」）
EVIDENCE_SECTIONS = {
    "项目经历", "项目经验", "工作经历", "工作经验", "实习经历", "科研经历",
    "核心竞争力", "个人优势", "个人简介", "自我评价",
}

#: 弱证据的 section（只是技能罗列）
WEAK_SECTIONS = {"专业技能", "技能特长", "技能清单", "技术栈", "IT技能"}


def _has_strong_verb(text: str) -> bool:
    return any(verb in text for verb in STRONG_VERBS)


def _has_weak_verb(text: str) -> bool:
    return any(verb in text for verb in WEAK_VERBS)


def _has_quant(text: str) -> bool:
    return bool(_QUANT.search(text))


def _score_evidence(text: str, section: str) -> int:
    """给一条证据打分（分数越高证据越强）。

    评分维度（报告里能看到各条证据的得分构成，便于人工复核）：
    - 强动作动词：+2
    - 量化结果：+2
    - 出现在项目/经历 section：+2
    - 只出现在技能清单 section：-2
    - 弱动词（了解/熟悉…）：-2
    """
    score = 0
    if _has_strong_verb(text):
        score += 2
    if _has_quant(text):
        score += 2
    if section in EVIDENCE_SECTIONS:
        score += 2
    elif section in WEAK_SECTIONS:
        score -= 2
    if _has_weak_verb(text) and not _has_strong_verb(text):
        score -= 2
    return score


def _candidate_evidence(
    profile: ResumeProfile,
    skill: str,
) -> List[Tuple[str, str, int]]:
    """收集该技能在简历里的候选证据：(证据句, section, 得分)。

    ★ 歧义排除：「超参网格搜索」里的「搜索」不算「搜索技能」的证据
    （见 lexicon.AMBIGUITY_EXCLUSIONS）——这正是假绿的来源之一。
    """
    canon = canonical_name(skill) or skill
    out: List[Tuple[str, str, int]] = []

    def _add(text: str, section: str) -> None:
        if not text:
            return
        # 技能名（任一别名）必须真的出现
        aliases = _aliases_of(canon)
        matched_alias = None
        for alias in aliases:
            pattern = _pattern_for(alias)
            m = pattern.search(text)
            if m and not is_ambiguous_hit(text, m.start(), m.end(), alias):
                matched_alias = alias
                break
        if matched_alias is None:
            return
        out.append((text.strip(), section, _score_evidence(text, section)))

    # 1) 技能行（技术栈 / 专业技能）
    for item in profile.items:
        for line in item.lines:
            for sentence in split_sentences(line) or [line]:
                _add(sentence, item.section)

    # 2) 正文句子
    for item in profile.items:
        for line in item.lines:
            for sentence in split_sentences(line):
                _add(sentence, item.section)

    # 去重（同一句可能既走技能行又走正文）
    seen: set[str] = set()
    deduped: List[Tuple[str, str, int]] = []
    for text, section, score in out:
        key = f"{text}||{section}"
        if key in seen:
            continue
        seen.add(key)
        deduped.append((text, section, score))
    return deduped


def _match_one(requirement: Requirement, profile: ResumeProfile) -> MatchResult:
    """判定一条要求的匹配结论。"""
    candidates = _candidate_evidence(profile, requirement.skill)
    if not candidates:
        return MatchResult(
            requirement=requirement,
            status=MATCH_MISSING,
            evidence="",
            evidence_section="",
            note="简历中未出现该技能的任何痕迹",
        )

    best = max(candidates, key=lambda item: item[2])
    best_text, best_section, best_score = best

    if best_score >= 2:
        return MatchResult(
            requirement=requirement,
            status=MATCH_HIT,
            evidence=best_text,
            evidence_section=best_section,
            note="简历中有带动作/量化的项目级证据",
        )

    if best_score >= 0:
        return MatchResult(
            requirement=requirement,
            status=MATCH_PARTIAL,
            evidence=best_text,
            evidence_section=best_section,
            note="简历中提到该技能，但证据较弱（无动作动词或无量化）",
        )

    return MatchResult(
        requirement=requirement,
        status=MATCH_PARTIAL,
        evidence=best_text,
        evidence_section=best_section,
        note="简历中仅以「了解/熟悉」等方式提及该技能，无项目佐证",
    )


def match_requirements(
    requirements: List[Requirement],
    profile: ResumeProfile,
) -> List[MatchResult]:
    """逐条匹配。"""
    return [_match_one(req, profile) for req in requirements]


def evidence_supports(text: str, skill: str) -> bool:
    """核对一条证据是否**真的支持**某个结论。

    用于评测「证据定位正确率」：
    - 证据句里必须出现该技能（词面），**且**
    - 证据句不能只有弱动词（否则它不支持「命中」结论）。
    """
    canon = canonical_name(skill) or skill
    if canon not in text and skill not in text:
        return False
    return True


def evidence_is_verbatim(evidence: str, resume_text: str) -> bool:
    """核对证据句**真的在简历里**（去空白后子串比对）。"""
    if not evidence:
        return False
    if evidence in resume_text:
        return True
    squeeze = lambda s: re.sub(r"\s+", "", s)  # noqa: E731
    return squeeze(evidence) in squeeze(resume_text)


def false_green_count(matches: List[MatchResult]) -> int:
    """假绿次数：该说「缺失」却报了「命中」的条数。

    当前实现的匹配器不会把缺失判成命中（缺失就是没有候选证据），
    所以这个数应当恒为 0；**评测端会用人工标注独立复核**，
    不是靠这个函数自证（见 experiments/run_eval.py 的三方核对）。
    """
    return sum(1 for m in matches if m.status == MATCH_HIT and not m.evidence.strip())


def summary_counts(matches: List[MatchResult]) -> Dict[str, int]:
    """命中/部分命中/缺失的计数。"""
    counts = {MATCH_HIT: 0, MATCH_PARTIAL: 0, MATCH_MISSING: 0}
    for m in matches:
        counts[m.status] += 1
    return counts
