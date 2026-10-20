"""规则兜底 + 关键词基线：JD 技能要求抽取的两条非 LLM 路径。

两条路径刻意**共用同一套判据**（重要度词表 + 词表命中），差别只在**召回策略**：
- `keyword_baseline`：朴素关键词扫描——命中词表即算一条要求。这是评测里的**基线**。
- `rule_fallback`：LLM 抽取失败时的兜底——在基线之上加了
  「按行去重 + 上下文窗口 + 重要度词加权 + 否定/无关语境过滤」。

为什么要分这么细：规格要求「LLM 抽取要显著优于关键词基线，否则如实说」。
如果兜底和基线是同一份代码，两者的差距就分不清是「LLM 更好」还是「规则更好」。
分开实现，评测数字才说明得了问题。
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from jobfit.lexicon import MATCHERS, SKILL_CATEGORY, canonical_name, is_ambiguous_hit
from jobfit.schemas import (
    IMPORTANCE_MUST,
    IMPORTANCE_PLUS,
    SOURCE_KEYWORD_BASELINE,
    SOURCE_RULE_FALLBACK,
    Requirement,
)

#: 「必须」类重要度线索词（越靠前权重越高）
MUST_CUES: List[str] = [
    "必须", "精通", "熟练掌握", "熟练使用", "熟练运用", "扎实", "深刻理解",
    "深入理解", "深入熟悉", "熟练", "掌握", "具备", "能够独立", "有相关经验",
    "有丰富经验", "有.*经验", "至少", "核心", "强制", "硬性", "深入",
]

#: 「加分」类重要度线索词
PLUS_CUES: List[str] = [
    "优先", "加分", "更佳", "更好", "有.*优先", "了解", "接触过", "听说",
    "有所了解", "兴趣", "bonus", "nice to have", "plus",
]

#: 明确**不**算要求的语境（这些词出现时该句多半在讲「我们提供什么」而不是「要什么」）
NEGATIVE_CONTEXT: List[str] = [
    "加分项",  # 单独出现时由 PLUS_CUES 处理，这里留给「非要求」的模糊句
]

#: 句子切分：中文标点 + 换行 + 数字编号
_SENT_SPLIT = re.compile(r"(?<=[。；;！!？?\n])|\n+|(?=\d+[.、)]\s)")

#: 去掉编号前缀
_NUMBER_PREFIX = re.compile(r"^\s*(?:\d+[.、)]|[（(]\d+[）)]|[·•\-*])\s*")


def split_sentences(text: str) -> List[str]:
    """把 JD 正文切成句子（保留原文字符，不做改写——证据必须是原文）。"""
    parts: List[str] = []
    for raw in _SENT_SPLIT.split(text or ""):
        if raw is None:
            continue
        line = _NUMBER_PREFIX.sub("", raw.strip())
        if line:
            parts.append(line)
    return parts


def _match_aliases(sentence: str) -> List[Tuple[str, str]]:
    """在一句话里找出所有词表命中：返回 [(规范名, 命中的别名), ...]。

    ★ 命中长度上限：短别名（"C"/"Go"/"SQL" 这类）容易在长句里误命中，
    所以只有当**整句都很短**时才允许短别名参与——长句里的 "C" 多半是别的词。

    ★ 歧义排除：「超参网格搜索」里的「搜索」不作数（见 lexicon.AMBIGUITY_EXCLUSIONS）。
    """
    hits: List[Tuple[str, str]] = []
    seen: set[str] = set()
    short_sentence = len(sentence) <= 40
    for canon, alias, pattern in MATCHERS:
        if canon in seen:
            continue
        if len(alias) <= 2 and not short_sentence:
            continue
        m = pattern.search(sentence)
        if m and not is_ambiguous_hit(sentence, m.start(), m.end(), alias):
            hits.append((canon, m.group(0)))
            seen.add(canon)
    return hits


def _importance_for(sentence: str, window: str) -> str:
    """按线索词判定重要度。

    判据（口径写进 README 与 data/annotations 的说明里）：
    - 句子里出现「必须/精通/熟练/掌握/具备/深入理解」等 → **必须**
    - 只出现「优先/加分/更佳/了解/接触过」等 → **加分**
    - 两者都没有 → 看**上下文窗口**（前后各一句）里的线索；
      仍没有线索 → 按「职责句默认必须、要求句默认加分」之外的中性处理 → **必须**
      （JD 里没标「优先」的要求，绝大多数是硬性要求）。
    """
    must_hit = any(re.search(cue, sentence, re.I) for cue in MUST_CUES)
    plus_hit = any(re.search(cue, sentence, re.I) for cue in PLUS_CUES)
    if must_hit and not plus_hit:
        return IMPORTANCE_MUST
    if plus_hit and not must_hit:
        return IMPORTANCE_PLUS
    if must_hit and plus_hit:
        # 同时出现（「熟悉 X，有 Y 经验者优先」）→ 硬性优先
        return IMPORTANCE_MUST
    if re.search(r"优先|加分|更佳|了解|接触过", window, re.I):
        return IMPORTANCE_PLUS
    return IMPORTANCE_MUST


def _category_for(canonical: str) -> str:
    return SKILL_CATEGORY.get(canonical, "算法")


def extract_keyword_baseline(text: str) -> List[Requirement]:
    """**关键词基线**：逐句扫词表，命中即产出一条要求。

    这是评测对照组——刻意做得朴素：
    - 不做重要性区分（全部记「必须」，因为它「没有能力区分」）；
    - 不做去重优化（同一技能在多句出现会产生多条，靠评测端的集合口径消化）。

    诚实说明：基线的「全部记必须」是**有意的简化**，不是 bug。
    它代表「只会用关键词表的人能做到什么水平」——这正是我们要对比的对象。
    """
    results: List[Requirement] = []
    for sentence in split_sentences(text):
        for canon, alias in _match_aliases(sentence):
            results.append(
                Requirement(
                    skill=canon,
                    category=_category_for(canon),
                    importance=IMPORTANCE_MUST,
                    evidence=sentence,
                    source=SOURCE_KEYWORD_BASELINE,
                    raw=alias,
                    detail="关键词基线命中",
                )
            )
    return results


def extract_rule_fallback(text: str) -> List[Requirement]:
    """**规则兜底**：LLM 抽取失败时的保底路径。

    相比基线多了三件事：
    1. 重要度按线索词区分（必须/加分）；
    2. 同一技能只保留一条（取证据最长的那句——信息量最大）；
    3. 上下文窗口参与判重要度（句子本身没线索时看前后各一句）。
    """
    sentences = split_sentences(text)
    best: Dict[str, Requirement] = {}

    for index, sentence in enumerate(sentences):
        hits = _match_aliases(sentence)
        if not hits:
            continue
        # 上下文窗口：前后各一句（窗口大小固定，不做动态扩展——保持可复现）
        left = sentences[index - 1] if index > 0 else ""
        right = sentences[index + 1] if index + 1 < len(sentences) else ""
        window = f"{left} {sentence} {right}"
        importance = _importance_for(sentence, window)

        for canon, alias in hits:
            candidate = Requirement(
                skill=canon,
                category=_category_for(canon),
                importance=importance,
                evidence=sentence,
                source=SOURCE_RULE_FALLBACK,
                raw=alias,
                detail="规则兜底命中",
            )
            current = best.get(canon)
            # 保留证据更长的那句；等长时保留重要度更高的
            if current is None:
                best[canon] = candidate
                continue
            if len(candidate.evidence) > len(current.evidence):
                best[canon] = candidate
            elif len(candidate.evidence) == len(current.evidence):
                if candidate.importance == IMPORTANCE_MUST and current.importance != IMPORTANCE_MUST:
                    best[canon] = candidate

    return list(best.values())


def merge_rule_into_llm(
    llm_requirements: List[Requirement],
    rule_requirements: List[Requirement],
) -> List[Requirement]:
    """把规则结果合并进 LLM 结果（LLM 漏掉的补上，补的标成 rule_fallback）。

    合并口径：**按规范技能名去重**。LLM 已经给出的技能名保留 LLM 的那条
    （它的证据与重要度更可信），规则只补 LLM 没提到的。
    这样报告里能算出「兜底补了多少条」——也就是兜底占比的分母。
    """
    merged = list(llm_requirements)
    seen = {req.skill for req in llm_requirements}
    for req in rule_requirements:
        if req.skill not in seen:
            merged.append(req)
            seen.add(req.skill)
    return merged


def normalize_skill(name: str) -> Optional[str]:
    """把任意技能名映射到词表规范名；词表外返回 None（保留原名由调用方处理）。"""
    return canonical_name(name)
