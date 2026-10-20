"""步骤 5：简历改写建议 —— 针对缺失项给措辞/结构建议；对含糊表述给改写示例。

两类产出：
1. **缺失项补写建议**（结构建议）：告诉用户在简历哪个位置、按什么句式补哪一条。
2. **含糊表述改写示例**（措辞建议）：扫描简历里的「弱表述」——
   只有「了解/熟悉/参与/负责」而没有动作与结果的句子——给出 before/after 示例。

「含糊」的判定口径（与步骤 3 的假绿防线同源）：
- 命中弱动词（了解/熟悉/参与/协助/学习）**且**
- 没有强动作动词 **且**
- 没有量化结果
→ 这类句子在面试里撑不住，属于需要改写的目标。
"""

from __future__ import annotations

import re
from typing import List, Optional

from jobfit.lexicon import canonical_name
from jobfit.matching import _has_quant, _has_strong_verb
from jobfit.schemas import (
    IMPORTANCE_MUST,
    MATCH_HIT,
    MATCH_MISSING,
    MATCH_PARTIAL,
    MatchResult,
    ResumeProfile,
    RewriteSuggestion,
)

#: 改写句式模板（按类别）。{skill} 与 {jd_evidence} 会替换。
REWRITE_TEMPLATES = {
    "语言": (
        "在「专业技能」中把 {skill} 从名词罗列改成可验证描述，句式：「熟练使用 {skill}，"
        "在 X 项目中负责 Y 模块，支撑 Z 规模/性能」"
    ),
    "框架": (
        "在相关项目条目里补一句以 {skill} 为技术栈的动作句，句式：「基于 {skill} 实现 X，"
        "通过 Y 优化把 Z 从 A 降到 B」——缺 {skill} 的项目要显式点名它"
    ),
    "工程": (
        "把工程经历里的名词换成带规模的描述，句式：「负责 X 的 Y 环节，支撑 Z 并发/数据量/时延，"
        "通过 A 手段把 B 指标优化 C%」"
    ),
    "算法": (
        "把算法名词升级成方法级描述，句式：「用 {skill} 的 X 方法解决 Y 问题，"
        "相比基线 Z 提升 N%」——只写「用了 {skill}」无法支撑追问"
    ),
    "业务": (
        "给技术描述补上业务落点，句式：「用 X 解决 Y 业务问题（数据来源/规模），"
        "最终影响 Z 指标/用户量」"
    ),
}

#: 通用改写句式（不依赖类别）
GENERIC_REWRITE = (
    "把「{before}」改写成「{after}」：补上动作对象 + 量化结果，让这句话能被追问也答得出"
)


#: **具体动作动词**——说明「做了什么」，是简历该有的写法
SPECIFIC_VERBS = [
    "从零实现", "独立完成", "主导", "重构", "优化", "调优", "训练", "微调",
    "部署", "落地", "复现", "产出", "集成", "撰写", "搭建", "构建", "量化对比",
    "消融", "对齐", "蒸馏", "压缩",
]

#: **泛用动词**——出现不等于「做了事」，单独用它们撑不起一句简历
GENERIC_VERBS = [
    "开发", "设计", "使用", "运用", "基于", "负责", "参与", "协助", "完成", "做",
    "处理", "支持", "涉及",
]

#: 弱表态词——明确表示「只是知道/接触过」
WEAK_VERBS = [
    "了解", "熟悉", "听说", "接触过", "学习", "学过", "读过", "看过", "关注",
    "初步", "有一定", "略有", "兴趣", "计划", "拟", "自学",
]


def _is_vague(sentence: str) -> bool:
    """判定一句是否「含糊」。

    判据（三条，命中任一即算含糊）：
    1. 出现**弱表态词**（了解/熟悉/学过…）→ 只是知道，没有产出；
    2. 出现「负责/参与/协助」但**没有具体动作动词、也没有量化结果**
       → 说不清自己到底做了什么；
    3. 只有**泛用动词**（开发/设计/使用…）且没有具体动作与量化
       → 「负责 AI 相关的开发工作」这种什么都能套的句子。

    ★ 为什么第 3 条要单独列：早期版本把「开发」当成强动词，
      结果「负责 AI 相关的开发工作」被判为不含糊——而这句话在面试里
      什么都撑不住。这正是改写建议要处理的目标。
    """
    text = sentence.strip()
    if not text:
        return False
    if _has_quant(text) and _has_strong_verb(text):
        return False  # 有量化结果的句子不算含糊
    if any(verb in text for verb in WEAK_VERBS):
        return True
    has_specific = any(verb in text for verb in SPECIFIC_VERBS)
    if has_specific:
        return False
    if _has_quant(text):
        return False
    return any(verb in text for verb in GENERIC_VERBS)


def _rewrite_example(sentence: str, skill: Optional[str]) -> tuple[str, str]:
    """给一句含糊话生成 before/after 示例。"""
    trimmed = sentence.strip()
    if skill:
        after = (
            f"在 {skill} 相关工作中，具体负责「{trimmed[:40]}」这一环，"
            "输入是 X / 输出是 Y，最终把 Z 指标从 A 提升到 B"
        )
    else:
        after = (
            f"{trimmed[:40]}——补上你具体做的动作、用的工具、产出的结果数字"
        )
    return trimmed, after


def build_rewrites(
    matches: List[MatchResult],
    profile: ResumeProfile,
) -> List[RewriteSuggestion]:
    """生成改写建议：缺失项补写 + 含糊表述改写。"""
    suggestions: List[RewriteSuggestion] = []

    # ---- 1) 缺失/部分命中项的结构建议 ----
    for match in matches:
        if match.status == MATCH_HIT:
            continue
        skill = match.requirement.skill
        template = REWRITE_TEMPLATES.get(match.requirement.category, REWRITE_TEMPLATES["算法"])
        target = (
            "简历「专业技能」或对应项目条目的技术栈行"
            if match.status == MATCH_MISSING
            else "简历中提到该技能的那一条项目描述"
        )
        reason = (
            f"JD 要求「{skill}」（{match.requirement.importance}），当前{match.status}。"
            f"JD 原文：{match.requirement.evidence[:100]}"
        )
        suggestions.append(
            RewriteSuggestion(
                skill=skill,
                target=target,
                before="",
                after=template.format(skill=skill, jd_evidence=match.requirement.evidence[:80]),
                reason=reason,
                source="rule",
            )
        )

    # ---- 2) 含糊表述的改写示例 ----
    for item in profile.items:
        for line in item.lines:
            if not _is_vague(line):
                continue
            # 找出这句里提到的技能
            skill = None
            for candidate in sorted(
                {s.skill for s in profile.skills}, key=len, reverse=True
            ):
                if candidate in line:
                    skill = candidate
                    break
            before, after = _rewrite_example(line, skill)
            suggestions.append(
                RewriteSuggestion(
                    skill=skill or "",
                    target=f"简历「{item.section}」段落",
                    before=before,
                    after=after,
                    reason="该句只有弱表述（了解/熟悉/负责/参与）且无量化结果，面试里撑不住追问",
                    source="rule",
                )
            )

    return suggestions
