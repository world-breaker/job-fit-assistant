"""步骤 4：差距与优先级建议 —— 按「重要度 × 缺失度」排序，给**具体动作**。

规格明确要求「给具体动作，不是『建议学 X』这种废话」。所以每条建议必须回答
「**接下来 3 天具体做什么**，做完之后简历上哪一行会变成什么样」。

动作来源有两层：
1. **动作模板库**（规则）：按技能类别给可执行的默认动作（写什么、补什么、怎么量化）。
2. **LLM 改写**（可选）：把模板动作结合这条 JD 的原句改写成更贴合的表述。
   LLM 不可用时用模板原文——模板本身就是完整可执行的动作，不是占位符。
"""

from __future__ import annotations

from typing import Dict, List, Optional

from jobfit.schemas import (
    IMPORTANCE_MUST,
    MATCH_HIT,
    MATCH_MISSING,
    MATCH_PARTIAL,
    GapAction,
    MatchResult,
)

#: 按类别给的动作模板。每条都要具体到「做什么 + 简历上补什么」。
CATEGORY_ACTIONS: Dict[str, List[str]] = {
    "语言": [
        "在简历「专业技能」里把 {skill} 的熟练度写成具体等级（如「熟练，独立完成过 X 万行/日均 X 次调用」），并对应到下面一个项目条目",
        "选简历里最接近的项目，补一条用 {skill} 写的、能被追问的最小实现（200 行以内也可以），把它写进项目描述第一行",
    ],
    "框架": [
        "在简历里补一条用 {skill} 做过的**可量化**产出（模型规模 / 数据量 / 精度提升 / 推理耗时），格式：「基于 {skill} 实现 X，达到 Y」",
        "把现有项目里隐含用了 {skill} 的地方显式写出来（现在多半只写了业务结果，没写技术栈），并在「技术栈」行补上 {skill}",
    ],
    "工程": [
        "在简历里补一条 {skill} 的**规模数字**：并发数 / 数据量 / 时延 / 部署次数 / 服务规模，没有数字的工程经历在面试里基本等于没写",
        "针对 {skill}，在项目描述里补「你负责哪一段」的边界（搭了什么、怎么上线、怎么监控），面试官只问自己做过的那段",
    ],
    "算法": [
        "在简历里补一条 {skill} 的**方法级描述**（用了哪个模型/损失/采样策略、为什么选它），并给出效果数字；只写「用 {skill} 做 X」无法支撑追问",
        "把 {skill} 对应到你已有的项目里，补一句「基线 → 改进 → 指标变化」的三段式描述（哪怕是课程/个人项目）",
    ],
    "业务": [
        "在简历里补一条 {skill} 的**业务落点**：数据从哪来、怎么用、影响什么指标或用户量；面试官关心的是业务判断不是工具名",
        "针对 {skill} 准备一个可讲 3 分钟的落地案例（背景 / 难点 / 取舍 / 结果），简历里留一句钩子让面试官问",
    ],
}

#: 部分命中（简历提过但没料）时的补强动作
PARTIAL_ACTIONS: Dict[str, List[str]] = {
    "语言": [
        "简历里已经出现 {skill}，但没有对应产出——补一条用 {skill} 独立完成的最小功能（工具/脚本/模块），并写清规模与结果",
    ],
    "框架": [
        "简历里已经出现 {skill}，但只在技能清单里——把某个真实项目里的对应环节重写出来，明确「用 {skill} 做了什么、指标如何」",
    ],
    "工程": [
        "简历里已经出现 {skill}，但没有数字——给这条经历补上规模/时延/并发等可验证数字，面试时能被追问也答得出",
    ],
    "算法": [
        "简历里已经出现 {skill}，但只写了名词——补上方法选择理由与效果对比，形成「基线→改进→指标」三段式",
    ],
    "业务": [
        "简历里已经出现 {skill}，但没写业务影响——补一句它影响了什么（用户量/时长/成本/准确率）",
    ],
}


def _priority(match: MatchResult) -> float:
    """优先级 = 重要度权重 × 缺失程度。

    - 必须项且缺失：3.0（最高优先）
    - 加分项且缺失：1.5
    - 必须项且部分命中：2.0
    - 加分项且部分命中：1.0
    """
    base = 2.0 if match.requirement.importance == IMPORTANCE_MUST else 1.0
    if match.status == MATCH_MISSING:
        factor = 1.5
    elif match.status == MATCH_PARTIAL:
        factor = 1.0
    else:
        factor = 0.0
    return round(base * factor, 3)


def build_gaps(matches: List[MatchResult]) -> List[GapAction]:
    """从匹配结果生成差距与建议（已按优先级排序）。"""
    gaps: List[GapAction] = []
    for match in matches:
        if match.status == MATCH_HIT:
            continue
        skill = match.requirement.skill
        category = match.requirement.category
        if match.status == MATCH_MISSING:
            templates = CATEGORY_ACTIONS.get(category, CATEGORY_ACTIONS["算法"])
        else:
            templates = PARTIAL_ACTIONS.get(category, PARTIAL_ACTIONS["算法"])
        action = templates[0].format(skill=skill)
        rationale = (
            f"JD 要求「{skill}」（{match.requirement.importance}），"
            f"简历中{match.status}。"
            f"JD 原文：{match.requirement.evidence[:120]}"
        )
        gaps.append(
            GapAction(
                skill=skill,
                importance=match.requirement.importance,
                priority=_priority(match),
                action=action,
                rationale=rationale,
                kind="gap" if match.status == MATCH_MISSING else "strengthen",
            )
        )
    gaps.sort(key=lambda g: (-g.priority, g.skill))
    return gaps


def alternative_actions(skill: str, category: str) -> List[str]:
    """给同一技能的其他候选动作（报告里列出来，让人有选择）。"""
    return [tpl.format(skill=skill) for tpl in CATEGORY_ACTIONS.get(category, [])]


def coverage_summary(matches: List[MatchResult]) -> Dict[str, float]:
    """覆盖率摘要：命中/总、部分命中/总、必须项命中率。"""
    total = len(matches)
    if total == 0:
        return {"total": 0, "hit_rate": 0.0, "partial_rate": 0.0, "missing_rate": 0.0, "must_hit_rate": 0.0}
    hits = sum(1 for m in matches if m.status == MATCH_HIT)
    partial = sum(1 for m in matches if m.status == MATCH_PARTIAL)
    missing = sum(1 for m in matches if m.status == MATCH_MISSING)
    must_items = [m for m in matches if m.requirement.importance == IMPORTANCE_MUST]
    must_hits = sum(1 for m in must_items if m.status == MATCH_HIT)
    return {
        "total": total,
        "hit_rate": round(hits / total, 4),
        "partial_rate": round(partial / total, 4),
        "missing_rate": round(missing / total, 4),
        "must_hit_rate": round(must_hits / len(must_items), 4) if must_items else 0.0,
    }
