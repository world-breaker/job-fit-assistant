"""步骤 6：预测面试题 —— 按要求类别生成，且**优先问简历上写得含糊/缺证据的地方**。

排序逻辑（这是本步骤的价值所在）：不是「按类别平均分配几道题」，而是
**把题目预算优先花在风险点上**：
1. 缺失的**必须项**（最高风险：整块不会）→ 问「你有类似经验吗 / 你打算怎么补」
2. **部分命中**的项（简历提了但没料，最容易被追问穿）→ 问「具体怎么做的 / 指标多少」
3. 简历里的**含糊表述**（写了但说不清）→ 问「这句具体指什么」
4. 命中的项 → 问深挖题（「为什么这么选」「如果重做会怎么改」）

每道题带 `risk`（风险等级）与 `reason`（为什么问它），让人知道这题为什么在这里。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from jobfit.schemas import (
    IMPORTANCE_MUST,
    MATCH_HIT,
    MATCH_MISSING,
    MATCH_PARTIAL,
    InterviewQuestion,
    MatchResult,
    ResumeProfile,
)

#: 缺失项的提问模板
MISSING_QUESTIONS: Dict[str, List[str]] = {
    "语言": [
        "JD 要求 {skill}，你的简历里没有直接体现。你有其他语言/项目可以迁移过来吗（比如用 {skill} 重写过某个模块）？",
    ],
    "框架": [
        "JD 要求 {skill}。如果现在让你用 {skill} 承接这个岗位的核心工作，第一个要解决的问题是什么？你会怎么在两周内补齐？",
    ],
    "工程": [
        "JD 要求 {skill}。你做过的最接近 {skill} 的工程实践是什么？当时规模多大、你怎么保证不出问题？",
    ],
    "算法": [
        "JD 要求 {skill}。你没有相关项目经验的话，能不能说一个你自学 {skill} 时做过的实验/复现，以及你从中得到的判断？",
    ],
    "业务": [
        "JD 要求 {skill} 方向。你在过往项目里有没有接触过类似的业务问题？你会怎么把现有经验迁移过来？",
    ],
}

#: 部分命中项的提问模板（最容易穿帮的地方）
PARTIAL_QUESTIONS: Dict[str, List[str]] = {
    "语言": [
        "简历里提到 {skill}，能具体讲讲你用它写过什么吗？代码规模多大、你怎么保证正确性？",
    ],
    "框架": [
        "简历里提到 {skill}——你用它做的是哪一步？是你自己搭的还是调用的现成能力？效果怎么衡量的？",
    ],
    "工程": [
        "简历里提到 {skill}，这个环节出了问题你当时是怎么定位的？有没有留下监控/日志？",
    ],
    "算法": [
        "简历里提到 {skill}，你的方案为什么优于基线？有没有做过消融或对比实验？",
    ],
    "业务": [
        "简历里提到 {skill}，当时业务上最难的判断是什么？你怎么权衡的？",
    ],
}

#: 命中项的深挖题
HIT_QUESTIONS: Dict[str, List[str]] = {
    "语言": ["你用 {skill} 写过最复杂的一段逻辑是什么？为什么那样设计？"],
    "框架": ["你用 {skill} 做过最关键的一次技术选型是什么？当时还有哪些选项、为什么放弃了别的？"],
    "工程": ["你负责的 {skill} 相关环节上线后出过什么问题？你怎么定位和改进的？"],
    "算法": ["你用 {skill} 做过效果最好的一次改进是什么？如果给你更多时间，你会怎么继续优化？"],
    "业务": ["你用 {skill} 解决过的业务问题里，ROI 最高的是哪个？为什么？"],
}

#: 风险等级排序
RISK_ORDER = {"高": 3, "中": 2, "低": 1}


def _pick_skill_in_line(line: str, skills: List[str]) -> Optional[str]:
    for candidate in sorted(skills, key=len, reverse=True):
        if candidate in line:
            return candidate
    return None


def build_questions(
    matches: List[MatchResult],
    profile: Optional[ResumeProfile] = None,
    max_questions: int = 8,
) -> List[InterviewQuestion]:
    """生成预测面试题，按风险排序。"""
    questions: List[InterviewQuestion] = []
    used: set[str] = set()

    def _add(q: str, category: str, skill: str, reason: str, risk: str, priority: float) -> None:
        key = q.strip()
        if key in used:
            return
        used.add(key)
        questions.append(
            InterviewQuestion(
                question=q,
                category=category,
                target_skill=skill,
                reason=reason,
                risk=risk,
                priority=priority,
            )
        )

    # ---- 1) 缺失的必须项（最高风险） ----
    for match in matches:
        if match.status != MATCH_MISSING or match.requirement.importance != IMPORTANCE_MUST:
            continue
        skill = match.requirement.skill
        template = MISSING_QUESTIONS.get(match.requirement.category, MISSING_QUESTIONS["算法"])[0]
        _add(
            template.format(skill=skill),
            match.requirement.category,
            skill,
            f"JD 必须项「{skill}」在简历中完全缺失，属于整块能力空白",
            "高",
            3.0,
        )

    # ---- 2) 部分命中的项（简历提了但没料，最易被追问穿） ----
    for match in matches:
        if match.status != MATCH_PARTIAL:
            continue
        skill = match.requirement.skill
        template = PARTIAL_QUESTIONS.get(match.requirement.category, PARTIAL_QUESTIONS["算法"])[0]
        risk = "高" if match.requirement.importance == IMPORTANCE_MUST else "中"
        _add(
            template.format(skill=skill),
            match.requirement.category,
            skill,
            f"简历提到「{skill}」但证据薄弱（{match.note}），面试官大概率追问细节",
            risk,
            2.5 if risk == "高" else 1.8,
        )

    # ---- 3) 简历里的含糊表述 ----
    if profile is not None:
        skills = sorted({s.skill for s in profile.skills}, key=len, reverse=True)
        for item in profile.items:
            for line in item.lines:
                if not _is_vague_line(line):
                    continue
                skill = _pick_skill_in_line(line, skills) or ""
                _add(
                    f"简历「{item.section}」里这句「{line.strip()[:40]}」——具体指什么？"
                    "你负责的边界是什么、结果怎么衡量？",
                    "表达",
                    skill,
                    "简历该句只有弱表述且无量化结果，是简历里最容易被问穿的地方",
                    "中",
                    1.6,
                )
                break  # 每个 section 最多挑一条含糊句，避免刷屏

    # ---- 4) 命中项的深挖题 ----
    for match in matches:
        if match.status != MATCH_HIT:
            continue
        skill = match.requirement.skill
        template = HIT_QUESTIONS.get(match.requirement.category, HIT_QUESTIONS["算法"])[0]
        _add(
            template.format(skill=skill),
            match.requirement.category,
            skill,
            f"简历有「{skill}」的证据，面试会往深里问设计与取舍",
            "低",
            1.0,
        )

    questions.sort(key=lambda q: -q.priority)
    return questions[:max_questions]


def _is_vague_line(line: str) -> bool:
    """与 rewrite.py 同口径的「含糊」判定（保持两处一致）。"""
    from jobfit.rewrite import _is_vague

    return _is_vague(line)
