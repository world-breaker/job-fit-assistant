"""建议 / 改写 / 面试题（步骤 4-6）。"""

from __future__ import annotations

from jobfit.advice import build_gaps, coverage_summary
from jobfit.interview import build_questions
from jobfit.rewrite import build_rewrites
from jobfit.resume import parse_resume
from jobfit.schemas import (
    IMPORTANCE_MUST,
    IMPORTANCE_PLUS,
    MATCH_HIT,
    MATCH_MISSING,
    MATCH_PARTIAL,
    MatchResult,
    Requirement,
)

RESUME = """候选人
项目经历
NLP 词向量训练 | 独立开发\t2026.07
• 从零实现 Word2Vec 反向传播与 BERT/Qwen 微调
• 负责 AI 相关的开发工作
• 技术栈：Python、PyTorch、Transformers

专业技能
编程语言：Python（熟练）、SQL
"""


def _match(skill: str, status: str, importance: str = IMPORTANCE_MUST) -> MatchResult:
    return MatchResult(
        requirement=Requirement(
            skill=skill, category="算法", importance=importance, evidence="JD 原文示例句"
        ),
        status=status,
        evidence="技术栈：Python、PyTorch、Transformers" if status != MATCH_MISSING else "",
        evidence_section="项目经历",
        note="测试",
    )


def test_gaps_sorted_by_priority():
    """差距建议按「重要度 × 缺失度」排序。"""
    matches = [
        _match("加分项技能", MATCH_MISSING, IMPORTANCE_PLUS),
        _match("必须项技能", MATCH_MISSING, IMPORTANCE_MUST),
        _match("部分项技能", MATCH_PARTIAL, IMPORTANCE_MUST),
    ]
    gaps = build_gaps(matches)
    assert [g.skill for g in gaps] == ["必须项技能", "部分项技能", "加分项技能"]
    assert gaps[0].priority > gaps[1].priority > gaps[2].priority


def test_gap_actions_are_concrete_not_vacuous():
    """规格要求：建议必须是「具体动作」，不是「建议学 X」这种废话。"""
    gaps = build_gaps([_match("Kubernetes", MATCH_MISSING)])
    assert gaps
    action = gaps[0].action
    assert "Kubernetes" in action
    # 空洞的建议会缺少这些可执行要素
    assert any(word in action for word in ("补", "写", "加", "改", "练")), action
    assert gaps[0].rationale


def test_no_gap_for_hit():
    assert build_gaps([_match("Python", MATCH_HIT)]) == []


def test_coverage_summary():
    matches = [
        _match("A", MATCH_HIT),
        _match("B", MATCH_PARTIAL),
        _match("C", MATCH_MISSING),
    ]
    summary = coverage_summary(matches)
    assert summary["total"] == 3
    assert summary["hit_rate"] == round(1 / 3, 4)
    assert summary["must_hit_rate"] == round(1 / 3, 4)


def test_rewrite_suggests_for_missing_and_vague(tmp_path):
    """改写建议要同时覆盖「缺失项补写」与「含糊表述改写」。"""
    path = tmp_path / "r.md"
    path.write_text(RESUME, encoding="utf-8")
    profile = parse_resume(path)

    matches = [
        _match("Kubernetes", MATCH_MISSING),
        _match("Python", MATCH_HIT),
    ]
    rewrites = build_rewrites(matches, profile)
    assert rewrites
    # 缺失项 → 有 after，无 before
    missing_rw = [r for r in rewrites if r.skill == "Kubernetes"]
    assert missing_rw
    assert missing_rw[0].after and not missing_rw[0].before
    # 含糊表述 → 有 before/after
    vague = [r for r in rewrites if r.before]
    assert vague, "应识别出含糊表述并给改写示例"


def test_questions_prioritize_risky_spots():
    """面试题要优先问「简历上写得含糊/缺证据」的地方。"""
    matches = [
        _match("缺失必须项", MATCH_MISSING, IMPORTANCE_MUST),
        _match("部分命中项", MATCH_PARTIAL, IMPORTANCE_MUST),
        _match("命中项", MATCH_HIT),
    ]
    questions = build_questions(matches, None, max_questions=8)
    assert questions
    assert questions[0].target_skill == "缺失必须项"
    assert questions[0].risk == "高"
    assert any(q.target_skill == "命中项" for q in questions)


def test_questions_have_reason():
    matches = [_match("缺失项", MATCH_MISSING)]
    for question in build_questions(matches, None):
        assert question.question
        assert question.reason


def test_questions_respect_max():
    matches = [_match(f"技能{i}", MATCH_MISSING) for i in range(20)]
    assert len(build_questions(matches, None, max_questions=5)) <= 5
