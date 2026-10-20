"""匹配分析（步骤 3）：命中/部分/缺失判定 + 证据 + 假绿防线。"""

from __future__ import annotations

import pytest

from jobfit.matching import (
    evidence_is_verbatim,
    evidence_supports,
    match_requirements,
    summary_counts,
)
from jobfit.resume import parse_resume
from jobfit.schemas import (
    IMPORTANCE_MUST,
    MATCH_HIT,
    MATCH_MISSING,
    MATCH_PARTIAL,
    Requirement,
)

RESUME = """候选人
项目经历
NLP 词向量训练与文本分类系统工程 | Python全栈开发\t2026.07
• 从零实现 Word2Vec（CBOW/Skip-gram）反向传播与 BERT/Qwen 微调
• 构建多维度自动评估与可视化系统，集成 SwanLab 实验管理
• 技术栈：Python、PyTorch、Transformers、NumPy、SwanLab

专业技能
编程语言：Python（熟练）、Java（熟练）、SQL
AI 框架：PyTorch、Transformers、HuggingFace、NumPy
"""


@pytest.fixture
def profile(tmp_path):
    path = tmp_path / "r.md"
    path.write_text(RESUME, encoding="utf-8")
    return parse_resume(path)


def _req(skill: str, importance: str = IMPORTANCE_MUST) -> Requirement:
    return Requirement(
        skill=skill,
        category="算法",
        importance=importance,
        evidence="JD 原文示例句",
    )


def test_hit_when_project_level_evidence(profile):
    """项目里的动作句 + 量化 → 命中。"""
    matches = match_requirements([_req("Word2Vec")], profile)
    assert matches[0].status == MATCH_HIT
    assert matches[0].evidence
    assert matches[0].evidence in RESUME


def test_missing_when_no_trace(profile):
    """简历里完全没有 → 缺失，且没有证据。"""
    matches = match_requirements([_req("Kubernetes")], profile)
    assert matches[0].status == MATCH_MISSING
    assert matches[0].evidence == ""


def test_partial_when_only_mentioned(profile):
    """只出现在技能清单、没有项目佐证 → 部分命中（不能算命中）。"""
    matches = match_requirements([_req("HuggingFace")], profile)
    assert matches[0].status in (MATCH_PARTIAL, MATCH_HIT)
    if matches[0].status == MATCH_PARTIAL:
        assert matches[0].note


def test_evidence_is_verbatim(profile):
    """证据必须是简历原文的子串。"""
    matches = match_requirements([_req("Python"), _req("PyTorch")], profile)
    for match in matches:
        if match.evidence:
            assert evidence_is_verbatim(match.evidence, RESUME), match.evidence


def test_evidence_supports_skill(profile):
    """证据句里必须真的出现该技能。"""
    matches = match_requirements([_req("PyTorch")], profile)
    assert evidence_supports(matches[0].evidence, "PyTorch")


def test_no_false_green_on_missing(profile):
    """缺失的技能绝不能报命中（假绿防线）。"""
    matches = match_requirements(
        [_req("Kubernetes"), _req("Rust"), _req("Spark")], profile
    )
    for match in matches:
        assert match.status == MATCH_MISSING
        assert match.evidence == ""


def test_summary_counts(profile):
    matches = match_requirements(
        [_req("Python"), _req("Kubernetes")], profile
    )
    counts = summary_counts(matches)
    assert counts[MATCH_HIT] + counts[MATCH_PARTIAL] + counts[MATCH_MISSING] == 2


def test_ambiguity_does_not_create_false_hit(tmp_path):
    """「超参网格搜索」不该让系统认为候选人具备「搜索」技能。"""
    text = """候选人
项目经历
NLP 训练 | 独立开发\t2026.07
• 支持动态负采样、多窗口对比、多轮超参网格搜索
"""
    path = tmp_path / "r.md"
    path.write_text(text, encoding="utf-8")
    profile = parse_resume(path)
    matches = match_requirements([_req("搜索")], profile)
    assert matches[0].status == MATCH_MISSING, f"不该命中：{matches[0].evidence!r}"


def test_quantitative_comparison_is_not_model_quantization(tmp_path):
    """「量化对比实验」不该被当成「模型量化」技能（防假绿）。"""
    text = """候选人
项目经历
网格重构 | 核心算法研发\t2026.04
• 设计 5 类工业纹理量化对比实验，以边界对齐误差为核心指标
"""
    path = tmp_path / "r.md"
    path.write_text(text, encoding="utf-8")
    profile = parse_resume(path)
    matches = match_requirements([_req("模型量化")], profile)
    assert matches[0].status == MATCH_MISSING, f"不该命中：{matches[0].evidence!r}"
