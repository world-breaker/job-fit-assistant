"""JD 抽取（步骤 1）：LLM 路径、规则兜底、校验、来源标记。"""

from __future__ import annotations

import json

import pytest

from jobfit.extractor import extract_requirements, fallback_ratio
from jobfit.schemas import IMPORTANCE_MUST, IMPORTANCE_PLUS, SOURCE_LLM, SOURCE_RULE_FALLBACK

SAMPLE_JD = """1.负责电商核心系统从需求分析、方案设计到开发上线的完整交付；
2.参与电商业务模型和平台架构设计，支撑复杂交易场景及业务持续发展；
1.具备扎实的 Java 基础和良好的工程实践，能够编写结构清晰、易于维护的代码；
2.熟悉 Spring Boot、Spring Cloud、MySQL、Redis、消息队列等常用技术，具备数据库设计、性能优化及相关项目经验；
3.具备分布式系统设计和生产实践能力，能够保障系统稳定运行，并独立定位和解决线上问题；
4.具备电商或其他复杂交易系统的分析与建模能力，能够深入负责一个业务方向，并理解电商核心链路及相关上下游逻辑；
5.熟悉 Kafka 与大模型应用者优先；
"""


def test_rule_fallback_extracts_and_marks_source(fake_llm):
    """LLM 不可用 → 全走规则兜底，且每条都标 source=rule_fallback。"""
    requirements, stats = extract_requirements(SAMPLE_JD, llm=None)
    assert requirements, "规则兜底应能抽出技能"
    assert all(r.source == SOURCE_RULE_FALLBACK for r in requirements)
    assert stats.fallback_used is True
    assert fallback_ratio(requirements) == 1.0

    skills = {r.skill for r in requirements}
    assert "Java" in skills
    assert "Spring Boot" in skills
    assert "MySQL" in skills


def test_llm_path_is_used_and_marked(fake_llm):
    """LLM 成功 → LLM 条目标 source=llm，兜底补的标 rule_fallback。"""
    llm = fake_llm(
        [
            {
                "skill": "Java",
                "category": "语言",
                "importance": "必须",
                "evidence": "具备扎实的 Java 基础和良好的工程实践，能够编写结构清晰、易于维护的代码",
            },
            {
                "skill": "Kafka",
                "category": "框架",
                "importance": "加分",
                "evidence": "熟悉 Kafka 与大模型应用者优先",
            },
        ]
    )
    requirements, stats = extract_requirements(SAMPLE_JD, llm=llm)
    by_skill = {r.skill: r for r in requirements}

    assert by_skill["Java"].source == SOURCE_LLM
    assert by_skill["Java"].importance == IMPORTANCE_MUST
    assert by_skill["Kafka"].importance == IMPORTANCE_PLUS
    # 规则补进来的（Spring Boot 等）必须标成兜底
    assert by_skill["Spring Boot"].source == SOURCE_RULE_FALLBACK
    assert stats.fallback_used is False
    assert 0 < fallback_ratio(requirements) < 1


def test_llm_evidence_must_be_verbatim_in_jd(fake_llm):
    """LLM 编造的证据必须被丢弃（不许拿假证据糊弄）。"""
    llm = fake_llm(
        [
            {
                "skill": "Java",
                "category": "语言",
                "importance": "必须",
                "evidence": "这句话根本不在 JD 原文里",
            }
        ]
    )
    requirements, stats = extract_requirements(SAMPLE_JD, llm=llm)
    assert stats.rejected_evidence_not_in_text == 1
    # 这条被丢掉后，Java 由规则补进来（标兜底）
    java = [r for r in requirements if r.skill == "Java"]
    assert java and java[0].source == SOURCE_RULE_FALLBACK


def test_llm_bad_importance_is_rejected(fake_llm):
    """非法重要度 → 丢弃该条。"""
    llm = fake_llm(
        [
            {
                "skill": "Java",
                "category": "语言",
                "importance": "非常优先",
                "evidence": "具备扎实的 Java 基础和良好的工程实践",
            }
        ]
    )
    _, stats = extract_requirements(SAMPLE_JD, llm=llm)
    assert stats.rejected_bad_importance == 1


def test_llm_exception_falls_back_to_rules(fake_llm):
    """LLM 抛异常 → 整条走兜底，且**异常被记下来**（不静默）。"""
    llm = fake_llm(error=RuntimeError("boom"))
    requirements, stats = extract_requirements(SAMPLE_JD, llm=llm)
    assert stats.fallback_used is True
    assert "boom" in stats.fallback_reason
    assert all(r.source == SOURCE_RULE_FALLBACK for r in requirements)


def test_llm_non_skill_items_are_filtered(fake_llm):
    """软素质/学历/会议名要被非技能过滤器挡掉。"""
    llm = fake_llm(
        [
            {
                "skill": "逻辑思维能力",
                "category": "工程",
                "importance": "必须",
                "evidence": "具备扎实的 Java 基础和良好的工程实践",
            },
            {
                "skill": "Java",
                "category": "语言",
                "importance": "必须",
                "evidence": "具备扎实的 Java 基础和良好的工程实践",
            },
        ]
    )
    requirements, stats = extract_requirements(SAMPLE_JD, llm=llm)
    assert stats.rejected_non_skill == 1
    assert "逻辑思维能力" not in {r.skill for r in requirements}


def test_requirement_requires_evidence():
    """Requirement 不允许没有证据（dataclass 层校验）。"""
    from jobfit.schemas import Requirement

    with pytest.raises(ValueError):
        Requirement(skill="X", category="工程", importance="必须", evidence="")


def test_requirement_rejects_bad_importance():
    from jobfit.schemas import Requirement

    with pytest.raises(ValueError):
        Requirement(skill="X", category="工程", importance="随便", evidence="原句")
