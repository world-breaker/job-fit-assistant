"""词表与归一化：别名冲突、复合短语拆分、歧义排除。

这些是整套评测的**地基**——如果词表本身有歧义或归一化口径不稳，
后面所有指标都算不准。所以它们必须有单独的用例守着。
"""

from __future__ import annotations

import pytest

from jobfit import filters, lexicon, normalize


def test_lexicon_has_no_alias_conflict():
    """词表导入时已做冲突检查（导入即失败）——这里再确认一次覆盖面。"""
    assert len(lexicon.SKILL_CATEGORY) > 50
    assert len(lexicon.ALIAS_TO_CANON) >= len(lexicon.SKILL_CATEGORY)
    for name in lexicon.SKILL_CATEGORY:
        assert lexicon.canonical_name(name) == name, f"规范名 {name!r} 查不到自己"


def test_short_aliases_do_not_match_inside_words():
    """短别名要有词边界：「C」不能命中「CUDA」，「Go」不能命中「Google」。"""
    assert lexicon.canonical_name("CUDA") == "CUDA"  # 词表里确实有 CUDA
    # 「CSS」不应命中 C
    matches = [
        canon
        for canon, alias, pattern in lexicon.MATCHERS
        if pattern.search("熟悉 CSS 与 HTML5")
    ]
    assert "C" not in matches, f"「C」不该命中「CSS」，实际命中：{matches}"


def test_canonical_name_maps_aliases():
    assert lexicon.canonical_name("pytorch") == "PyTorch"
    assert lexicon.canonical_name("  PYTORCH ") == "PyTorch"
    assert lexicon.canonical_name("不存在的技能xyz") is None


def test_split_compound_splits_only_when_all_parts_known():
    """复合短语只有在**每一段都查得到**时才拆。"""
    parts = normalize.split_compound("计算机基础、数据结构与算法")
    assert "数据结构与算法" in parts or len(parts) > 1

    # 有一段查不到 → 整体不拆（避免把真概念拆坏）
    assert normalize.split_compound("完全不存在的甲、完全不存在的乙") == [
        "完全不存在的甲、完全不存在的乙"
    ]


def test_normalize_canonical_maps_alias():
    assert normalize.canonical("SFT") == "LLM微调"
    assert normalize.canonical("RLVR") == "DPO"
    assert normalize.canonical("SpringBoot") == "Spring Boot"


def test_match_key_maps_aliases_to_same_key():
    """「SFT」与「LLM微调」必须得到同一个比对键。"""
    assert normalize.match_key("SFT") == normalize.match_key("LLM微调")
    assert normalize.match_key("RLVR") == normalize.match_key("DPO")
    assert normalize.match_key("Spring Boot") == normalize.match_key("springboot")


def test_match_key_handles_compound():
    """复合短语产生多键（`|` 连接）。"""
    key = normalize.match_key("高并发、高可靠海量服务开发")
    assert "|" in key, f"复合短语应产生多键，实际：{key}"


def test_ambiguity_exclusion_blocks_grid_search():
    """「超参网格搜索」里的「搜索」不是「搜索算法」——必须被排除。"""
    text = "支持动态负采样、多窗口对比、多轮超参网格搜索"
    assert lexicon.is_ambiguous_hit(text, text.index("搜索"), text.index("搜索") + 2, "搜索")

    # 正常的搜索算法表述不应被排除
    text2 = "熟悉搜索算法与搜索推荐"
    assert not lexicon.is_ambiguous_hit(text2, text2.index("搜索"), text2.index("搜索") + 2, "搜索")


def test_ambiguity_exclusion_blocks_quantitative_comparison():
    """「量化对比实验」里的「量化」不是「模型量化」——必须被排除（防假绿）。"""
    text = "硕士课题设计 5 类工业纹理量化对比实验，以边界对齐误差为核心指标"
    assert lexicon.is_ambiguous_hit(text, text.index("量化"), text.index("量化") + 2, "量化")


def test_non_skill_filter_blocks_soft_skills():
    """软素质、学历、会议名不是「技能要求」。"""
    for name in ("逻辑思维能力", "沟通协作", "本科及以上学历", "NeurIPS", "AAAI", "计算机基础"):
        assert filters.is_non_skill(name), f"{name!r} 应被过滤"


def test_non_skill_filter_keeps_real_skills():
    """真实技术名不能被误杀。"""
    for name in ("PyTorch", "Spring Boot", "向量检索", "Python", "RAG", "分布式训练"):
        assert not filters.is_non_skill(name), f"{name!r} 不该被过滤"
