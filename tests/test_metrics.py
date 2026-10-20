"""评测指标（metrics）与数据集完整性。

★ 这些用例守的是「评测本身可不可信」：
数据标注与原文对不上、词表有冲突、真值表缺项——任何一条都会让
后面所有数字失去意义，所以要在 CI 里挡住。
"""

from __future__ import annotations

import json

import pytest

from jobfit.metrics import Prf, aggregate_prf, macro_f1, score_extraction


def test_score_extraction_basic():
    prf, detail = score_extraction(
        [("Python", "必须"), ("Java", "必须")],
        [("Python", "必须"), ("Rust", "必须")],
    )
    assert prf.tp == 1
    assert prf.fp == 1
    assert prf.fn == 1
    assert prf.precision == 0.5
    assert prf.recall == 0.5
    assert prf.f1 == 0.5
    kinds = {d["kind"] for d in detail}
    assert kinds == {"命中", "误报", "漏报"}


def test_score_extraction_counts_aliases_as_same():
    """「SFT」与「LLM微调」算命中（归一化到词表规范名）。"""
    prf, _ = score_extraction([("SFT", "必须")], [("LLM微调", "必须")])
    assert prf.tp == 1, "别名应算命中"
    assert prf.fp == 0


def test_score_extraction_importance_weighting():
    """重要度加权：命中「必须」的权重是「加分」的两倍。"""
    must_prf, _ = score_extraction([("A", "必须")], [("A", "必须")])
    plus_prf, _ = score_extraction([("B", "加分")], [("B", "加分")])
    assert must_prf.weighted_tp == 2.0
    assert plus_prf.weighted_tp == 1.0


def test_empty_predictions_give_zero_not_crash():
    prf, _ = score_extraction([], [("Python", "必须")])
    assert prf.precision == 0.0
    assert prf.recall == 0.0
    assert prf.f1 == 0.0


def test_aggregate_and_macro():
    a, _ = score_extraction([("X", "必须")], [("X", "必须")])
    b, _ = score_extraction([("Y", "必须")], [("Y", "必须")])
    total = aggregate_prf([a, b])
    assert total.tp == 2
    macro = macro_f1([a, b])
    assert macro["f1"] == 1.0


# ------------------------------------------------------------------ 数据集

def test_all_jds_have_annotations(jd_dir):
    """每份 JD 都必须有人工标注，且标注非空。"""
    files = [p for p in jd_dir.glob("*.json") if p.name != "index.json"]
    assert len(files) == 12, f"应有 12 份 JD，实际 {len(files)}"
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        requirements = payload["annotations"]["requirements"]
        assert requirements, f"{path.name} 没有标注"
        for req in requirements:
            assert req["skill"] and req["importance"] and req["evidence"]


def test_jd_annotations_evidence_is_verbatim(jd_dir):
    """标注的证据必须是 JD 原文的子串（标注与原文脱节 → 评测失效）。"""
    import re

    for path in [p for p in jd_dir.glob("*.json") if p.name != "index.json"]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw = payload["raw_text"]
        squeeze = lambda s: re.sub(r"\s+", "", s)  # noqa: E731
        for req in payload["annotations"]["requirements"]:
            evidence = req["evidence"]
            assert evidence in raw or squeeze(evidence) in squeeze(raw), (
                f"{path.name} 的标注证据不在原文中：{evidence!r}"
            )


def test_jd_source_metadata_present(jd_dir):
    """真实 JD 必须有来源链接与抓取日期；自写变体必须明确标注。"""
    public = 0
    for path in [p for p in jd_dir.glob("*.json") if p.name != "index.json"]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["source_type"] == "public_real":
            public += 1
            assert payload["source_url"], f"{path.name} 缺来源链接"
            assert payload["fetched_at"], f"{path.name} 缺抓取日期"
        else:
            assert payload["source_type"] == "authored_variant"
    assert public >= 6, f"规格要求 ≥6 份公开真实 JD，实际 {public}"


def test_truth_table_covers_all_eval_skills(truth_table, jd_dir):
    """真值表要覆盖评测集里出现的全部技能。"""
    from jobfit.normalize import canonical

    skills: set[str] = set()
    for path in [p for p in jd_dir.glob("*.json") if p.name != "index.json"]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for req in payload["annotations"]["requirements"]:
            skills.add(canonical(req["skill"]) or req["skill"])
    for resume, table in truth_table["resumes"].items():
        missing = skills - set(table)
        assert not missing, f"{resume} 真值表缺少技能：{sorted(missing)}"


def test_resumes_are_anonymized(resume_dir):
    """入库简历不得含姓名/手机/邮箱/薪资。

    ★ 敏感串**不在测试里硬编码**——测试文件本身要进版本库，
    硬编码真实个人信息等于在仓库里留一份明文。这里改成**结构化检查**：
    断言简历里没有「手机号形态」「邮箱形态」「薪资行」这些**模式**，
    而不是比对具体字符串。
    """
    import re

    phone = re.compile(r"(?<!\d)1[3-9]\d[\s-]?\d{4}[\s-]?\d{4}(?!\d)")
    email = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
    salary = re.compile(r"(期望薪资|薪资期望|期望待遇)\s*[:：]")

    for path in resume_dir.glob("*.md"):
        text = path.read_text(encoding="utf-8")
        assert not phone.search(text), f"{path.name} 含手机号形态"
        assert not email.search(text), f"{path.name} 含邮箱形态"
        assert not salary.search(text), f"{path.name} 含薪资行"
        # 姓名行必须是占位符，不能是中文人名
        first = text.splitlines()[0].strip()
        assert first == "候选人", f"{path.name} 首行应为「候选人」，实际：{first!r}"


def test_resumes_have_candidate_placeholder(resume_dir):
    """脱敏后姓名应替换为「候选人」。"""
    for path in resume_dir.glob("*.md"):
        text = path.read_text(encoding="utf-8")
        assert "候选人" in text, f"{path.name} 没有「候选人」占位"
