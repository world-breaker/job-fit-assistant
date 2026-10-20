"""简历解析（步骤 2）：分段、技能抽取、.docx 真读。"""

from __future__ import annotations

import pytest

from jobfit.resume import parse_resume, read_lines

SAMPLE_MD = """候选人
华中科技大学 软件工程 本硕 | 求职：AI平台/算法工程化工程师 | 武汉

个人优势
熟练掌握 Python、PyTorch 生态，具备 AI 算法工程化落地全流程能力。

教育经历
华中科技大学\t软件工程 硕士\t2022 - 2026

项目经历
三维激光刻蚀自适应网格重构与超分系统 | Python算法工程化\t2024.09 - 2026.04
• 使用 Python 开发高保真三维网格重构管线，单模型处理耗时 < 0.05s
• 技术栈：Python、PyTorch、NumPy、Open3D

专业技能
编程语言：Python（熟练）、Java（熟练）、SQL
"""


def test_parse_resume_extracts_sections_and_skills(tmp_path):
    path = tmp_path / "r.md"
    path.write_text(SAMPLE_MD, encoding="utf-8")

    profile = parse_resume(path)
    sections = {item.section for item in profile.items}
    assert "项目经历" in sections
    assert "专业技能" in sections

    skills = {s.skill for s in profile.skills}
    assert "Python" in skills
    assert "PyTorch" in skills


def test_every_skill_has_evidence(tmp_path):
    """每项技能都必须带原文证据（规格硬要求）。"""
    path = tmp_path / "r.md"
    path.write_text(SAMPLE_MD, encoding="utf-8")
    profile = parse_resume(path)
    for skill in profile.skills:
        assert skill.evidence.strip(), f"技能 {skill.skill} 缺证据"
        assert skill.evidence in SAMPLE_MD, f"证据不是原文：{skill.evidence!r}"


def test_tech_stack_line_is_evidence(tmp_path):
    """「技术栈：…」这一行本身就是最强的技能证据。"""
    path = tmp_path / "r.md"
    path.write_text(SAMPLE_MD, encoding="utf-8")
    profile = parse_resume(path)
    numpy = [s for s in profile.skills if s.skill == "NumPy"]
    assert numpy, "应识别出 NumPy"
    assert "技术栈" in numpy[0].evidence


def test_read_lines_supports_md(tmp_path):
    path = tmp_path / "r.md"
    path.write_text(SAMPLE_MD, encoding="utf-8")
    lines, fmt = read_lines(path)
    assert fmt == "md"
    assert any("个人优势" in line for line in lines)


def test_read_lines_supports_docx(tmp_path):
    """真读 .docx：段落边界要保住（这是 docx 相对 txt 的关键优势）。"""
    docx = pytest.importorskip("docx")
    document = docx.Document()
    document.add_paragraph("候选人")
    document.add_paragraph("项目经历")
    document.add_paragraph("• 使用 Python 开发高保真管线")
    document.add_paragraph("• 技术栈：Python、PyTorch")
    path = tmp_path / "r.docx"
    document.save(str(path))

    lines, fmt = read_lines(path)
    assert fmt == "docx"
    assert lines[0] == "候选人"
    assert "项目经历" in lines

    profile = parse_resume(path)
    assert "Python" in {s.skill for s in profile.skills}


def test_unknown_extension_falls_back_to_text(tmp_path):
    path = tmp_path / "r.weird"
    path.write_text(SAMPLE_MD, encoding="utf-8")
    lines, fmt = read_lines(path)
    assert fmt == "txt"
    assert lines


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        read_lines("/nonexistent/path/to/resume.md")
