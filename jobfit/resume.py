"""步骤 2：简历解析 → 结构化 `{技能, 项目/经历, 每项的原文证据}`。

支持三种输入：`.docx`（真读 XML，不靠「解压看看」）、`.md`、`.txt`。
解析出来的东西**每一项都带原文证据**——因为步骤 3 的「证据定位正确率」
要能核对「这句证据是不是真的在简历里」。

分段规则：简历里的**标题行**（个人优势 / 教育经历 / 项目经历 / 专业技能 / 资格证书…）
被识别成 section 名，其后的行归到该 section。识别不到标题的行走「其他」section。
段落内以「•」「·」「-」「*」开头的条目各自成行（一条一个经历点）。
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from jobfit.lexicon import MATCHERS, SKILL_CATEGORY, canonical_name, is_ambiguous_hit
from jobfit.rules import split_sentences
from jobfit.schemas import ResumeItem, ResumeProfile, ResumeSkill

#: 简历常见 section 标题（匹配时忽略空白与全角括号差异）
SECTION_HEADINGS = [
    "个人优势", "个人简介", "自我评价", "个人总结", "职业目标",
    "教育经历", "教育背景", "学习经历",
    "项目经历", "项目经验", "工作经历", "工作经验", "实习经历", "科研经历", "论文",
    "专业技能", "技能特长", "技能清单", "技术栈", "IT技能",
    "资格证书", "证书", "荣誉奖项", "获奖情况",
    "自我介绍", "基本信息",
]

#: 结尾像项目/经历标题的行（含日期区间或角色后缀）
_EXPERIENCE_TITLE = re.compile(
    r"(19|20)\d{2}\s*[-–—.年]\s*(19|20)?\d{2}"
    r"|\d{4}\.\d{1,2}\s*[-–—]\s*\d{4}\.\d{1,2}"
    r"|\d{4}\s*[-–—]\s*(19|20)?\d{2}\s*年?"
)

#: 条目前缀符号
_BULLET = re.compile(r"^\s*[•·●▪*\-–—]+\s*")

#: 技术栈行：`技术栈：Python、PyTorch` / `编程语言：Python（熟练）、Java`
_TECH_STACK = re.compile(r"^(技术栈|编程语言|语言|框架|工具|工具与平台|AI框架|后端|后端框架|工程工具|领域能力|其他语言|前端基础|模型评估|后训练|NLP方向|AI工程)\s*[:：]")


def _norm_heading(line: str) -> Optional[str]:
    """判断一行是否是 section 标题；是则返回标准 section 名。"""
    text = _BULLET.sub("", line).strip().strip("：:").strip()
    if not text or len(text) > 12:
        return None
    compact = re.sub(r"[\s（）()【】\[\]]", "", text)
    for heading in SECTION_HEADINGS:
        if compact == re.sub(r"[\s（）()【】\[\]]", "", heading):
            return heading
    return None


def _read_docx_text(path: Path) -> List[str]:
    """真读 .docx：解析 word/document.xml，提取段落文本。

    为什么不用「解压后 grep」：docx 的正文是 XML，段落之间由 <w:p> 分隔，
    直接读 XML 能拿到**段落边界**（简历的结构信息就在段落边界上）；
    纯文本拼接会把所有内容连成一行，段落结构全丢。
    """
    import xml.etree.ElementTree as ET

    ns = {
        "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    }
    with zipfile.ZipFile(path) as archive:
        try:
            xml_bytes = archive.read("word/document.xml")
        except KeyError as exc:
            raise ValueError(f"{path.name} 不是有效的 .docx（缺少 word/document.xml）") from exc
    root = ET.fromstring(xml_bytes)
    lines: List[str] = []
    for para in root.findall(".//w:p", ns):
        texts = [node.text or "" for node in para.findall(".//w:t", ns)]
        line = "".join(texts).strip()
        if line:
            lines.append(line)
    return lines


def _read_text_file(path: Path) -> List[str]:
    content = path.read_text(encoding="utf-8", errors="replace")
    return [line.rstrip() for line in content.splitlines()]


def _read_pdf_text(path: Path) -> List[str]:
    """PDF 只做「尽力而为」的文本抽取。

    注意：**PDF 不是本项目的入库格式**（入库只放 .md）。这里支持它是为了
    CLI 临时用；抽取质量取决于 PDF 内嵌字体，扫版 PDF 常常抽不出可读文本，
    抽不出就报错——不返回空壳结果让人误以为「简历里没有这些技能」。
    """
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError:
        try:
            from PyPDF2 import PdfReader  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "读取 PDF 需要 pypdf（或 PyPDF2）。本项目的入库格式是 .md，"
                "PDF 仅供临时查看；请改用 .md/.txt/.docx 输入。"
            ) from exc
    reader = PdfReader(str(path))
    lines: List[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        lines.extend(line.strip() for line in text.splitlines() if line.strip())
    if not lines:
        raise ValueError(f"{path.name} 未能抽取到文本（可能是扫描版 PDF）")
    return lines


def read_lines(path: str | Path) -> Tuple[List[str], str]:
    """读文件 → (行列表, 格式名)。"""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在：{file_path}")
    suffix = file_path.suffix.lower()
    if suffix == ".docx":
        return _read_docx_text(file_path), "docx"
    if suffix in (".md", ".markdown"):
        return _read_text_file(file_path), "md"
    if suffix == ".txt":
        return _read_text_file(file_path), "txt"
    if suffix == ".pdf":
        return _read_pdf_text(file_path), "pdf"
    # 未知后缀：按纯文本试一次
    return _read_text_file(file_path), "txt"


def _split_items(section: str, lines: List[str]) -> List[ResumeItem]:
    """把一个 section 的行拆成经历块。

    规则：出现日期区间的行 → 新块的标题；其后的 bullet 行 → 该块的内容。
    没有日期的普通行 → 归到前一个块（作为描述）或自成一块。
    """
    items: List[ResumeItem] = []
    current: Optional[ResumeItem] = None

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        bullet = bool(_BULLET.match(line))
        line = _BULLET.sub("", line).strip()
        if not line:
            continue

        is_title = bool(_EXPERIENCE_TITLE.search(line)) and len(line) < 120
        if is_title or (not bullet and not current):
            current = ResumeItem(title=line, section=section, lines=[])
            items.append(current)
            continue
        if current is None:
            current = ResumeItem(title="", section=section, lines=[])
            items.append(current)
        current.lines.append(line)

    return items


def parse_resume(path: str | Path, source_path_label: str = "") -> ResumeProfile:
    """解析简历 → 结构化结果。

    `source_path_label`：写进结果的路径标签。**默认留空**——
    提交物里不能出现绝对路径（评测结果要入库），调用方可以传相对路径。
    """
    lines, fmt = read_lines(path)
    profile = ResumeProfile(
        raw_lines=[line for line in lines if line.strip()],
        source_path=source_path_label,
        source_format=fmt,
    )

    # ---- 分段 ----
    sections: List[Tuple[str, List[str]]] = []
    current_name = "其他"
    current_lines: List[str] = []
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        heading = _norm_heading(line)
        if heading:
            if current_lines:
                sections.append((current_name, current_lines))
            current_name = heading
            current_lines = []
            continue
        current_lines.append(line)
    if current_lines:
        sections.append((current_name, current_lines))

    # ---- 经历块 ----
    for section_name, section_lines in sections:
        profile.items.extend(_split_items(section_name, section_lines))

    # ---- 技能（带证据） ----
    profile.skills = _extract_skills(profile)
    return profile


def _extract_skills(profile: ResumeProfile) -> List[ResumeSkill]:
    """从简历里抽技能，每项带原文证据句。

    证据来源有两种，优先级：
    1. 「技术栈：…」这类显式技能行 → 证据就是这一行（信息最明确）；
    2. 正文里出现技能名的句子 → 证据是那句话。
    """
    skills: Dict[str, ResumeSkill] = {}

    def _record(skill: str, evidence: str, section: str) -> None:
        canon = canonical_name(skill) or skill
        if canon not in SKILL_CATEGORY:
            return  # 词表外的词不当作「技能」（避免把「系统」「项目」当成技能）
        current = skills.get(canon)
        if current is None or len(evidence) > len(current.evidence):
            skills[canon] = ResumeSkill(
                skill=canon,
                category=SKILL_CATEGORY[canon],
                evidence=evidence.strip(),
                section=section,
            )

    # 1) 显式技能行
    for item in profile.items:
        for line in item.lines:
            if _TECH_STACK.match(line):
                for canon, _alias, pattern in MATCHERS:
                    if canon in skills:
                        continue
                    if pattern.search(line):
                        _record(canon, line, item.section)
            # 「技术栈：…」常常出现在项目块的最后一行，本身不带前缀
            if line.startswith("技术栈"):
                for canon, _alias, pattern in MATCHERS:
                    if canon in skills:
                        continue
                    if pattern.search(line):
                        _record(canon, line, item.section)

    # 2) 正文句子
    for item in profile.items:
        for line in item.lines:
            for sentence in split_sentences(line):
                if not sentence:
                    continue
                for canon, _alias, pattern in MATCHERS:
                    if canon in skills:
                        continue
                    m = pattern.search(sentence)
                    if m and not is_ambiguous_hit(sentence, m.start(), m.end(), _alias):
                        _record(canon, sentence, item.section)

    return list(skills.values())


def skill_evidence_map(profile: ResumeProfile) -> Dict[str, List[str]]:
    """技能 -> 所有证据句（评测「证据定位正确率」时用来核对）。"""
    out: Dict[str, List[str]] = {}
    for skill in profile.skills:
        out.setdefault(skill.skill, []).append(skill.evidence)
    return out
