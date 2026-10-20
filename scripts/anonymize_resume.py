"""简历脱敏：把真实简历变成可公开入库的 .md。

★ 为什么必须脱敏（README 里写清）：
本仓库要开源，而简历里的姓名、手机号、邮箱、薪资期望、目标公司、证件号
这些信息**足以直接定位到具体的人**。把它们提交上去等于把个人信息公开——
这既是对别人的不负责（作者本人的简历也是真实个人信息），也会让仓库
在招聘方眼里直接失去可信度。

脱敏口径（替换规则）：
| 类型 | 处理 | 例子 |
|---|---|---|
| 姓名 | 替换为「候选人」 | 张三 → 候选人 |
| 手机号 | 整段移除 | 138xxxxxxxx → （删除） |
| 邮箱 | 整段移除 | someone@example.com → （删除） |
| 身份证号 | 整段移除 | 18 位 → （删除） |
| 学号 | 整段移除 | （删除） |
| 期望薪资 | 整行移除 | 「期望薪资：XX-XXK」→ 删除该行 || 目标公司 | 移除 | 具体公司名 → （删除） |
| 毕业年份/学校 | 保留 | 「某大学 软件工程 硕士」→ 保留（招聘必需信息） |
| 项目经历 | 保留 | 项目名、技术栈、量化指标都保留（这是评测集的主体） |

★ 保留学校与项目是有意的：这些是**岗位匹配必需**的信息，抹掉会让简历失去
评测价值；姓名+联系方式+薪资这些才是「定位到人」的部分。

★ 本文件里**不写任何真实个人信息**（连示例都用虚构值）——
它要进版本库，本身不能成为泄露源。真实的敏感串只存在于**本机原始文件**里。

用法：
    python scripts/anonymize_resume.py --in <docx|md|txt> --out data/resumes/<name>.md
    python scripts/anonymize_resume.py --in-dir <dir> --out-dir data/resumes
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---- 替换规则（全部用正则，不写死具体人名，避免脚本本身泄露隐私） ----

#: 手机号（11 位中国手机号，允许带分隔符）
PHONE_RE = re.compile(r"(?<!\d)(1[3-9]\d[\s-]?\d{4}[\s-]?\d{4})(?!\d)")

#: 邮箱
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")

#: 身份证号（15 位或 18 位）
ID_CARD_RE = re.compile(r"(?<!\d)(\d{6}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx])(?!\d)")

#: 学号（常见格式：10-12 位数字，且前面有「学号」二字）
STUDENT_ID_RE = re.compile(r"(学号\s*[:：]?\s*)(\d{8,14})")

#: 薪资相关行（整行移除）。**这里写的是「要匹配的模式」，不是任何真实数据**——
#: 本文件要进版本库，本身不能成为泄露源。
SALARY_LINE_RE = re.compile(r"^\s*(期望薪资|薪资期望|期望待遇|薪资|待遇)\s*[:：].*$", re.MULTILINE)

#: 目标公司（在「求职：」「意向公司」「目标公司」等字段里）
TARGET_COMPANY_RE = re.compile(
    r"(求职|意向|目标|期望)(公司|企业|单位)?\s*[:：]\s*([^\n|｜]{2,30}?)(?=\s*[|｜\n]|$)"
)

#: 姓名（中文姓名，2-4 个汉字，出现在开头独立成行时）
NAME_LINE_RE = re.compile(r"^\s*([\u4e00-\u9fa5]{2,4})\s*$", re.MULTILINE)

#: 姓名（出现在「姓名：xxx」字段里）
NAME_FIELD_RE = re.compile(r"(姓\s*名)\s*[:：]\s*([\u4e00-\u9fa5]{2,4})")

#: 替换后的占位符
CANDIDATE = "候选人"
REDACTED = ""

#: 已知的目标公司关键词（公司名 → 删除）。用公司后缀来匹配，避免写死公司名。
_COMPANY_SUFFIXES = (
    "有限公司", "股份有限公司", "集团", "科技公司", "科技有限", "信息技术",
    "网络科技", "科技股份",
)


def _redact_target_company(text: str) -> Tuple[str, int]:
    """把「求职：<公司名>」里的公司名去掉，保留岗位方向。

    例：「求职：字节跳动 后端工程师」→「求职： 后端工程师」
    """
    count = 0

    def _replace(match: re.Match) -> str:
        nonlocal count
        field, company = match.group(1), match.group(3)
        # 只在「字段后面跟的是公司名」时才删；跟的是岗位方向就保留
        if any(suffix in company for suffix in _COMPANY_SUFFIXES) or len(company) <= 6:
            # 短名字也可能是岗位方向（"后端工程师" 6 字）→ 用岗位关键词判断
            if not any(kw in company for kw in ("工程师", "研发", "算法", "开发", "研究", "实习", "管培")):
                count += 1
                return f"{field}："
        return match.group(0)

    return TARGET_COMPANY_RE.sub(_replace, text), count


def anonymize_text(text: str) -> Tuple[str, Dict[str, int]]:
    """脱敏一段文本。返回 (脱敏后文本, 各规则命中次数)。"""
    stats: Dict[str, int] = {
        "phone": 0,
        "email": 0,
        "id_card": 0,
        "student_id": 0,
        "salary_line": 0,
        "target_company": 0,
        "name": 0,
    }

    # 顺序要紧：先删结构化字段（整行/整段），再删散落的联系方式
    text, n = SALARY_LINE_RE.subn("", text)
    stats["salary_line"] = n

    text, n = STUDENT_ID_RE.subn(r"\1" + REDACTED, text)
    stats["student_id"] = n

    text, n = ID_CARD_RE.subn(REDACTED, text)
    stats["id_card"] = n

    text, n = EMAIL_RE.subn(REDACTED, text)
    stats["email"] = n

    text, n = PHONE_RE.subn(REDACTED, text)
    stats["phone"] = n

    text, n = NAME_FIELD_RE.subn(r"\1：" + CANDIDATE, text)
    stats["name"] += n

    # 开头独立成行的中文姓名 → 候选人（只替换第一处，避免误伤「项目经历」等标题）
    replaced = False

    def _name_line(match: re.Match) -> str:
        nonlocal replaced
        candidate = match.group(1)
        # 常见的非姓名标题要排除
        if candidate in {
            "个人优势", "教育经历", "项目经历", "专业技能", "资格证书", "工作经历",
            "实习经历", "科研经历", "基本信息", "自我介绍", "技能特长", "获奖情况",
        }:
            return match.group(0)
        if replaced:
            return match.group(0)
        replaced = True
        stats["name"] += 1
        return CANDIDATE

    text = NAME_LINE_RE.sub(_name_line, text)

    text, n = _redact_target_company(text)
    stats["target_company"] = n

    # 清理脱敏后残留的空分隔符与空行
    # （删除联系方式后会留下「男 | 25岁 |  |」这种尾巴，要收拾干净）
    text = re.sub(r"[ \t]*\|\s*(?=[|\n])", " ", text)      # 去掉空的表格单元
    text = re.sub(r"[ \t]*\|\s*$", "", text, flags=re.MULTILINE)  # 去掉行尾的孤立竖线
    text = re.sub(r"^\s*\|\s*", "", text, flags=re.MULTILINE)     # 去掉行首的孤立竖线
    text = re.sub(r"[ \t]{2,}", " ", text)                  # 压缩多余空格
    text = re.sub(r"\n{3,}", "\n\n", text)                 # 压缩多余空行
    return text.strip() + "\n", stats


def read_source(path: Path) -> str:
    """读源文件（.docx / .md / .txt）成纯文本。"""
    if path.suffix.lower() == ".docx":
        import docx  # python-docx

        document = docx.Document(str(path))
        lines = [para.text.strip() for para in document.paragraphs if para.text.strip()]
        return "\n".join(lines)
    return path.read_text(encoding="utf-8", errors="replace")


def anonymize_file(src: Path, dst: Path) -> Dict[str, int]:
    """脱敏一个文件并写出。"""
    text = read_source(src)
    cleaned, stats = anonymize_text(text)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(cleaned, encoding="utf-8")
    return stats


def _guess_output_name(src: Path) -> str:
    """从源文件名推出输出名（去掉人名部分）。"""
    stem = src.stem
    # 「<姓名>简历202607-AI工程」→ 「简历202607-AI工程」
    stem = re.sub(r"^[\u4e00-\u9fa5]{2,4}(?=(简历|个人|求职))", "", stem)
    stem = stem.strip("-_ ") or "resume"
    return stem + ".md"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="简历脱敏：把真实简历转成可公开入库的 .md")
    parser.add_argument("--in", dest="src", help="输入文件（.docx/.md/.txt）")
    parser.add_argument("--in-dir", dest="src_dir", help="输入目录（批量处理 .docx）")
    parser.add_argument("--out", dest="dst", help="输出文件路径")
    parser.add_argument("--out-dir", dest="dst_dir", default="data/resumes", help="输出目录（默认 data/resumes）")
    args = parser.parse_args(argv)

    if not args.src and not args.src_dir:
        parser.error("需要 --in 或 --in-dir")
    if args.src and args.src_dir:
        parser.error("--in 与 --in-dir 不能同时用")

    root = Path(__file__).resolve().parent.parent
    out_dir = Path(args.dst_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir

    jobs: List[Tuple[Path, Path]] = []
    if args.src:
        src = Path(args.src)
        dst = Path(args.dst) if args.dst else out_dir / _guess_output_name(src)
        jobs.append((src, dst))
    else:
        src_dir = Path(args.src_dir)
        for src in sorted(src_dir.glob("*.docx")):
            jobs.append((src, out_dir / _guess_output_name(src)))

    if not jobs:
        print("[错误] 没有找到可处理的文件", file=sys.stderr)
        return 1

    total: Dict[str, int] = {}
    for src, dst in jobs:
        if not src.exists():
            print(f"[跳过] 文件不存在：{src}", file=sys.stderr)
            continue
        stats = anonymize_file(src, dst)
        for key, value in stats.items():
            total[key] = total.get(key, 0) + value
        print(f"[完成] {src.name} -> {dst.name}  {stats}")

    print("\n合计：", total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
