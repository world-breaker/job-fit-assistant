"""技能名归一化：让「LLM 抽的」「规则抽的」「人工标的」能对上同一个东西。

评测要算 P/R/F1，前提是三方的技能名**能对齐**。对齐规则必须写死在这里，
并且被抽取器与评测器**共用**——否则评测数字算的是「字符串巧合」。

归一化分三步：
1. **查词表**：别名 → 规范名（jobfit.lexicon）。
2. **拆复合短语**：LLM 常把多个技术塞进一条（「计算机基础、数据结构与算法」）。
   这类要拆成单个技术名词——一条要求 = 一个可独立验证的技术。
   拆不开的（拆出来不在词表里）保留原样，但评测时按「复合短语」口径处理。
3. **清洗**：去空白、去首尾标点、全角转半角、大小写统一。

★ 为什么拆短语是必须的（不是洁癖）：
「一条要求」这个单位如果不唯一，P/R/F1 就没有意义——
「计算机基础、数据结构与算法」这一条，既可以算命中「数据结构与算法」，
也可以算没命中，取决于评测端怎么切。切分规则写在代码里（这里），
而不是每个评测脚本各切各的。
"""

from __future__ import annotations

import re
import unicodedata
from typing import List, Optional, Set, Tuple

from jobfit.lexicon import ALIAS_TO_CANON, SKILL_CATEGORY

#: 复合短语的连接符
_SPLIT_PATTERN = re.compile(r"[、，,;；/]|和|与|及|或者|\+")

#: **强枚举分隔符**（顿号/逗号/分号/斜杠）——中文里这是明确的「并列列举」信号。
#: 「和」「与」是弱连接（可能是词的一部分，如「零售与批发」），要求更严。
_STRONG_SPLIT_RE = re.compile(r"[、，,;；/]")

#: 弱连接符（只有当**每一段都查得到**时才用它拆）
_WEAK_SPLIT_RE = re.compile(r"和|与|及|或者|\+")

#: 归一化时剥掉的首尾噪声字符
_TRIM_CHARS = " \t\r\n　。．.，,、；;：:（）()【】[]「」\"'’“”"

#: 全角 → 半角
_FULLWIDTH_OFFSET = 0xFEE0


def to_halfwidth(text: str) -> str:
    """全角字符转半角（中文 JD 里全角括号/冒号很常见）。"""
    out = []
    for ch in text:
        code = ord(ch)
        if code == 0x3000:  # 全角空格
            out.append(" ")
        elif 0xFF01 <= code <= 0xFF5E:
            out.append(chr(code - _FULLWIDTH_OFFSET))
        else:
            out.append(ch)
    return "".join(out)


def clean(text: str) -> str:
    """基础清洗：全角转半角 + 去首尾噪声 + 压缩空白。"""
    value = to_halfwidth(str(text or "")).strip(_TRIM_CHARS)
    return re.sub(r"\s+", " ", value).strip()


def normalize_key(text: str) -> str:
    """用于**集合比对**的键：小写 + 去所有空白与常见分隔符。

    「Spring Boot」/「spring boot」/「SpringBoot」→ 同一个键。
    """
    value = to_halfwidth(str(text or "")).lower()
    return re.sub(r"[\s\-_/·]+", "", value).strip(_TRIM_CHARS)


def canonical(text: str) -> Optional[str]:
    """查词表：别名 → 规范名；不在词表里返回 None。"""
    key = clean(text)
    if not key:
        return None
    hit = ALIAS_TO_CANON.get(key.lower())
    if hit:
        return hit
    # 退路：去掉常见的「经验」「熟悉」「精通」等修饰词后再查
    stripped = re.sub(r"(相关的?|方面的?|经验|技术|框架|语言|工具|库)$", "", key).strip()
    if stripped:
        return ALIAS_TO_CANON.get(stripped.lower())
    return None


def split_compound(text: str) -> List[str]:
    """把复合短语拆成单个技术名词。

    拆分规则（分强/弱两级）：
    - **强分隔符**（顿号/逗号/分号/斜杠）：中文里这是明确的「并列列举」信号，
      所以只要**至少有一段**能在词表里查到，就拆。
      例：「高并发、高可靠海量服务开发」→ 高并发 + 高可靠海量服务开发。
    - **弱分隔符**（和/与/及）：可能是词的一部分（「零售与批发」），
      所以要求**每一段都查得到**才拆。

    拆不动（没有任何一段查得到）→ 保留原样（它本来就是一个专有概念）。
    """
    value = clean(text)
    if not value:
        return []

    if _STRONG_SPLIT_RE.search(value):
        parts = [p.strip() for p in _STRONG_SPLIT_RE.split(value) if p and p.strip()]
        if len(parts) >= 2 and any(canonical(p) for p in parts):
            return parts
        return [value]

    if _WEAK_SPLIT_RE.search(value):
        parts = [p.strip() for p in _WEAK_SPLIT_RE.split(value) if p and p.strip()]
        if len(parts) >= 2 and all(canonical(p) for p in parts):
            return parts
        return [value]

    return [value]


def normalize_skill(text: str) -> str:
    """把任意技能名归一化成**单个**规范名。

    口径：
    1. 查词表命中 → 用规范名；
    2. 未命中但可拆 → 用第一个拆出来的规范名（其余在评测时算作额外召回）；
    3. 都不可 → 用清洗后的原文。
    """
    value = clean(text)
    if not value:
        return ""
    hit = canonical(value)
    if hit:
        return hit
    parts = split_compound(value)
    if parts and canonical(parts[0]):
        return canonical(parts[0]) or parts[0]
    return value


def normalize_set(names: List[str]) -> Set[str]:
    """一组技能名 → 归一化后的键集合（用于集合比对）。"""
    out: Set[str] = set()
    for name in names:
        cleaned = clean(name)
        if not cleaned:
            continue
        canon = canonical(cleaned)
        if canon:
            out.add(normalize_key(canon))
            continue
        parts = split_compound(cleaned)
        if len(parts) > 1:
            for part in parts:
                out.add(normalize_key(part))
        else:
            out.add(normalize_key(cleaned))
    return out


def match_key(text: str) -> str:
    """**集合比对用的键**：先归一化到词表规范名，再做字符级归一。

    ★ 为什么不能直接用 `normalize_key`：那样「SFT」和「LLM微调」会被当成两个技能，
    「RLVR」和「DPO」也是——但它们在词表里是同一个东西（别名）。
    评测必须按**语义同一**来比对，否则指标算的是字符串巧合。
    """
    value = clean(text)
    if not value:
        return ""
    canon = canonical(value)
    if canon:
        return normalize_key(canon)
    parts = split_compound(value)
    if len(parts) > 1:
        # 复合短语：每个部分各自成一个键（这样「A、B」能对上标注里的 A 与 B）
        keys = []
        for part in parts:
            part_canon = canonical(part)
            keys.append(normalize_key(part_canon or part))
        return "|".join(sorted(set(keys)))
    return normalize_key(value)


def match_keys(names: List[str]) -> Set[str]:
    """一组技能名 → 比对用的键集合。

    ★ 复合短语会产生多个键（用 `|` 连接）：
    「计算机基础、数据结构与算法」→ {`算法`, `计算机基础|数据结构与算法`}。
    评测时用**子集包含**判定——预测里的复合键只要覆盖了标注里的键就算命中，
    反之标注里的复合键只要被预测覆盖也算命中（详见 metrics.score_extraction）。
    """
    out: Set[str] = set()
    for name in names:
        key = match_key(name)
        if key:
            out.add(key)
    return out


def category_of(name: str) -> str:
    """技能名的类别（词表外返回空串）。"""
    canon = canonical(name)
    if canon:
        return SKILL_CATEGORY.get(canon, "")
    return ""
