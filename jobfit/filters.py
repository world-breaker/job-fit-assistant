"""非技能项过滤：把「软素质 / 学历要求 / 会议名 / 通用能力」挡在技能之外。

★ 为什么需要这个过滤器：
LLM 抽取 JD 技能要求时，会把很多东西当成「技能」——
「逻辑思维能力」「沟通协作」「计算机基础」「AAAI 会议发表」「广告推荐经验」…
这些**不是技能要求**，把它们算进去会让 P/R/F1 的分母虚高、精度虚低。

但过滤不能靠「词表里有没有」——词表里没有不代表它不是技能
（真实 JD 会提到词表没收录的技术）。所以这里分两类处理：
- **明确不是技能的**（软素质、学历、年龄、会议名、通用形容词）→ 直接过滤；
- **词表外但可能是真技能的** → 保留，交给评测端判分。

★ 过滤规则本身也是「口径」的一部分，写在这里并写进 README，
让读者能判断这套过滤是否合理（而不是藏在 prompt 里看不见）。
"""

from __future__ import annotations

import re
from typing import List, Tuple

#: 明确不是「技能要求」的软素质/通用能力（大小写不敏感的子串匹配）
NON_SKILL_PATTERNS: List[str] = [
    # 软素质 / 通用能力
    "逻辑思维", "沟通协作", "沟通能力", "协作能力", "团队协作", "团队合作",
    "责任心", "抗压", "自驱", "学习能力", "主动性", "问题解决能力", "问题拆解",
    "热情", "兴趣", "态度", "性格", "沟通与", "统筹规划", "带教", "培养",
    "分析师", "产品经理", "技术方案输出", "跨团队",
    # 学历 / 年龄 / 背景
    "本科", "硕士", "博士", "学历", "学位", "相关专业", "计算机专业", "数学专业",
    "工作经验", "工作年限", "年以上", "应届", "在校", "毕业",
    # 会议 / 期刊 / 学术
    "NeurIPS", "ICML", "ICLR", "KDD", "AAAI", "IJCAI", "CVPR", "ACL", "SIGIR",
    "NearlPS", "顶会", "顶刊", "一作", "第一作者", "论文发表", "期刊发表",
    # 通用形容 / 泛泛的名词
    "计算机基础", "计算机科学", "工程能力", "科研能力", "实验能力", "工程实践能力",
    "良好的", "较强的", "扎实的", "优秀的", "丰富的", "完善", "体系", "机制",
    "平台", "系统", "项目", "业务", "产品", "需求", "流程", "规范", "标准",
    "经验", "能力", "技术", "问题", "工作", "开发", "设计", "优化", "分析",
]

#: 明显的「非技能名词」——通用业务名词（不是具体技术）
GENERIC_BUSINESS: List[str] = [
    "广告", "推荐", "搜索", "电商", "社交", "游戏", "金融", "教育", "医疗",
    "风控", "支付", "交易", "商品", "内容", "用户", "客户", "数据",
]

#: 会议/期刊名的正则（大小写不敏感）
_ACADEMIC_RE = re.compile(
    r"\b(neurips|nips|icml|iclr|kdd|aaai|ijcai|cvpr|eccv|acl|emnlp|naacl|sigir|"
    r"www|kdd|cvpr)\b",
    re.IGNORECASE,
)


def is_non_skill(name: str) -> bool:
    """判断一个抽取出的技能名是否「明确不是技能要求」。

    命中任何一条非技能模式 → True（应被过滤掉）。
    """
    text = (name or "").strip()
    if not text:
        return True
    if _ACADEMIC_RE.search(text):
        return True
    for pattern in NON_SKILL_PATTERNS:
        if pattern in text:
            return True
    return False


def is_generic_business(name: str) -> bool:
    """判断是否只是泛泛的业务名词（如「广告」「推荐」）。

    这类**不直接过滤**——因为「推荐系统」「搜索引擎」确实是技能；
    只有当它既不是词表里的技能、又只是个泛业务词时才过滤。
    """
    text = (name or "").strip()
    return text in GENERIC_BUSINESS


def filter_non_skills(
    items: List[Tuple[str, str]],
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """把 (技能名, 重要度) 列表分成「保留」与「过滤掉」两堆。

    返回 (保留, 过滤掉)。过滤掉的部分会记进统计，报告里能看见
    「有多少条被非技能过滤器挡掉了」——这本身是个可审计的数字。
    """
    kept: List[Tuple[str, str]] = []
    dropped: List[Tuple[str, str]] = []
    for name, importance in items:
        if is_non_skill(name):
            dropped.append((name, importance))
        else:
            kept.append((name, importance))
    return kept, dropped
