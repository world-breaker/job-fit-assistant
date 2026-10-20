"""数据结构：六步流程里每一步的输入输出都落成显式结构。

为什么用 dataclass 而不是裸 dict：报告、JSON 导出、评测三处都要读同一批字段，
裸 dict 会在某处少写一个 key，直到运行期才炸。这里每个字段都有类型，
序列化走 `to_dict()` 一处实现，避免各处 `dict` 拼装口径不一致。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

#: 需求重要度取值
IMPORTANCE_MUST = "必须"
IMPORTANCE_PLUS = "加分"
IMPORTANCE_VALUES = (IMPORTANCE_MUST, IMPORTANCE_PLUS)

#: 重要度加权 F1 用的权重（必须项权重更高）
IMPORTANCE_WEIGHTS: Dict[str, float] = {IMPORTANCE_MUST: 2.0, IMPORTANCE_PLUS: 1.0}

#: 匹配结论取值
MATCH_HIT = "命中"
MATCH_PARTIAL = "部分命中"
MATCH_MISSING = "缺失"
MATCH_VALUES = (MATCH_HIT, MATCH_PARTIAL, MATCH_MISSING)

#: 抽取来源标记：LLM 抽取 / 规则兜底 / 关键词基线
SOURCE_LLM = "llm"
SOURCE_RULE_FALLBACK = "rule_fallback"
SOURCE_KEYWORD_BASELINE = "keyword_baseline"
SOURCE_RULE = "rule"


@dataclass
class Requirement:
    """一条 JD 技能要求（步骤 1 的产物）。

    - skill：规范技能名（词表口径），非词表内的保留原文写法
    - category：语言 / 框架 / 工程 / 算法 / 业务
    - importance：必须 / 加分
    - evidence：JD 原文里支撑这条要求的**原句**
    - source：这条要求是从哪来的（llm / rule_fallback / keyword_baseline）
    - raw：抽取时看到的原始技能名（LLM 可能给出词表外的写法）
    """

    skill: str
    category: str
    importance: str
    evidence: str
    source: str = SOURCE_LLM
    raw: str = ""
    detail: str = ""

    def __post_init__(self) -> None:
        if self.importance not in IMPORTANCE_VALUES:
            raise ValueError(f"importance 必须是 {IMPORTANCE_VALUES} 之一，实际 {self.importance!r}")
        if not self.evidence:
            raise ValueError(f"要求 {self.skill!r} 缺少原文证据（evidence 不能为空）")

    @property
    def weight(self) -> float:
        return IMPORTANCE_WEIGHTS.get(self.importance, 1.0)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ResumeSkill:
    """简历里的一项技能（步骤 2 的产物）——永远带原文证据。"""

    skill: str
    category: str
    evidence: str
    section: str = ""
    source: str = SOURCE_RULE

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ResumeItem:
    """简历里的一个经历块（项目 / 教育 / 技能栏…）。"""

    title: str
    section: str
    lines: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ResumeProfile:
    """简历结构化结果（步骤 2 的产物）。"""

    items: List[ResumeItem] = field(default_factory=list)
    skills: List[ResumeSkill] = field(default_factory=list)
    raw_lines: List[str] = field(default_factory=list)
    source_path: str = ""
    source_format: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "items": [i.to_dict() for i in self.items],
            "skills": [s.to_dict() for s in self.skills],
            "raw_line_count": len(self.raw_lines),
            "source_path": self.source_path,
            "source_format": self.source_format,
        }


@dataclass
class MatchResult:
    """一条要求在简历里的匹配结论（步骤 3 的产物）。"""

    requirement: Requirement
    status: str
    evidence: str = ""
    evidence_section: str = ""
    note: str = ""

    def __post_init__(self) -> None:
        if self.status not in MATCH_VALUES:
            raise ValueError(f"status 必须是 {MATCH_VALUES} 之一，实际 {self.status!r}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "requirement": self.requirement.to_dict(),
            "status": self.status,
            "evidence": self.evidence,
            "evidence_section": self.evidence_section,
            "note": self.note,
        }


@dataclass
class GapAction:
    """一条差距/建议（步骤 4、5 的产物）。"""

    skill: str
    importance: str
    priority: float
    action: str
    rationale: str = ""
    kind: str = "gap"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RewriteSuggestion:
    """一条简历改写建议（步骤 5 的产物）。"""

    skill: str
    target: str
    before: str = ""
    after: str = ""
    reason: str = ""
    source: str = SOURCE_RULE

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class InterviewQuestion:
    """一道预测面试题（步骤 6 的产物）。"""

    question: str
    category: str
    target_skill: str = ""
    reason: str = ""
    risk: str = ""
    priority: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LlmUsage:
    """LLM 调用与费用记录（费用要进报告，预算上限由调用方强制）。"""

    backend: str = "none"
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    price_known: bool = True
    errors: List[str] = field(default_factory=list)

    def add(self, prompt: int, completion: int, cost: float, price_known: bool = True) -> None:
        self.calls += 1
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.cost_usd += cost
        self.price_known = self.price_known and price_known

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "price_known": self.price_known,
            "errors": list(self.errors),
        }


@dataclass
class TraceRef:
    """trace 文件引用——报告里要能看到编排证据。"""

    path: str = ""
    steps: int = 0
    events: Dict[str, int] = field(default_factory=dict)
    tool_calls: int = 0
    tool_failures: int = 0
    tool_degraded: int = 0
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AnalysisReport:
    """一次分析的完整产物（六步的汇总）。"""

    jd_path: str = ""
    resume_path: str = ""
    jd_title: str = ""
    requirements: List[Requirement] = field(default_factory=list)
    resume_profile: Optional[ResumeProfile] = None
    matches: List[MatchResult] = field(default_factory=list)
    gaps: List[GapAction] = field(default_factory=list)
    rewrites: List[RewriteSuggestion] = field(default_factory=list)
    questions: List[InterviewQuestion] = field(default_factory=list)
    llm_usage: LlmUsage = field(default_factory=LlmUsage)
    trace: TraceRef = field(default_factory=TraceRef)
    fallback_ratios: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    started_at: str = ""
    elapsed_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "jd_path": self.jd_path,
            "resume_path": self.resume_path,
            "jd_title": self.jd_title,
            "requirements": [r.to_dict() for r in self.requirements],
            "resume_profile": self.resume_profile.to_dict() if self.resume_profile else None,
            "matches": [m.to_dict() for m in self.matches],
            "gaps": [g.to_dict() for g in self.gaps],
            "rewrites": [r.to_dict() for r in self.rewrites],
            "questions": [q.to_dict() for q in self.questions],
            "llm_usage": self.llm_usage.to_dict(),
            "trace": self.trace.to_dict(),
            "fallback_ratios": self.fallback_ratios,
            "notes": list(self.notes),
            "started_at": self.started_at,
            "elapsed_ms": self.elapsed_ms,
        }
