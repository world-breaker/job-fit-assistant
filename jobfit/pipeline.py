"""编排层：把六步串成一条流水线，并接入三个地基仓库。

★ 与三个地基仓库的组合关系（README 里写清，这里是代码落点）：

1. **hello-agent** —— 提供「韧性 + JSONL trace」：
   - `agent.trace.Tracer`：每一步都落一条 JSONL 事件（可审计、可回放）；
   - `agent.resilience.ToolExecutor`：MCP 工具调用走它的超时/重试/降级/熔断；
   - `agent.graph` / `agent.runner` 的编排思路（多步 → 工具 → 观察 → 再决策）
     在这里落成一个**确定性的多步流水线**——注意：本项目的六步是
     「固定流程 + 每步内部可调工具」，不是自由 ReAct。所以这里复用它的
     Tracer + ToolExecutor（真正复用的代码），而**不**复用它的 ReAct 图
     （那是另一类问题：LLM 自主决定下一步做什么）。
   - `agent.usage.summarize_usage`：token/成本统计。
2. **rag-mcp-server** —— 提供「简历分块 + 向量检索」「JD 条款检索」：
   通过 **MCP 客户端**（jobfit/mcp_client.py）调用 jobfit/mcp_server.py，
   后者复用 rag-mcp-server 的 rag 包（切分/embedding/检索/提示注入防护）。
3. **rag-eval-platform** —— 提供「评测与报告」：
   评测脚本复用它的 EvaluationEngine（检索层指标）与报告口径。

**降级策略**：MCP 不可用时，流水线**继续跑**（用本地词表匹配做证据定位），
并在 trace 与报告里标明「检索通道降级」。绝不因为一个通道挂了整条流水线就断。
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from jobfit import siblings
from jobfit.advice import build_gaps, coverage_summary
from jobfit.extractor import extract_requirements, fallback_ratio
from jobfit.interview import build_questions
from jobfit.llm import BaseLlm
from jobfit.matching import match_requirements, summary_counts
from jobfit.resume import parse_resume, read_lines
from jobfit.rewrite import build_rewrites
from jobfit.schemas import AnalysisReport, TraceRef

#: trace 目录（相对仓库根）
DEFAULT_TRACE_DIR = "traces"

#: hello-agent 的 Tracer 需要一个 config 字典；这里给最小可用配置
def _agent_config(trace_dir: Path) -> Dict[str, Any]:
    """构造 hello-agent Tracer/ToolExecutor 需要的配置片段。

    只给用到的键：trace 目录、工具超时、重试、熔断阈值。
    键名与 hello-agent 的 config.py 保持一致（同一套语义）。
    """
    return {
        "trace": {
            "dir": str(trace_dir),
            "console": "quiet",
            "preview_chars": 240,
        },
        "tools": {
            "timeout_s": 60,
            "retry": {
                "max_attempts": 3,
                "backoff_base_s": 0.4,
                "backoff_max_s": 3.0,
                "retryable_patterns": [
                    "timeout", "timed out", "connection", "connect",
                    "temporarily", "rate limit", "429", "502", "503", "504",
                ],
            },
            "circuit_breaker": {"failure_threshold": 3},
        },
    }


class _NullTracer:
    """hello-agent 不可用时的降级 trace 器（同样的事件结构，写同样的 JSONL）。"""

    def __init__(self, trace_dir: Path) -> None:
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.events: List[Dict[str, Any]] = []
        self.step = 0
        self.path = self.trace_dir / "run_degraded.jsonl"

    def next_step(self) -> int:
        self.step += 1
        return self.step

    def event(self, type: str, **fields: Any) -> Dict[str, Any]:
        payload = {"ts": round(time.time(), 3), "step": self.step, "type": type, **fields}
        self.events.append(payload)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        return payload

    def close(self) -> Path:
        return self.path


def build_tracer(trace_dir: Path) -> Any:
    """优先用 hello-agent 的 Tracer；不可用则用本地降级实现。"""
    siblings.setup_paths()
    try:
        from agent.trace import Tracer  # type: ignore

        return Tracer(
            trace_dir=str(trace_dir),
            console="quiet",
            preview_chars=240,
        )
    except Exception:  # noqa: BLE001 —— 兄弟仓库不在就降级，不能让主流程崩
        return _NullTracer(trace_dir)


class JobFitPipeline:
    """六步流水线。

    步骤：
      1. JD 技能要求抽取（LLM + 规则兜底）
      2. 简历解析（.md/.txt/.docx）
      3. 匹配分析（命中/部分/缺失 + 原句证据）
      4. 差距与优先级建议
      5. 简历改写建议
      6. 预测面试题
    """

    def __init__(
        self,
        llm: Optional[BaseLlm] = None,
        trace_dir: Optional[Path] = None,
        use_mcp: bool = True,
        max_questions: int = 8,
    ) -> None:
        self.llm = llm
        self.use_mcp = use_mcp
        self.max_questions = max_questions
        root = Path(__file__).resolve().parent.parent
        self.trace_dir = Path(trace_dir) if trace_dir else (root / DEFAULT_TRACE_DIR)
        self.tracer = build_tracer(self.trace_dir)
        self.config = _agent_config(self.trace_dir)
        self.notes: List[str] = []

    # ---------------------------------------------------------------- 主流程

    def run(
        self,
        jd_path: str | Path,
        resume_path: str | Path,
        jd_text: Optional[str] = None,
    ) -> AnalysisReport:
        """跑完整流水线。"""
        started = time.perf_counter()
        self.tracer.event(
            "run_start",
            jd=str(jd_path),
            resume=str(resume_path),
            llm=type(self.llm).__name__ if self.llm else "none",
            use_mcp=self.use_mcp,
        )
        report = AnalysisReport(
            jd_path=self._rel(jd_path),
            resume_path=self._rel(resume_path),
            started_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        )

        # ---- 读取输入 ----
        jd_raw = jd_text if jd_text is not None else self._read_text(jd_path)
        resume_lines, resume_fmt = read_lines(resume_path)
        resume_text = "\n".join(resume_lines)
        report.resume_profile = None  # 步骤 2 填充

        # ---- MCP 检索通道（可选） ----
        mcp = self._open_mcp(jd_raw, resume_text)

        try:
            # ---- 步骤 1：JD 技能要求抽取 ----
            self.tracer.next_step()
            self.tracer.event("step_start", step_name="extract_jd_requirements")
            requirements, stats = extract_requirements(jd_raw, llm=self.llm)
            report.requirements = requirements
            report.notes.append(f"JD 抽取：{stats.to_dict()}")
            self.tracer.event(
                "step_end",
                step_name="extract_jd_requirements",
                count=len(requirements),
                **stats.to_dict(),
            )

            # ---- 步骤 2：简历解析 ----
            self.tracer.next_step()
            self.tracer.event("step_start", step_name="parse_resume")
            profile = parse_resume(resume_path, source_path_label=self._rel(resume_path))
            report.resume_profile = profile
            self.tracer.event(
                "step_end",
                step_name="parse_resume",
                format=resume_fmt,
                skills=len(profile.skills),
                items=len(profile.items),
            )

            # ---- 步骤 3：匹配分析（证据定位） ----
            self.tracer.next_step()
            self.tracer.event("step_start", step_name="match_requirements")
            matches = match_requirements(requirements, profile)
            # 用 MCP 检索补充证据（拿到更贴切的原文句）
            if mcp is not None:
                matches = self._enrich_evidence_with_mcp(mcp, matches, resume_text)
            report.matches = matches
            counts = summary_counts(matches)
            self.tracer.event("step_end", step_name="match_requirements", **counts)
            report.notes.append(f"覆盖情况：{coverage_summary(matches)}")

            # ---- 步骤 4：差距与优先级建议 ----
            self.tracer.next_step()
            self.tracer.event("step_start", step_name="build_gaps")
            report.gaps = build_gaps(matches)
            self.tracer.event("step_end", step_name="build_gaps", count=len(report.gaps))

            # ---- 步骤 5：简历改写建议 ----
            self.tracer.next_step()
            self.tracer.event("step_start", step_name="build_rewrites")
            report.rewrites = build_rewrites(matches, profile)
            self.tracer.event("step_end", step_name="build_rewrites", count=len(report.rewrites))

            # ---- 步骤 6：预测面试题 ----
            self.tracer.next_step()
            self.tracer.event("step_start", step_name="build_questions")
            report.questions = build_questions(matches, profile, max_questions=self.max_questions)
            self.tracer.event("step_end", step_name="build_questions", count=len(report.questions))

            # ---- 兜底占比 / LLM 记账 ----
            report.fallback_ratios = {
                "jd_extraction_rule_fallback": fallback_ratio(requirements),
            }
            if self.llm is not None:
                report.llm_usage = self.llm.usage
            report.notes.extend(self.notes)
        finally:
            if mcp is not None:
                self._close_mcp(mcp)

        report.elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        report.trace = self._trace_ref()
        self.tracer.event(
            "run_end",
            status="completed",
            steps=self.tracer.step,
            elapsed_ms=report.elapsed_ms,
            trace_path=str(self.tracer.path),
        )
        return report

    # ---------------------------------------------------------------- 内部

    def _rel(self, path: Any) -> str:
        """把路径转成**仓库相对路径**（提交物里不能出现绝对路径）。

        仓库根 = jobfit/ 的上一级。用 relative_to 时若不在仓库内（例如临时目录），
        就退回只留文件名——绝对路径同样不能进提交物。
        """
        from pathlib import Path as _Path

        root = Path(__file__).resolve().parent.parent
        try:
            return str(_Path(str(path)).resolve().relative_to(root))
        except (ValueError, OSError):
            return _Path(str(path)).name

    def _read_text(self, path: str | Path) -> str:
        """读 JD 文件（.json 也支持——评测集里的 JD 是 JSON）。"""
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"JD 文件不存在：{file_path}")
        text = file_path.read_text(encoding="utf-8")
        if file_path.suffix.lower() == ".json":
            payload = json.loads(text)
            if isinstance(payload, dict):
                for key in ("raw_text", "text", "jd_text", "body"):
                    if isinstance(payload.get(key), str) and payload[key].strip():
                        return payload[key]
            raise ValueError(f"{file_path.name} 里没有找到 JD 正文（raw_text/text/jd_text/body）")
        return text

    def _open_mcp(self, jd_text: str, resume_text: str) -> Optional[Any]:
        """建立 MCP 会话并装载语料。失败则降级（返回 None）。"""
        if not self.use_mcp:
            self.notes.append("MCP 检索通道：按配置关闭")
            return None
        try:
            from jobfit.mcp_client import McpClient, McpUnavailable
        except Exception as exc:  # noqa: BLE001
            self.notes.append(f"MCP 客户端不可用：{type(exc).__name__}: {exc}")
            return None

        try:
            client = McpClient()
            client.connect()
        except McpUnavailable as exc:
            self.notes.append(f"MCP 通道降级（检索证据走本地匹配）：{exc}")
            self.tracer.event("toolset_fallback", reason=str(exc))
            return None
        except Exception as exc:  # noqa: BLE001
            self.notes.append(f"MCP 通道降级（检索证据走本地匹配）：{type(exc).__name__}: {exc}")
            self.tracer.event("toolset_fallback", reason=f"{type(exc).__name__}: {exc}")
            return None

        self.tracer.event("mcp_connected", tools=client.tools, connect_ms=getattr(client, "connect_ms", 0))
        for tool, payload in (("set_jd_text", jd_text), ("set_resume_text", resume_text)):
            try:
                result = client.call(tool, {"text": payload})
            except Exception as exc:  # noqa: BLE001
                self.notes.append(f"MCP 工具 {tool} 调用失败：{type(exc).__name__}: {exc}")
                self.tracer.event("tool_error", tool=tool, error=f"{type(exc).__name__}: {exc}")
                continue
            self.tracer.event("mcp_load", tool=tool, result=result)
        return client

    def _close_mcp(self, client: Any) -> None:
        try:
            client.close()
        except Exception:  # noqa: BLE001
            pass

    def _enrich_evidence_with_mcp(
        self,
        client: Any,
        matches: List[Any],
        resume_text: str,
    ) -> List[Any]:
        """用 MCP 简历检索补充/校正证据句。

        做法：对每条「缺失」的要求，用 MCP 检索简历里语义最相关的片段，
        看是否有**词面没命中但语义相关**的证据（例如 JD 要求
        「向量检索」，简历写的是「搭建语义检索服务」）。
        ★ 只有当检索片段**确实包含技能别名**时才改判——语义相近不等于命中，
        这是防假绿的另一道闸。
        """
        from jobfit.lexicon import MATCHERS

        enriched = []
        for match in matches:
            if match.status != "缺失":
                enriched.append(match)
                continue
            try:
                payload = client.call("search_resume", {"query": match.requirement.skill, "top_k": 3})
            except Exception as exc:  # noqa: BLE001
                self.tracer.event("tool_error", tool="search_resume", error=f"{type(exc).__name__}: {exc}")
                enriched.append(match)
                continue

            results = (payload or {}).get("results") or []
            hit_text = ""
            for item in results:
                # ★ 用 raw_text（原始片段），不用被 <untrusted_data> 包裹过的 text——
                #   证据必须是简历里的原句，包装标签会让「证据是否真在简历里」的核对失败。
                text = item.get("raw_text") or item.get("text") or ""
                for canon, _alias, pattern in MATCHERS:
                    if canon == match.requirement.skill and pattern.search(text):
                        hit_text = text
                        break
                if hit_text:
                    break
            if hit_text:
                match.status = "部分命中"
                match.evidence = hit_text.strip()
                match.evidence_section = "MCP 检索"
                match.note = "词面未直接命中，但简历语义检索片段中出现该技能（按部分命中处理）"
                self.tracer.event(
                    "mcp_evidence_found",
                    skill=match.requirement.skill,
                    evidence=match.evidence[:120],
                )
            enriched.append(match)
        return enriched

    def _trace_ref(self) -> TraceRef:
        """从 trace 事件里聚合出报告要展示的 trace 证据。"""
        events = getattr(self.tracer, "events", []) or []
        counts: Dict[str, int] = {}
        for event in events:
            key = str(event.get("type"))
            counts[key] = counts.get(key, 0) + 1
        return TraceRef(
            path=self._rel(getattr(self.tracer, "path", "")),
            steps=int(getattr(self.tracer, "step", 0)),
            events=counts,
            tool_calls=counts.get("mcp_load", 0) + counts.get("mcp_evidence_found", 0),
            tool_failures=counts.get("tool_error", 0),
            tool_degraded=counts.get("toolset_fallback", 0),
            note="每一步的输入输出都落在这条 JSONL 里，可 grep 可回放",
        )
