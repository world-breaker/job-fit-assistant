"""报告渲染：Markdown（人读、可直接贴进求职流程）+ JSON（机器读）。

Markdown 报告的结构与规格里的六步一一对应，让人能从上往下顺着看：
抽取结果 → 简历解析 → 匹配分析 → 差距建议 → 改写建议 → 面试题 → 附录（trace / 费用 / 兜底占比）。
"""

from __future__ import annotations

from typing import Any, Dict, List

from jobfit.schemas import (
    MATCH_HIT,
    MATCH_MISSING,
    MATCH_PARTIAL,
    SOURCE_LLM,
    SOURCE_RULE_FALLBACK,
    AnalysisReport,
    MatchResult,
)

_STATUS_ICON = {MATCH_HIT: "✅", MATCH_PARTIAL: "⚠️", MATCH_MISSING: "❌"}


def _fmt_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def render_markdown(report: AnalysisReport) -> str:
    """渲染 Markdown 报告。"""
    lines: List[str] = []
    add = lines.append

    # ---------------- 概览 ----------------
    add("# 求职匹配分析报告")
    add("")
    add(f"- JD：`{report.jd_path}`")
    add(f"- 简历：`{report.resume_path}`")
    add(f"- 生成时间：{report.started_at}（耗时 {report.elapsed_ms / 1000:.2f}s）")
    if report.llm_usage.backend != "none":
        add(
            f"- LLM：`{report.llm_usage.backend}`，调用 {report.llm_usage.calls} 次，"
            f"token {report.llm_usage.prompt_tokens}+{report.llm_usage.completion_tokens}"
        )
        cost_text = (
            f"${report.llm_usage.cost_usd:.4f}"
            if report.llm_usage.price_known
            else "价格未知（只报 token，不报钱）"
        )
        add(f"- 费用：{cost_text}")
    else:
        add("- LLM：未使用（全部规则路径）")
    add("")

    # ---------------- 1. JD 技能要求 ----------------
    add("## 一、JD 技能要求抽取")
    add("")
    if not report.requirements:
        add("_未抽取出任何技能要求。_")
        add("")
    else:
        add(f"共 {len(report.requirements)} 条。其中规则兜底补充 "
            f"{report.fallback_ratios.get('jd_extraction_rule_fallback', 0) * 100:.1f}%"
            "（来源标记为 `rule_fallback` 的条目）。")
        add("")
        add("| # | 技能 | 类别 | 重要度 | 来源 | JD 原文证据 |")
        add("|---|------|------|--------|------|-----------|")
        for index, req in enumerate(report.requirements, 1):
            source = "LLM" if req.source == SOURCE_LLM else "规则兜底"
            evidence = req.evidence.replace("|", "／").replace("\n", " ")
            add(
                f"| {index} | {req.skill} | {req.category} | {req.importance} | "
                f"{source} | {evidence} |"
            )
        add("")

    # ---------------- 2. 简历解析 ----------------
    add("## 二、简历解析")
    add("")
    profile = report.resume_profile
    if profile is None:
        add("_未解析简历。_")
        add("")
    else:
        add(
            f"格式 `{profile.source_format}`，解析出 {len(profile.items)} 个经历块、"
            f"{len(profile.skills)} 项技能。"
        )
        add("")
        add("| 技能 | 类别 | 原文证据 |")
        add("|------|------|---------|")
        for skill in profile.skills:
            evidence = skill.evidence.replace("|", "／").replace("\n", " ")
            add(f"| {skill.skill} | {skill.category} | {evidence} |")
        add("")

    # ---------------- 3. 匹配分析 ----------------
    add("## 三、匹配分析")
    add("")
    if not report.matches:
        add("_无匹配结果。_")
        add("")
    else:
        counts = {MATCH_HIT: 0, MATCH_PARTIAL: 0, MATCH_MISSING: 0}
        for match in report.matches:
            counts[match.status] += 1
        total = len(report.matches)
        add(
            f"命中 {counts[MATCH_HIT]} 条（{_fmt_pct(counts[MATCH_HIT] / total)}）、"
            f"部分命中 {counts[MATCH_PARTIAL]} 条、"
            f"缺失 {counts[MATCH_MISSING]} 条。"
        )
        add("")
        add("| 状态 | 技能 | 重要度 | 简历原句证据 | 判定说明 |")
        add("|------|------|--------|--------------|---------|")
        for match in report.matches:
            evidence = (match.evidence or "（简历中无任何痕迹）").replace("|", "／").replace("\n", " ")
            add(
                f"| {_STATUS_ICON.get(match.status, '')} {match.status} | "
                f"{match.requirement.skill} | {match.requirement.importance} | "
                f"{evidence} | {match.note} |"
            )
        add("")

    # ---------------- 4. 差距与优先级建议 ----------------
    add("## 四、差距与优先级建议")
    add("")
    if not report.gaps:
        add("_没有需要补的差距项。_")
        add("")
    else:
        add("按「重要度 × 缺失度」排序，优先级高的在前。")
        add("")
        for index, gap in enumerate(report.gaps, 1):
            add(
                f"{index}. **{gap.skill}**（{gap.importance}，优先级 {gap.priority}）"
                f"{'〔部分命中，需补强〕' if gap.kind == 'strengthen' else '〔缺失〕'}"
            )
            add(f"   - **具体动作**：{gap.action}")
            add(f"   - **依据**：{gap.rationale}")
        add("")

    # ---------------- 5. 简历改写建议 ----------------
    add("## 五、简历改写建议")
    add("")
    if not report.rewrites:
        add("_无需改写。_")
        add("")
    else:
        for index, rw in enumerate(report.rewrites, 1):
            title = f"{index}. **{rw.skill or '整体表述'}** → 位置：{rw.target}"
            add(title)
            if rw.before:
                add(f"   - 改前：{rw.before}")
            add(f"   - 改后：{rw.after}")
            add(f"   - 理由：{rw.reason}")
        add("")

    # ---------------- 6. 预测面试题 ----------------
    add("## 六、预测面试题")
    add("")
    if not report.questions:
        add("_无预测题目。_")
        add("")
    else:
        add("按风险排序：先问简历上最容易被问穿的地方。")
        add("")
        for index, question in enumerate(report.questions, 1):
            add(
                f"{index}. **[风险 {question.risk}]** {question.question}"
            )
            add(f"   - 为什么问这个：{question.reason}")
        add("")

    # ---------------- 附录 ----------------
    add("## 附录：编排证据与诚实边界")
    add("")
    trace = report.trace
    if trace.path:
        add(f"- Trace 文件：`{trace.path}`（{trace.steps} 步，事件类型分布：{trace.events}）")
    if trace.tool_failures or trace.tool_degraded:
        add(
            f"- ⚠️ 检索通道异常：工具失败 {trace.tool_failures} 次，"
            f"降级 {trace.tool_degraded} 次（此时证据定位走本地匹配，精度会下降）"
        )
    add(f"- 兜底占比：{report.fallback_ratios}")
    add("")
    if report.notes:
        add("### 运行备注")
        add("")
        for note in report.notes:
            add(f"- {note}")
        add("")
    return "\n".join(lines)


def render_json(report: AnalysisReport) -> str:
    """渲染 JSON（ensure_ascii=False，中文可直接读）。"""
    import json

    return json.dumps(report.to_dict(), ensure_ascii=False, indent=2)


def render(report: AnalysisReport, fmt: str = "md") -> str:
    """按格式渲染。"""
    if fmt == "json":
        return render_json(report)
    return render_markdown(report)
