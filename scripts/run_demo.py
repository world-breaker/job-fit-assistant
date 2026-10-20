"""端到端 demo：真实 JD（AI 工程方向）+ 4 份真实脱敏简历 → 4 份真实报告。

★ 这是「落地」的证据：作者的求职流程真的跑通了，不是示意图。

产出（experiments/results/demo/）：
- `demo_<JD id>_<简历名>.md` —— 4 份完整的六步报告
- `demo_<JD id>_summary.md` —— 4 份报告的横向对比（推荐哪版 + 为什么）
- `demo_run.json` —— 机器可读的全部结果

用法：
    python scripts/run_demo.py
    python scripts/run_demo.py --llm none        # 不花钱的纯规则档
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from jobfit.llm import build_llm  # noqa: E402
from jobfit.pipeline import JobFitPipeline  # noqa: E402
from jobfit.report import render_json, render_markdown  # noqa: E402

#: demo 用的 JD：真实公开 JD（AI 工程方向）
DEMO_JD_ID = "2008789833664847872"

#: demo 用的 4 份简历
DEMO_RESUMES = [
    "简历202607-AI工程",
    "简历202607-Java开发",
    "简历202607-NLP算法",
    "简历202607-评测后训练-有联",
]


def run_demo(
    out_dir: Path,
    jd_id: str = DEMO_JD_ID,
    llm_backend: str = "auto",
    budget_usd: float = 0.5,
    use_mcp: bool = True,
) -> Dict[str, Any]:
    jd_path = ROOT / "data" / "jds" / f"{jd_id}.json"
    if not jd_path.exists():
        raise SystemExit(f"找不到 demo 用的 JD：{jd_path}")
    jd = json.loads(jd_path.read_text(encoding="utf-8"))
    jd_text = jd["raw_text"]

    llm = build_llm(backend=llm_backend, budget_usd=budget_usd)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    started = time.perf_counter()

    for resume_name in DEMO_RESUMES:
        resume_path = ROOT / "data" / "resumes" / f"{resume_name}.md"
        if not resume_path.exists():
            print(f"[跳过] 找不到简历：{resume_path}", file=sys.stderr)
            continue

        pipeline = JobFitPipeline(
            llm=llm,
            trace_dir=ROOT / "traces",
            use_mcp=use_mcp,
            max_questions=8,
        )
        report = pipeline.run(jd_path, resume_path, jd_text=jd_text)

        md = render_markdown(report)
        target = out_dir / f"demo_{jd_id}_{resume_name}.md"
        target.write_text(md, encoding="utf-8")

        payload = json.loads(render_json(report))
        results.append(
            {
                "resume": resume_name,
                "report_path": str(target.relative_to(ROOT)),
                "requirements": len(report.requirements),
                "matches": len(report.matches),
                "hit": sum(1 for m in report.matches if m.status == "命中"),
                "partial": sum(1 for m in report.matches if m.status == "部分命中"),
                "missing": sum(1 for m in report.matches if m.status == "缺失"),
                "gaps": len(report.gaps),
                "rewrites": len(report.rewrites),
                "questions": len(report.questions),
                "trace": report.trace.path,
                "json": payload,
            }
        )
        print(
            f"[完成] {resume_name:26s} 命中 {results[-1]['hit']:2d} / "
            f"部分 {results[-1]['partial']:2d} / 缺失 {results[-1]['missing']:2d}  → {target.name}"
        )

    # ---- 横向对比汇总 ----
    from jobfit.advice import coverage_summary

    summary_lines = [
        "# 端到端 Demo：4 份简历对同一份 JD 的匹配对比",
        "",
        f"- JD：`{jd['title']}`（{jd['source_site']}）",
        f"- 来源链接：{jd.get('source_url') or '（自写变体）'}",
        f"- 抓取日期：{jd.get('fetched_at') or '—'}",
        f"- 抽取到的技能要求：{results[0]['requirements'] if results else 0} 条",
        "",
        "## 匹配结果横向对比",
        "",
        "| 简历 | 命中 | 部分命中 | 缺失 | 覆盖率 | 改写建议 | 面试题 |",
        "|------|------|----------|------|--------|----------|--------|",
    ]
    for row in results:
        total = row["hit"] + row["partial"] + row["missing"]
        rate = f"{row['hit'] / total * 100:.1f}%" if total else "—"
        summary_lines.append(
            f"| {row['resume']} | {row['hit']} | {row['partial']} | {row['missing']} | "
            f"{rate} | {row['rewrites']} | {row['questions']} |"
        )

    if results:
        best = max(results, key=lambda r: (r["hit"], r["partial"]))
        summary_lines += [
            "",
            "## 结论",
            "",
            f"**推荐版本：{best['resume']}**（命中 {best['hit']} 条，"
            f"缺失 {best['missing']} 条最少）",
            "",
            "其余三份报告见同目录下的 `demo_*.md` 文件。",
        ]
    summary_lines += [
        "",
        "## 说明",
        "",
        f"- 本次运行 LLM 后端：`{type(llm).__name__ if llm else 'none（纯规则）'}`",
        f"- 每份报告的编排证据（JSONL trace）在 `traces/` 下，可回放每一步。",
        "- 命中/部分命中/缺失的判定口径见 jobfit/matching.py 的模块 docstring。",
    ]
    (out_dir / f"demo_{jd_id}_summary.md").write_text(
        "\n".join(summary_lines), encoding="utf-8"
    )

    payload = {
        "jd_id": jd_id,
        "jd_title": jd["title"],
        "jd_source": {
            "site": jd["source_site"],
            "url": jd.get("source_url", ""),
            "fetched_at": jd.get("fetched_at", ""),
            "source_type": jd["source_type"],
        },
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "llm_backend": type(llm).__name__ if llm else "none",
        "llm_usage": llm.usage.to_dict() if llm else {"backend": "none", "calls": 0},
        "mcp_enabled": use_mcp,
        "results": results,
    }
    (out_dir / "demo_run.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nDemo 完成，用时 {payload['elapsed_seconds']}s → {out_dir}")
    return payload


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="端到端 demo：真实 JD + 4 份脱敏简历")
    parser.add_argument("--out", default="experiments/results/demo", help="输出目录")
    parser.add_argument("--jd", default=DEMO_JD_ID, help="demo 用的 JD id")
    parser.add_argument("--llm", default="auto", choices=["auto", "deepseek", "local", "none"])
    parser.add_argument("--budget-usd", type=float, default=0.5, help="LLM 预算上限")
    parser.add_argument("--no-mcp", action="store_true", help="关闭 MCP 检索通道")
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    run_demo(
        out_dir=out_dir,
        jd_id=args.jd,
        llm_backend=args.llm,
        budget_usd=args.budget_usd,
        use_mcp=not args.no_mcp,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
