"""CLI 入口。

    python -m jobfit analyze --jd <file> --resume <file> [--format md|json]
    python -m jobfit serve [--host 127.0.0.1] [--port 8765]
    python -m jobfit doctor        # 环境自检（兄弟仓库 / LLM / 依赖）

设计原则：
- 默认**不联网**（除非配置了 LLM key 且用 `--llm auto`）；`--llm none` 强制纯规则；
- `--out` 写文件，`--out -` 写 stdout（默认行为）；
- 任何一步失败都要给出**可操作的**错误信息，不是 traceback 裸奔。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, List, Optional

from jobfit import siblings

#: 仓库根目录
ROOT = Path(__file__).resolve().parent.parent


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jobfit",
        description="求职助手：JD 技能要求抽取 → 简历解析 → 匹配分析 → 差距建议 → 改写建议 → 面试题",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze", help="分析一份 JD 与一份简历")
    analyze.add_argument("--jd", required=True, help="JD 文件路径（.txt/.md/.json）")
    analyze.add_argument("--resume", required=True, help="简历文件路径（.md/.txt/.docx）")
    analyze.add_argument("--format", default="md", choices=["md", "json"], help="输出格式（默认 md）")
    analyze.add_argument("--out", default="-", help="输出文件路径；'-' 表示 stdout（默认）")
    analyze.add_argument(
        "--llm",
        default="auto",
        choices=["auto", "deepseek", "local", "none"],
        help="LLM 后端：auto=有 key 用 API 否则用本机模型；none=纯规则（默认 auto）",
    )
    analyze.add_argument("--llm-model", default="", help="覆盖模型名")
    analyze.add_argument(
        "--budget-usd", type=float, default=0.50, help="LLM 预算上限（美元），超出即停（默认 0.5）"
    )
    analyze.add_argument("--no-mcp", action="store_true", help="关闭 MCP 检索通道（只用本地匹配）")
    analyze.add_argument("--max-questions", type=int, default=8, help="预测面试题数量上限（默认 8）")
    analyze.add_argument("--trace-dir", default="", help="trace 输出目录（默认仓库内 traces/）")

    serve_cmd = sub.add_parser("serve", help="启动 HTTP 服务（/healthz + /analyze）")
    serve_cmd.add_argument("--host", default="127.0.0.1", help="监听地址（默认只监听本机）")
    serve_cmd.add_argument("--port", type=int, default=8765, help="监听端口（默认 8765）")
    serve_cmd.add_argument(
        "--llm", default="auto", choices=["auto", "deepseek", "local", "none"], help="LLM 后端"
    )
    serve_cmd.add_argument("--llm-model", default="", help="覆盖模型名")
    serve_cmd.add_argument("--budget-usd", type=float, default=0.50, help="LLM 预算上限（美元）")

    sub.add_parser("doctor", help="环境自检：兄弟仓库 / LLM / 依赖是否就绪")
    return parser


def _build_llm(args: argparse.Namespace) -> Optional[Any]:
    from jobfit.llm import build_llm

    if getattr(args, "llm", "auto") == "none":
        return None
    llm = build_llm(
        backend=args.llm,
        model=getattr(args, "llm_model", ""),
        budget_usd=getattr(args, "budget_usd", 0.50),
    )
    return llm


def cmd_analyze(args: argparse.Namespace) -> int:
    from jobfit.pipeline import JobFitPipeline
    from jobfit.report import render

    trace_dir = Path(args.trace_dir) if args.trace_dir else (ROOT / "traces")
    pipeline = JobFitPipeline(
        llm=_build_llm(args),
        trace_dir=trace_dir,
        use_mcp=not args.no_mcp,
        max_questions=args.max_questions,
    )
    try:
        report = pipeline.run(args.jd, args.resume)
    except FileNotFoundError as exc:
        print(f"[错误] 文件不存在：{exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 —— CLI 要给可读错误，不是 traceback
        print(f"[错误] 分析失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    output = render(report, args.format)
    if args.out == "-":
        print(output)
    else:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output, encoding="utf-8")
        print(f"报告已写入：{out_path}")
        if report.trace.path:
            print(f"trace：{report.trace.path}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from jobfit.server import serve

    llm = _build_llm(args)
    serve(host=args.host, port=args.port, llm=llm)
    return 0


def cmd_doctor(_args: argparse.Namespace) -> int:
    """环境自检——把「能不能跑」这件事一次说清。"""
    print("=== 兄弟仓库（地基）===")
    status = siblings.available_siblings()
    for name, ok in status.items():
        path = siblings.sibling_path(name)
        print(f"  {'OK  ' if ok else 'MISS'} {name:20s} {path}")

    print("\n=== 可导入性 ===")
    checks = [
        ("hello-agent agent 包", siblings.hello_agent_available),
        ("rag-mcp-server rag 包", siblings.rag_available),
        ("rag-eval-platform src 包", siblings.rag_eval_available),
    ]
    for label, fn in checks:
        print(f"  {'OK  ' if fn() else 'MISS'} {label}")

    print("\n=== LLM ===")
    from jobfit.llm import _api_key_from_env, build_llm

    key = _api_key_from_env()
    print(f"  API key：{'已配置' if key else '未配置（将走本机离线模型或纯规则）'}")
    llm = build_llm(backend="auto")
    print(f"  auto 选中的后端：{type(llm).__name__ if llm else '无（纯规则模式）'}")

    print("\n=== Python 依赖 ===")
    import importlib

    for module in ("mcp", "langchain_core", "yaml", "numpy", "docx", "pytest"):
        try:
            importlib.import_module(module)
            print(f"  OK   {module}")
        except Exception as exc:  # noqa: BLE001
            print(f"  MISS {module}（{type(exc).__name__}）")

    ok = all(status.values())
    print("\n结论：" + ("全部就绪" if ok else "部分地基仓库缺失——检索/评测能力会降级，其余功能仍可用"))
    return 0 if ok else 1


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "analyze":
        return cmd_analyze(args)
    if args.command == "serve":
        return cmd_serve(args)
    if args.command == "doctor":
        return cmd_doctor(args)
    parser.error(f"未知命令：{args.command}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
