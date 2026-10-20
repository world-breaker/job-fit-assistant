"""HTTP 服务：stdlib `http.server`，零新依赖。

两个端点：
- `GET  /healthz` —— 只读健康检查（不触发任何模型加载，不做检索）
- `POST /analyze` —— 分析一个 JD + 一份简历，返回 Markdown 或 JSON

为什么用 stdlib 而不是 FastAPI：规格要求「零新依赖优先」。
本机确实装了 FastAPI，但用它的收益只是「自动生成 OpenAPI 文档」，
而代价是给一个自用小服务引入框架依赖 + 启动开销。用 stdlib 的 http.server
足够，且**没有依赖漂移风险**（不像某个库的版本变化导致启动失败）。

安全边界（诚实说明）：
- 服务**只监听 127.0.0.1**（默认），不对外暴露；
- 没有鉴权——它假定跑在开发者本机；
- `/analyze` 会**读本地文件路径**，因此只应传可信路径；
- 请求体大小有上限（默认 2MB），防止被超大 body 拖死。
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from jobfit.llm import BaseLlm
from jobfit.pipeline import JobFitPipeline

#: 默认监听地址（只监听本机）
DEFAULT_HOST = "127.0.0.1"

#: 默认端口
DEFAULT_PORT = 8765

#: 请求体大小上限（字节）
MAX_BODY_BYTES = 2 * 1024 * 1024

#: 服务启动时间（用于 /healthz 的 uptime）
STARTED_AT = time.time()


class _Handler(BaseHTTPRequestHandler):
    """请求处理器。"""

    server_version = "jobfit/0.1"
    protocol_version = "HTTP/1.1"

    # ---- 工具 ----
    def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, status: int, text: str, content_type: str = "text/plain; charset=utf-8") -> None:
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003 —— 覆盖默认 stderr 日志
        """默认实现会往 stderr 打日志；这里改成安静（避免污染调用方输出）。"""
        return

    # ---- 路由 ----
    def do_GET(self) -> None:  # noqa: N802 —— BaseHTTPRequestHandler 的约定
        path = self.path.split("?")[0].rstrip("/") or "/"
        if path == "/healthz":
            self._send_json(200, self._healthz())
        elif path == "/":
            self._send_text(
                200,
                "job-fit-assistant\n"
                "  GET  /healthz          健康检查\n"
                "  POST /analyze          分析（JSON body: {\"jd\":..., \"resume\":..., \"format\":\"md|json\"}）\n",
            )
        else:
            self._send_json(404, {"error": "not_found", "path": path})

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?")[0].rstrip("/") or "/"
        if path != "/analyze":
            self._send_json(404, {"error": "not_found", "path": path})
            return

        length_header = self.headers.get("Content-Length", "0")
        try:
            length = int(length_header)
        except ValueError:
            self._send_json(400, {"error": "bad_content_length"})
            return
        if length <= 0:
            self._send_json(400, {"error": "empty_body", "hint": "body: {\"jd\": path, \"resume\": path}"})
            return
        if length > MAX_BODY_BYTES:
            self._send_json(413, {"error": "body_too_large", "limit": MAX_BODY_BYTES})
            return

        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._send_json(400, {"error": "invalid_json", "detail": str(exc)})
            return

        jd = payload.get("jd")
        resume = payload.get("resume")
        if not jd or not resume:
            self._send_json(400, {"error": "missing_fields", "need": ["jd", "resume"]})
            return

        fmt = str(payload.get("format") or "md").lower()
        if fmt not in ("md", "json"):
            self._send_json(400, {"error": "bad_format", "allowed": ["md", "json"]})
            return

        try:
            result = self._analyze(str(jd), str(resume), payload)
        except FileNotFoundError as exc:
            self._send_json(404, {"error": "file_not_found", "detail": str(exc)})
            return
        except Exception as exc:  # noqa: BLE001 —— 服务不能因单个请求崩掉
            self._send_json(500, {"error": type(exc).__name__, "detail": str(exc)})
            return

        if fmt == "json":
            self._send_json(200, result["json"])
        else:
            self._send_text(200, result["markdown"], "text/markdown; charset=utf-8")

    # ---- 业务 ----
    def _healthz(self) -> Dict[str, Any]:
        """只读健康检查：**不加载模型、不建索引、不做检索**。"""
        from jobfit import siblings

        status = siblings.available_siblings()
        llm = getattr(self.server, "llm", None)
        return {
            "status": "ok",
            "uptime_seconds": round(time.time() - STARTED_AT, 1),
            "llm_backend": type(llm).__name__ if llm else "none",
            "siblings": status,
            "version": "0.1.0",
        }

    def _analyze(
        self,
        jd: str,
        resume: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """跑一次分析（每次请求独立建流水线，避免状态串味）。"""
        from jobfit.report import render_json, render_markdown

        llm: Optional[BaseLlm] = getattr(self.server, "llm", None)
        pipeline = JobFitPipeline(
            llm=llm,
            use_mcp=bool(payload.get("use_mcp", True)),
            max_questions=int(payload.get("max_questions", 8)),
        )
        report = pipeline.run(jd, resume)
        return {
            "markdown": render_markdown(report),
            "json": json.loads(render_json(report)),
        }


def serve(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    llm: Optional[BaseLlm] = None,
    block: bool = True,
) -> ThreadingHTTPServer:
    """启动服务。`block=False` 时立刻返回（给测试用）。"""
    server = ThreadingHTTPServer((host, port), _Handler)
    server.llm = llm  # type: ignore[attr-defined]
    if not block:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server
    print(f"job-fit-assistant 服务已启动：http://{host}:{port}/healthz")
    print(f"  POST http://{host}:{port}/analyze  {{\"jd\": \"...\", \"resume\": \"...\", \"format\": \"md\"}}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n收到 Ctrl-C，正在停止……")
    finally:
        server.server_close()
    return server
