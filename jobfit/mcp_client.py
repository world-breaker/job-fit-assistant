"""MCP 客户端：通过 **stdio 协议**调用 jobfit 的 MCP 服务（jobfit/mcp_server.py）。

为什么走真的 MCP 协议而不是进程内直连：
- MCP 是「工具对 Agent 暴露」的标准方式，走一遍协议才能验证这条链路真的通；
- 进程隔离：embedding 模型只在子进程里驻留，主进程（Agent）保持轻量；
- 与 hello-agent 的做法一致（它也是用 MCP 客户端接 rag-mcp-server 的）。

★ 事件循环归属（踩过的坑）：
MCP 会话里的读写流绑定在**创建它的事件循环**上。如果每次调用都用
`asyncio.run(...)`（每次都新建并关闭一个循环），第二次调用就会挂在
「等一个已关闭循环上的流」上——表现是**第一次能通、后面全部卡死**。
因此本模块给客户端配一个**专属的后台线程 + 常驻事件循环**，
所有协程都通过 `run_coroutine_threadsafe` 提交到那一个循环上执行。

★ 降级策略（韧性）：MCP 连接失败（子进程起不来 / 握手超时）时，
**不抛异常**，而是返回 `available=False` 并给出原因。上层（pipeline）会记到 trace 里，
并退回到「不依赖检索的本地匹配路径」。工具调用失败也走同一套降级。
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

JOBFIT_ROOT = Path(__file__).resolve().parent.parent

#: MCP 服务脚本路径
MCP_SERVER_SCRIPT = JOBFIT_ROOT / "jobfit" / "mcp_server.py"

#: 连接超时（秒）——拉起子进程 + 握手 + 拉工具列表的硬上限
CONNECT_TIMEOUT_S = float(os.environ.get("JOBFIT_MCP_CONNECT_TIMEOUT", "180"))


class McpUnavailable(RuntimeError):
    """MCP 不可用（连接失败）。上层据此降级，不当致命错误。"""


class McpClient:
    """jobfit MCP 服务的客户端。

    用法：
        client = McpClient()
        client.start()                 # 起后台线程 + 常驻事件循环
        client.connect()               # 建立 MCP 会话（失败抛 McpUnavailable）
        payload = client.call("search_resume", {"query": "..."})
        client.close()
    """

    def __init__(self, connect_timeout_s: float = CONNECT_TIMEOUT_S) -> None:
        self.connect_timeout_s = connect_timeout_s
        self.tools: List[str] = []
        self.available = False
        self.reason = ""
        self.connect_ms = 0.0

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._session = None
        self._stack = None
        self._started = False

    # ---------------- 事件循环生命周期 ----------------

    def start(self) -> None:
        """起后台线程 + 常驻事件循环（幂等）。"""
        if self._started:
            return
        ready = threading.Event()

        def _runner() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            ready.set()
            loop.run_forever()

        self._thread = threading.Thread(target=_runner, name="jobfit-mcp-loop", daemon=True)
        self._thread.start()
        ready.wait(timeout=10)
        self._started = True

    def _run(self, coro, timeout: Optional[float] = None):
        """把协程提交到专属事件循环上执行。"""
        if self._loop is None:
            raise McpUnavailable("事件循环未启动")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout if timeout is not None else self.connect_timeout_s)

    def close(self) -> None:
        """关闭会话并停掉事件循环。"""
        if self._loop is not None and self._session is not None:
            try:
                self._run(self._close_session(), timeout=15)
            except Exception:  # noqa: BLE001 —— 关闭失败不影响主流程
                pass
        self._session = None
        if self._loop is not None:
            try:
                self._loop.call_soon_threadsafe(self._loop.stop)
            except Exception:  # noqa: BLE001
                pass
        if self._thread is not None:
            self._thread.join(timeout=10)
        self._thread = None
        self._loop = None
        self._started = False
        self.available = False

    async def _close_session(self) -> None:
        if self._stack is not None:
            try:
                await self._stack.aclose()
            except Exception:  # noqa: BLE001
                pass
        self._stack = None

    # ---------------- 连接与调用 ----------------

    def connect(self) -> None:
        """建立 MCP 会话。失败抛 McpUnavailable（调用方决定降级）。"""
        self.start()
        started = time.perf_counter()
        try:
            self._run(self._connect_async())
        except McpUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 —— 任何失败都转成「不可用」，让上层降级
            raise McpUnavailable(f"MCP 连接失败：{type(exc).__name__}: {exc}") from exc
        self.available = True
        self.connect_ms = round((time.perf_counter() - started) * 1000, 2)

    async def _connect_async(self) -> None:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        if not MCP_SERVER_SCRIPT.exists():
            raise McpUnavailable(f"MCP 服务脚本不存在：{MCP_SERVER_SCRIPT}")

        params = StdioServerParameters(
            command=sys.executable,
            args=[str(MCP_SERVER_SCRIPT)],
            cwd=str(JOBFIT_ROOT),
            env={
                # MCP SDK 拉子进程时只传一小撮环境变量；不显式指定的话，
                # 子进程在中文 Windows 上会用 GBK 写 stderr → 日志乱码。
                "PYTHONUTF8": "1",
                "PYTHONIOENCODING": "utf-8",
                "PATH": os.environ.get("PATH", ""),
                "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
                "COMSPEC": os.environ.get("COMSPEC", ""),
                "PATHEXT": os.environ.get("PATHEXT", ""),
                "JOBFIT_SIBLINGS_ROOT": os.environ.get(
                    "JOBFIT_SIBLINGS_ROOT", str(JOBFIT_ROOT.parent)
                ),
                "JOBFIT_EMBED_DEVICE": os.environ.get("JOBFIT_EMBED_DEVICE", "cpu"),
                "JOBFIT_EMBED_CACHE": os.environ.get(
                    "JOBFIT_EMBED_CACHE", str(JOBFIT_ROOT / ".cache" / "embeddings")
                ),
                "HOME": os.environ.get("HOME", os.environ.get("USERPROFILE", "")),
                "USERPROFILE": os.environ.get("USERPROFILE", ""),
                "TEMP": os.environ.get("TEMP", ""),
                "TMP": os.environ.get("TMP", ""),
            },
        )

        from contextlib import AsyncExitStack

        self._stack = AsyncExitStack()
        try:
            read, write = await asyncio.wait_for(
                self._stack.enter_async_context(stdio_client(params)),
                timeout=self.connect_timeout_s,
            )
            session = await asyncio.wait_for(
                self._stack.enter_async_context(ClientSession(read, write)),
                timeout=self.connect_timeout_s,
            )
            await asyncio.wait_for(session.initialize(), timeout=self.connect_timeout_s)
            listed = await asyncio.wait_for(session.list_tools(), timeout=self.connect_timeout_s)
        except asyncio.TimeoutError as exc:
            await self._close_session()
            raise McpUnavailable(
                f"MCP 连接超时（>{self.connect_timeout_s}s）：子进程或握手卡住"
            ) from exc
        except Exception as exc:  # noqa: BLE001
            await self._close_session()
            raise McpUnavailable(f"MCP 连接失败：{type(exc).__name__}: {exc}") from exc

        self._session = session
        self.tools = [tool.name for tool in listed.tools]

    def call(self, name: str, args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """调用一个 MCP 工具，返回解析后的 JSON。"""
        if self._session is None:
            raise McpUnavailable("MCP 会话未建立")
        return self._run(self._call_async(name, args or {}))

    async def _call_async(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        result = await asyncio.wait_for(
            self._session.call_tool(name, args), timeout=self.connect_timeout_s
        )
        # MCP 工具返回的是内容块列表，取第一个 text 块
        parts = getattr(result, "content", None) or []
        text = ""
        for block in parts:
            if getattr(block, "type", "") == "text":
                text = getattr(block, "text", "") or ""
                break
        if not text:
            return {"error": "MCP 工具未返回文本内容"}
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"text": text}


def call_mcp_tools(
    client: McpClient,
    calls: List[Tuple[str, Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """批量调用 MCP 工具（串行——避免同时加载多份索引）。"""
    out: List[Dict[str, Any]] = []
    for name, args in calls:
        try:
            payload = client.call(name, args)
        except Exception as exc:  # noqa: BLE001 —— 工具失败不中断整体流程
            payload = {"error": f"{type(exc).__name__}: {exc}"}
        out.append({"tool": name, "args": args, "result": payload})
    return out
