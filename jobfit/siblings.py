"""三个地基仓库的本地路径集成。

★ 目录约定（README 里写清，CI 里用 actions/checkout 把三个仓库 checkout 到同一父目录）：

    <parent>/
      job-fit-assistant/     ← 本仓库
      hello-agent/           ← Agent 框架（trace / 韧性）
      rag-mcp-server/        ← MCP 工具服务（检索引擎）
      rag-eval-platform/     ← 评测平台（指标 / 报告）

环境变量 `JOBFIT_SIBLINGS_ROOT` 可以覆盖这个父目录（CI 里 checkout 到别处时用）。

**不复制粘贴任何兄弟仓库的代码**——本模块只负责把它们的包加进 sys.path，
然后由各功能模块直接 import 它们的类。这样「组合」是真组合：
上游改了行为，这里跟着变；本仓库的 git 历史里也不会出现别人的代码副本。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Optional

JOBFIT_ROOT = Path(__file__).resolve().parent.parent

#: 兄弟仓库在父目录下的名字
SIBLING_NAMES = ("hello-agent", "rag-mcp-server", "rag-eval-platform")


def siblings_root() -> Path:
    """三个兄弟仓库的公共父目录。"""
    env = os.environ.get("JOBFIT_SIBLINGS_ROOT", "").strip()
    if env:
        return Path(env).resolve()
    return JOBFIT_ROOT.parent


def sibling_path(name: str) -> Path:
    """某个兄弟仓库的路径。"""
    if name not in SIBLING_NAMES:
        raise ValueError(f"未知兄弟仓库 {name!r}，可选：{SIBLING_NAMES}")
    return siblings_root() / name


def _ensure_on_path(path: Path) -> bool:
    """把目录加进 sys.path（幂等）。返回是否成功。"""
    if not path.exists():
        return False
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)
    return True


def available_siblings() -> Dict[str, bool]:
    """哪些兄弟仓库在本机能找到。"""
    return {name: sibling_path(name).exists() for name in SIBLING_NAMES}


def require_sibling(name: str) -> Path:
    """要求某个兄弟仓库存在，否则给出可执行的错误信息。"""
    path = sibling_path(name)
    if not path.exists():
        raise FileNotFoundError(
            f"找不到兄弟仓库 {name}（期望路径：{path}）。"
            f" 请把它 checkout 到 {siblings_root()} 下，"
            " 或设置环境变量 JOBFIT_SIBLINGS_ROOT 指向三个仓库的公共父目录。"
        )
    return path


def setup_paths() -> Dict[str, bool]:
    """把所有能找到的兄弟仓库加进 sys.path。返回各仓库的可用状态。

    找不到的**不抛异常**——本仓库的 CLI 有「纯本地降级模式」，
    在只有本仓库的情况下也能跑（只是检索能力不可用）。
    """
    status: Dict[str, bool] = {}
    for name in SIBLING_NAMES:
        path = sibling_path(name)
        status[name] = _ensure_on_path(path)
    return status


def hello_agent_available() -> bool:
    """hello-agent 的 agent 包是否可导入。"""
    if not setup_paths()["hello-agent"]:
        return False
    try:
        import agent.trace  # noqa: F401

        return True
    except Exception:  # noqa: BLE001 —— 依赖缺失时视为不可用，不让导入炸掉
        return False


def rag_available() -> bool:
    """rag-mcp-server 的 rag 包是否可导入。"""
    if not setup_paths()["rag-mcp-server"]:
        return False
    try:
        import rag.chunking  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


def rag_eval_available() -> bool:
    """rag-eval-platform 的 src 包是否可导入。"""
    if not setup_paths()["rag-eval-platform"]:
        return False
    try:
        import src.evaluation  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False
