"""pytest 共享夹具。

★ 全局约束：测试**不联网、不调 API、不加载真实模型**。
所有需要 LLM 的用例用桩（FakeLlm）代替——这既是速度要求，
也是「CI 上必须能跑」的硬要求（CI 没有 API key，也没有 HF 缓存）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class FakeLlm:
    """假的 LLM：按预设返回 JSON，记账但不花钱。

    它实现 jobfit.llm.BaseLlm 的接口子集（complete / usage），
    所以能直接塞进 pipeline / extractor。
    """

    def __init__(self, payload: Any = None, error: Exception | None = None) -> None:
        from jobfit.schemas import LlmUsage

        self.model = "fake-llm"
        self.budget_usd = 1.0
        self.usage = LlmUsage(backend="fake")
        self._payload = payload if payload is not None else []
        self._error = error
        self.calls: List[Dict[str, Any]] = []

    def complete(self, prompt: str, system: str = "", max_tokens: int = 900):
        from jobfit.llm import LlmError, LlmResponse

        self.calls.append({"prompt": prompt, "system": system})
        self.usage.calls += 1
        if self._error is not None:
            raise LlmError(str(self._error))
        text = self._payload if isinstance(self._payload, str) else json.dumps(
            self._payload, ensure_ascii=False
        )
        return LlmResponse(
            text=text,
            prompt_tokens=len(prompt) // 4,
            completion_tokens=len(text) // 4,
            cost_usd=0.0,
            price_known=True,
        )


@pytest.fixture
def fake_llm():
    """返回 FakeLlm 的工厂。"""

    def _make(payload: Any = None, error: Exception | None = None) -> FakeLlm:
        return FakeLlm(payload, error)

    return _make


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def jd_dir(repo_root: Path) -> Path:
    return repo_root / "data" / "jds"


@pytest.fixture(scope="session")
def resume_dir(repo_root: Path) -> Path:
    return repo_root / "data" / "resumes"


@pytest.fixture(scope="session")
def sample_jd(jd_dir: Path) -> Dict[str, Any]:
    """一份带人工标注的 JD（自写变体，文本稳定）。"""
    path = jd_dir / "authored_nlp_classification.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def sample_resume(resume_dir: Path) -> Path:
    return resume_dir / "简历202607-AI工程.md"


@pytest.fixture(scope="session")
def truth_table(repo_root: Path) -> Dict[str, Any]:
    path = repo_root / "data" / "annotations" / "resume_skill_truth.json"
    if not path.exists():
        pytest.skip("缺少人工标注真值表：先运行 scripts/annotate_resume_truth.py")
    return json.loads(path.read_text(encoding="utf-8"))
