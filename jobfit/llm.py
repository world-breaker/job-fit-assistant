"""LLM 客户端：统一封装「调模型 → 拿结构化结果 → 记费用 → 超预算就停」。

三个后端，按可用性自动降级：
1. **deepseek**（默认）：走 OpenAI 兼容的 `/chat/completions`。环境变量任一可用即可：
   `DEEPSEEK_API_KEY` / `ANTHROPIC_AUTH_TOKEN`；`OPENAI_BASE_URL` 可覆盖端点。
2. **local**（离线兜底）：本机 HF 缓存里的 Qwen2.5-1.5B-Instruct，transformers 直接加载。
   显存紧张，**只在没有 API key 时启用**，且串行使用（不与 embedding 模型同时驻留）。
3. **none**：完全不用 LLM，所有步骤走规则兜底（评测里的对照组）。

成本记账：`LlmUsage` 记录 token 数与按公开价目表估算的美元成本；
价目表未命中的模型只报 token 不报钱（`price_known=False`）——**不编造成本**。

预算上限：`BudgetExceeded` 在**调用前**检查（预估输入 token + 上限输出 token），
超了就直接抛，不做「先花钱再发现超了」。
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from jobfit.schemas import LlmUsage

#: 公开价目表（USD / 1M tokens），来源：https://api-docs.deepseek.com/quick_start/pricing
#: 抓取日期：2026-10-21。该页有「峰值/非峰值」两档（峰值时段 01:00-04:00 与
#: 06:00-10:00 UTC，周一至周五），这里**统一按非峰值档记账**（保守方向：
#: 非峰值价 ≤ 峰值价，报出来的费用不会偏高）。要按峰值记账请改这里的数字。
#: 未命中的模型 → 只报 token 不报钱（不猜价格）。
PRICING: Dict[str, Dict[str, float]] = {
    # deepseek-flash（deepseek-v4-flash 是它的旧名，服务同一模型、同一价）
    "deepseek-v4-flash": {"input": 0.15, "output": 0.60},
    "deepseek-flash": {"input": 0.15, "output": 0.60},
    "deepseek-chat": {"input": 0.15, "output": 0.60},
    # deepseek-v4-pro
    "deepseek-v4-pro": {"input": 0.66, "output": 1.98},
}

#: 本地模型：没有价目表（不花钱，但也不是「零成本」——有本机算力成本）
LOCAL_MODELS = {"qwen2.5-1.5b-instruct"}

#: 默认预算上限（美元）。评测脚本会覆盖这个值。
DEFAULT_BUDGET_USD = 0.50

#: 单次调用的最大输出 token（结构化抽取不需要长文本）
#:
#: ★ 为什么是 1400 而不是 500：deepseek-v4-flash 默认开「思考模式」，
#:   思考 token 与输出 token **共用同一个 max_tokens 预算**。
#:   早期用 900 时，思考吃掉 819 token → 正文只剩 0 → 返回空字符串。
#:   修法有两条，这里两条都上：
#:   ① 请求带 `thinking: {"type": "disabled"}` 关掉思考模式（省 token 也更稳）；
#:   ② max_tokens 留足余量，万一思考没关掉也不会把正文挤没。
DEFAULT_MAX_TOKENS = 1400

#: 开启思考模式时的额外余量（思考 token 与输出共享预算）
THINKING_TOKEN_HEADROOM = 1200


class BudgetExceeded(RuntimeError):
    """预算用尽。抛出而不是静默截断——调用方必须知道「这一档没跑」。"""


class LlmError(RuntimeError):
    """LLM 调用或解析失败。"""


@dataclass
class LlmResponse:
    """一次 LLM 调用的结果。"""

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    price_known: bool = True
    backend: str = ""
    model: str = ""
    elapsed_ms: float = 0.0


def _estimate_tokens(text: str) -> int:
    """粗略 token 估算：中文按 1 字 ≈ 1 token，英文按 4 字符 ≈ 1 token。

    只用于**预算预检**，不用于记账——记账用 API 返回的真实 usage。
    """
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    other = len(text) - cjk
    return int(cjk + other / 4) + 8


def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> Tuple[float, bool]:
    """按价目表估成本。模型不在表里 → (0.0, False)，表示「价格未知，不报钱」。"""
    price = PRICING.get(model)
    if not price:
        return 0.0, False
    cost = (prompt_tokens / 1_000_000) * price["input"] + (completion_tokens / 1_000_000) * price["output"]
    return cost, True


class BaseLlm:
    """LLM 后端基类：统一预算检查与记账，子类只实现 `complete`。"""

    backend = "none"

    def __init__(self, model: str = "", budget_usd: float = DEFAULT_BUDGET_USD) -> None:
        self.model = model
        self.budget_usd = float(budget_usd)
        self.usage = LlmUsage(backend=self.backend)

    # ---------- 子类实现 ----------
    def _complete(self, prompt: str, system: str, max_tokens: int) -> LlmResponse:
        raise NotImplementedError

    # ---------- 公共逻辑 ----------
    def _precheck(self, prompt: str, system: str, max_tokens: int) -> None:
        """调用前检查预算。超了直接抛，不花钱。"""
        if self.budget_usd <= 0:
            raise BudgetExceeded(
                f"预算上限为 {self.budget_usd} USD（<=0 表示禁止调用 LLM）"
            )
        est_prompt = _estimate_tokens(prompt) + _estimate_tokens(system)
        est_cost, known = _estimate_cost(self.model, est_prompt, max_tokens)
        if known and self.usage.cost_usd + est_cost > self.budget_usd:
            raise BudgetExceeded(
                f"预算上限 {self.budget_usd:.4f} USD 将被超出："
                f"已用 {self.usage.cost_usd:.4f} + 预估 {est_cost:.4f} > {self.budget_usd:.4f}。"
                f"（模型 {self.model}）"
            )

    def complete(self, prompt: str, system: str = "", max_tokens: int = DEFAULT_MAX_TOKENS) -> LlmResponse:
        self._precheck(prompt, system, max_tokens)
        started = time.perf_counter()
        try:
            resp = self._complete(prompt, system, max_tokens)
        except BudgetExceeded:
            raise
        except Exception as exc:  # noqa: BLE001 —— 统一转成 LlmError，调用方据此走规则兜底
            raise LlmError(f"{type(exc).__name__}: {exc}") from exc
        resp.elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        resp.backend = self.backend
        resp.model = self.model
        self.usage.add(
            resp.prompt_tokens, resp.completion_tokens, resp.cost_usd, resp.price_known
        )
        return resp


class DeepSeekLlm(BaseLlm):
    """DeepSeek（OpenAI 兼容端点）。只用标准库 urllib，不引入 openai SDK 依赖。"""

    backend = "deepseek"

    def __init__(
        self,
        model: str = "deepseek-v4-flash",
        api_key: str = "",
        base_url: str = "",
        budget_usd: float = DEFAULT_BUDGET_USD,
        timeout_s: float = 120.0,
        disable_thinking: bool = True,
    ) -> None:
        super().__init__(model=model, budget_usd=budget_usd)
        self.api_key = api_key
        self.base_url = (base_url or "https://api.deepseek.com").rstrip("/")
        self.timeout_s = timeout_s
        self.disable_thinking = disable_thinking

    def _complete(self, prompt: str, system: str, max_tokens: int) -> LlmResponse:
        import urllib.request

        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,  # 抽取任务要可复现
        }
        if self.disable_thinking:
            # ★ 思考模式与输出共享 max_tokens 预算：开着思考时，
            #   短 JSON 抽取的正文会被思考 token 挤空（实测返回空字符串）。
            #   结构化抽取不需要推理链，直接关掉更稳也更省。
            payload["thinking"] = {"type": "disabled"}
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        text = message.get("content") or ""
        usage = data.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        cost, known = _estimate_cost(self.model, prompt_tokens, completion_tokens)
        if not text.strip():
            # 明确报出来：空正文不是「抽取到 0 条」，而是**调用没成功**。
            # 调用方据此走规则兜底，而不是把空结果当成「JD 里没有技能要求」。
            raise LlmError(
                f"模型返回空正文（finish_reason={choice.get('finish_reason')!r}，"
                f"usage={usage}）——思考 token 可能吃掉了全部预算"
            )
        return LlmResponse(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
            price_known=known,
        )


class LocalQwenLlm(BaseLlm):
    """本机离线模型（Qwen2.5-1.5B-Instruct，HF 缓存里已有，不下载）。

    只在没有 API key 时兜底。显存紧张是硬约束，因此：
    - `device` 默认 auto（有 CUDA 用 CUDA），但允许强制 CPU；
    - **不与 embedding 模型同时加载**——由调用方保证生命周期不重叠。
    """

    backend = "local"

    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-1.5B-Instruct",
        budget_usd: float = DEFAULT_BUDGET_USD,
        device: str = "auto",
    ) -> None:
        super().__init__(model=model, budget_usd=budget_usd)
        self.device = device
        self._tokenizer = None
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(self.model, local_files_only=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model,
            dtype=torch.float16 if device == "cuda" else torch.float32,
            local_files_only=True,
        ).to(device)
        self._model.eval()
        self.device = device

    def _complete(self, prompt: str, system: str, max_tokens: int) -> LlmResponse:
        import torch

        self._load()
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self.device)
        prompt_tokens = int(inputs["input_ids"].shape[-1])
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,  # 可复现
                pad_token_id=self._tokenizer.eos_token_id,
            )
        generated = outputs[0][prompt_tokens:]
        completion_text = self._tokenizer.decode(generated, skip_special_tokens=True).strip()
        # 本地模型没有「费用」，但 token 数照实报
        return LlmResponse(
            text=completion_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=int(generated.shape[-1]),
            cost_usd=0.0,
            price_known=True,  # 本地推理确实不产生 API 账单
        )

    def release(self) -> None:
        """主动卸载，释放显存。评测流程里换后端时调用。"""
        self._model = None
        self._tokenizer = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 —— 释放失败不该影响主流程
            pass


def _api_key_from_env() -> str:
    """按优先级取 API key：显式环境变量 → 兼容 DeepSeek 的 Anthropic 端点令牌。"""
    for name in ("DEEPSEEK_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


def build_llm(
    backend: str = "auto",
    model: str = "",
    budget_usd: float = DEFAULT_BUDGET_USD,
    device: str = "auto",
) -> Optional[BaseLlm]:
    """按可用性构建 LLM 后端。

    - `auto`：有 API key → deepseek；否则 → 本机离线模型；都没有 → None（全规则）。
    - `none`：显式不用 LLM。
    - 返回 None 表示「本次运行不调用 LLM」，调用方必须走规则兜底路径。
    """
    backend = (backend or "auto").lower()
    if backend == "none":
        return None

    api_key = _api_key_from_env()
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip() or os.environ.get(
        "DEEPSEEK_BASE_URL", ""
    ).strip()
    # Anthropic 兼容端点的 base_url 形如 https://api.deepseek.com/anthropic，
    # 它不是 OpenAI 兼容端点 → 这里把 /anthropic 后缀去掉。
    if base_url.endswith("/anthropic"):
        base_url = base_url[: -len("/anthropic")]

    if backend in ("auto", "deepseek") and api_key:
        chosen = model or os.environ.get("JOBFIT_LLM_MODEL", "deepseek-v4-flash")
        return DeepSeekLlm(model=chosen, api_key=api_key, base_url=base_url, budget_usd=budget_usd)

    if backend in ("auto", "local"):
        chosen = model or "Qwen/Qwen2.5-1.5B-Instruct"
        try:
            return LocalQwenLlm(model=chosen, budget_usd=budget_usd, device=device)
        except Exception:  # noqa: BLE001 —— 加载失败就当没有本地模型
            return None

    return None


def extract_json_array(text: str) -> List[Dict[str, Any]]:
    """从 LLM 输出里抽出 JSON 数组。

    小模型经常在 JSON 外面套 ```json 代码块或加前后废话——这里做宽容解析：
    先找最外层的 `[...]`，找不到再退回 `{...}` 数组包裹。
    解析失败抛 LlmError（调用方据此走规则兜底，并把「兜底」标出来）。
    """
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.S)
    if fence:
        cleaned = fence.group(1).strip()

    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(cleaned[start : end + 1])
            if isinstance(parsed, list):
                return [item for item in parsed if isinstance(item, dict)]
        except json.JSONDecodeError:
            pass

    # 退路：模型把对象包在 { } 里
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(cleaned[start : end + 1])
            if isinstance(parsed, dict):
                for key in ("requirements", "items", "skills", "data", "结果"):
                    if isinstance(parsed.get(key), list):
                        return [item for item in parsed[key] if isinstance(item, dict)]
        except json.JSONDecodeError:
            pass

    raise LlmError(f"无法从 LLM 输出解析出 JSON 数组：{text[:300]!r}")
