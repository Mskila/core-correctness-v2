"""Lazy adapter for the optional, pinned ``openai-codex`` reviewer SDK."""

from __future__ import annotations

import asyncio
import tempfile
from types import SimpleNamespace
from typing import Any, AsyncIterator, Callable, Protocol, cast

from .decision_reviewer import (
    CodexModelConfigurationError,
    CodexNotLoggedInError,
    CodexReviewerUnavailableError,
    CodexRuntimeResponseV1,
)


CODEX_SDK_VERSION = "0.144.4"
CODEX_REVIEW_MODEL = "gpt-5.6-sol"
CODEX_REASONING_EFFORT = "medium"
CODEX_SERVICE_TIER = "fast"
# The ChatGPT app-server catalog currently exposes Fast-capable routing under
# the internal accelerated-tier id ``priority``.  The request itself must
# still use the public Fast-mode configuration documented for Codex.
CODEX_CATALOG_ACCELERATED_TIER = "priority"

_DEVELOPER_INSTRUCTIONS = """你是 AlphaMaster 的只读交易候选审查器。
只能依据用户提供的结构化市场摘要和预先计算的订单方案作答。
不得调用任何工具，不得读取文件、网络或环境，不得计算或修改价格、方向、订单类型或手数。
候选是执行前理论方案，经纪商约束会在后续执行阶段校验；不得因尚未执行而拒绝。
只能 approve/reject；approve 时只能原样选择一个给定 plan_id。
只输出符合所给 JSON Schema 的对象，不得输出额外字段或解释。"""

_DISALLOWED_ITEM_TYPES = frozenset(
    {
        "hookPrompt",
        "plan",
        "commandExecution",
        "fileChange",
        "mcpToolCall",
        "dynamicToolCall",
        "collabAgentToolCall",
        "subAgentActivity",
        "webSearch",
        "imageView",
        "sleep",
        "imageGeneration",
        "enteredReviewMode",
        "exitedReviewMode",
        "contextCompaction",
    }
)


def _load_sdk() -> SimpleNamespace:
    try:
        from openai_codex import ApprovalMode, AsyncCodex, Sandbox, __version__
        from openai_codex.types import ReasoningEffort
    except (ImportError, ModuleNotFoundError) as error:
        raise CodexReviewerUnavailableError(
            "openai-codex is not installed; install requirements-codex.txt"
        ) from error
    return SimpleNamespace(
        __version__=__version__,
        AsyncCodex=AsyncCodex,
        ApprovalMode=ApprovalMode,
        Sandbox=Sandbox,
        ReasoningEffort=ReasoningEffort,
    )


def _enum_value(value: object) -> object:
    return getattr(value, "value", value)


def _account_kind(response: object) -> str | None:
    account = getattr(response, "account", None)
    if account is None:
        return None
    root = getattr(account, "root", account)
    kind = _enum_value(getattr(root, "type", None))
    return kind if type(kind) is str else None


def _verify_model_catalog(catalog: object) -> None:
    models = getattr(catalog, "data", ())
    selected = next(
        (
            model
            for model in models
            if getattr(model, "model", None) == CODEX_REVIEW_MODEL
            or getattr(model, "id", None) == CODEX_REVIEW_MODEL
        ),
        None,
    )
    if selected is None:
        raise CodexModelConfigurationError(
            f"exact reviewer model {CODEX_REVIEW_MODEL!r} is unavailable"
        )
    efforts = {
        _enum_value(getattr(option, "reasoning_effort", None))
        for option in getattr(selected, "supported_reasoning_efforts", ())
    }
    if CODEX_REASONING_EFFORT not in efforts:
        raise CodexModelConfigurationError(
            f"reviewer model does not support {CODEX_REASONING_EFFORT!r} reasoning"
        )
    tiers = {getattr(tier, "id", None) for tier in getattr(selected, "service_tiers", ()) or ()}
    if not tiers.intersection({CODEX_SERVICE_TIER, CODEX_CATALOG_ACCELERATED_TIER}):
        raise CodexModelConfigurationError(
            f"reviewer model does not support {CODEX_SERVICE_TIER!r} service tier"
        )


def _thread_item(payload: object) -> object | None:
    item = getattr(payload, "item", None)
    return getattr(item, "root", item)


class _TurnHandle(Protocol):
    def stream(self) -> AsyncIterator[object]: ...

    async def interrupt(self) -> object: ...


class OpenAICodexRuntime:
    """Create one fresh ephemeral Codex thread for each review request."""

    def __init__(self, *, sdk_loader: Callable[[], Any] = _load_sdk) -> None:
        self._sdk_loader = sdk_loader

    async def evaluate(
        self,
        request_json: str,
        output_schema: dict[str, object],
    ) -> CodexRuntimeResponseV1:
        sdk = self._sdk_loader()
        if getattr(sdk, "__version__", None) != CODEX_SDK_VERSION:
            raise CodexModelConfigurationError(
                f"openai-codex must be exactly {CODEX_SDK_VERSION}"
            )

        try:
            # On Windows the app-server process keeps its cwd open until the
            # SDK context closes, so the temporary directory must outlive it.
            with tempfile.TemporaryDirectory(prefix="alphamaster-codex-review-") as cwd:
                async with sdk.AsyncCodex() as codex:
                    account = await codex.account()
                    if _account_kind(account) != "chatgpt":
                        raise CodexNotLoggedInError(
                            "the reviewer requires the existing local ChatGPT login"
                        )
                    _verify_model_catalog(await codex.models())

                    thread = await codex.thread_start(
                        approval_mode=sdk.ApprovalMode.deny_all,
                        config={
                            "features": {"fast_mode": True},
                            "web_search": "disabled",
                        },
                        cwd=cwd,
                        developer_instructions=_DEVELOPER_INSTRUCTIONS,
                        ephemeral=True,
                        model=CODEX_REVIEW_MODEL,
                        sandbox=sdk.Sandbox.read_only,
                        service_tier=CODEX_SERVICE_TIER,
                    )
                    prompt = (
                        "审查下面的不可变候选快照。不得调用工具；只能选择给定 plan_id "
                        "或拒绝。严格按 JSON Schema 输出。\n" + request_json
                    )
                    handle = await thread.turn(
                        prompt,
                        approval_mode=sdk.ApprovalMode.deny_all,
                        effort=sdk.ReasoningEffort.medium,
                        model=CODEX_REVIEW_MODEL,
                        output_schema=output_schema,
                        sandbox=sdk.Sandbox.read_only,
                        service_tier=CODEX_SERVICE_TIER,
                    )
                    return await self._collect(handle)
        except (CodexNotLoggedInError, CodexModelConfigurationError):
            raise
        except asyncio.CancelledError:
            raise
        except Exception as error:
            raise CodexReviewerUnavailableError("Codex reviewer runtime failed") from error

    async def _collect(self, handle: object) -> CodexRuntimeResponseV1:
        turn_handle = cast(_TurnHandle, handle)
        rerouted = False
        used_tools = False
        last_agent_message: str | None = None
        final_agent_message: str | None = None
        completed = False
        try:
            async for event in turn_handle.stream():
                method = getattr(event, "method", "")
                payload = getattr(event, "payload", None)
                if method == "model/rerouted":
                    rerouted = True
                    continue
                if method == "item/completed":
                    item = _thread_item(payload)
                    item_type = getattr(item, "type", None)
                    if item_type in _DISALLOWED_ITEM_TYPES or item_type not in {
                        "userMessage",
                        "reasoning",
                        "agentMessage",
                    }:
                        used_tools = True
                    if item_type == "agentMessage":
                        text = getattr(item, "text", None)
                        if type(text) is str:
                            last_agent_message = text
                            if _enum_value(getattr(item, "phase", None)) == "final_answer":
                                final_agent_message = text
                    continue
                if method == "turn/completed":
                    turn = getattr(payload, "turn", None)
                    status = _enum_value(getattr(turn, "status", None))
                    if status != "completed":
                        error = getattr(getattr(turn, "error", None), "message", None)
                        raise CodexReviewerUnavailableError(
                            error or f"Codex turn ended with status {status!r}"
                        )
                    completed = True
        except asyncio.CancelledError:
            interrupt = getattr(turn_handle, "interrupt", None)
            if interrupt is not None:
                try:
                    await interrupt()
                except Exception:
                    pass
            raise

        if not completed:
            raise CodexReviewerUnavailableError("Codex turn did not report completion")
        raw_output = final_agent_message or last_agent_message
        if raw_output is None:
            raise CodexReviewerUnavailableError("Codex turn returned no final response")
        return CodexRuntimeResponseV1(
            raw_output=raw_output,
            used_tools=used_tools,
            rerouted=rerouted,
        )
