from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from trading_core.codex_runtime import (
    CODEX_CATALOG_ACCELERATED_TIER,
    CODEX_REASONING_EFFORT,
    CODEX_REVIEW_MODEL,
    CODEX_SDK_VERSION,
    CODEX_SERVICE_TIER,
    CodexModelConfigurationError,
    CodexNotLoggedInError,
    OpenAICodexRuntime,
)


class _Value:
    def __init__(self, value: str) -> None:
        self.value = value


class _FakeHandle:
    def __init__(self, raw_output: str, *, event_types: tuple[str, ...], rerouted: bool) -> None:
        self.id = "turn-1"
        self.raw_output = raw_output
        self.event_types = event_types
        self.rerouted = rerouted
        self.interrupted = False

    async def stream(self):
        if self.rerouted:
            yield SimpleNamespace(method="model/rerouted", payload=SimpleNamespace())
        for item_type in self.event_types:
            item = SimpleNamespace(type=item_type)
            if item_type == "agentMessage":
                item.text = self.raw_output
                item.phase = _Value("final_answer")
            yield SimpleNamespace(
                method="item/completed",
                payload=SimpleNamespace(turn_id=self.id, item=SimpleNamespace(root=item)),
            )
        yield SimpleNamespace(
            method="turn/completed",
            payload=SimpleNamespace(
                turn=SimpleNamespace(
                    id=self.id,
                    status=_Value("completed"),
                    error=None,
                )
            ),
        )

    async def interrupt(self) -> None:
        self.interrupted = True


class _FakeThread:
    def __init__(self, owner: "_FakeCodex") -> None:
        self.owner = owner

    async def turn(self, prompt: str, **kwargs: object) -> _FakeHandle:
        self.owner.turn_calls.append((prompt, kwargs))
        return _FakeHandle(
            self.owner.raw_output,
            event_types=self.owner.event_types,
            rerouted=self.owner.rerouted,
        )


class _FakeCodex:
    instances: list["_FakeCodex"] = []
    account_kind = "chatgpt"
    model = CODEX_REVIEW_MODEL
    efforts = (CODEX_REASONING_EFFORT,)
    tiers = (CODEX_CATALOG_ACCELERATED_TIER,)
    raw_output = "{}"
    event_types = ("userMessage", "reasoning", "agentMessage")
    rerouted = False

    def __init__(self) -> None:
        self.thread_start_calls: list[dict[str, object]] = []
        self.turn_calls: list[tuple[str, dict[str, object]]] = []
        type(self).instances.append(self)

    async def __aenter__(self) -> "_FakeCodex":
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    async def account(self):
        if self.account_kind == "none":
            return SimpleNamespace(account=None, requires_openai_auth=True)
        return SimpleNamespace(
            account=SimpleNamespace(root=SimpleNamespace(type=self.account_kind)),
            requires_openai_auth=True,
        )

    async def models(self):
        return SimpleNamespace(
            data=[
                SimpleNamespace(
                    id=self.model,
                    model=self.model,
                    supported_reasoning_efforts=[
                        SimpleNamespace(reasoning_effort=_Value(value))
                        for value in self.efforts
                    ],
                    service_tiers=[SimpleNamespace(id=value) for value in self.tiers],
                )
            ]
        )

    async def thread_start(self, **kwargs: object) -> _FakeThread:
        self.thread_start_calls.append(kwargs)
        return _FakeThread(self)


def _fake_sdk() -> SimpleNamespace:
    class _ApprovalMode:
        deny_all = "deny_all"

    class _Sandbox:
        read_only = "read_only"

    class _ReasoningEffort:
        medium = "medium"

    return SimpleNamespace(
        __version__=CODEX_SDK_VERSION,
        AsyncCodex=_FakeCodex,
        ApprovalMode=_ApprovalMode,
        Sandbox=_Sandbox,
        ReasoningEffort=_ReasoningEffort,
    )


@pytest.fixture(autouse=True)
def _reset_fake() -> None:
    _FakeCodex.instances = []
    _FakeCodex.account_kind = "chatgpt"
    _FakeCodex.model = CODEX_REVIEW_MODEL
    _FakeCodex.efforts = (CODEX_REASONING_EFFORT,)
    _FakeCodex.tiers = (CODEX_CATALOG_ACCELERATED_TIER,)
    _FakeCodex.raw_output = "{}"
    _FakeCodex.event_types = ("userMessage", "reasoning", "agentMessage")
    _FakeCodex.rerouted = False


def test_runtime_pins_login_model_effort_tier_and_fresh_ephemeral_thread() -> None:
    request_json = json.dumps({"decision_id": "decision-1"})
    output_schema: dict[str, object] = {"type": "object"}
    runtime = OpenAICodexRuntime(sdk_loader=_fake_sdk)

    first = asyncio.run(runtime.evaluate(request_json, output_schema))
    second = asyncio.run(runtime.evaluate(request_json, output_schema))

    assert first.raw_output == "{}"
    assert second.raw_output == "{}"
    assert len(_FakeCodex.instances) == 2
    for instance in _FakeCodex.instances:
        start = instance.thread_start_calls[0]
        assert start["model"] == CODEX_REVIEW_MODEL
        assert start["service_tier"] == CODEX_SERVICE_TIER
        assert start["ephemeral"] is True
        assert start["sandbox"] == "read_only"
        assert start["approval_mode"] == "deny_all"
        assert start["config"] == {
            "features": {"fast_mode": True},
            "web_search": "disabled",
        }

        prompt, turn = instance.turn_calls[0]
        assert request_json in prompt
        assert turn["model"] == CODEX_REVIEW_MODEL
        assert turn["effort"] == "medium"
        assert turn["service_tier"] == CODEX_SERVICE_TIER
        assert turn["sandbox"] == "read_only"
        assert turn["approval_mode"] == "deny_all"
        assert turn["output_schema"] == output_schema


def test_runtime_rejects_non_chatgpt_login_and_catalog_mismatch() -> None:
    runtime = OpenAICodexRuntime(sdk_loader=_fake_sdk)
    _FakeCodex.account_kind = "apiKey"
    with pytest.raises(CodexNotLoggedInError, match="ChatGPT"):
        asyncio.run(runtime.evaluate("{}", {"type": "object"}))

    _FakeCodex.account_kind = "chatgpt"
    _FakeCodex.efforts = ("low",)
    with pytest.raises(CodexModelConfigurationError, match="medium"):
        asyncio.run(runtime.evaluate("{}", {"type": "object"}))

    _FakeCodex.efforts = (CODEX_REASONING_EFFORT,)
    _FakeCodex.tiers = ("standard",)
    with pytest.raises(CodexModelConfigurationError, match="fast"):
        asyncio.run(runtime.evaluate("{}", {"type": "object"}))


def test_runtime_reports_tools_and_model_reroute_for_fail_closed_review() -> None:
    runtime = OpenAICodexRuntime(sdk_loader=_fake_sdk)
    _FakeCodex.event_types = ("reasoning", "commandExecution", "agentMessage")
    _FakeCodex.rerouted = True

    result = asyncio.run(runtime.evaluate("{}", {"type": "object"}))

    assert result.used_tools is True
    assert result.rerouted is True
