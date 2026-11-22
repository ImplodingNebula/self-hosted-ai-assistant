"""Tests for the FakeProvider, focused on the multi-call tool-loop scripting."""

from __future__ import annotations

from app.providers.base import ChatMessage, ManagementCapabilities, ModelCapabilities, ToolCall
from app.providers.fake_provider import FakeProvider, FakeTurn


async def _run(provider: FakeProvider, messages: list[ChatMessage]) -> list:
    return [chunk async for chunk in provider.chat_stream(messages)]


async def test_single_script_still_replays_every_call() -> None:
    # Backward compatibility: no turns -> the old behaviour, same script each call.
    provider = FakeProvider(script=["hi"])
    first = await _run(provider, [ChatMessage(role="user", content="a")])
    second = await _run(provider, [ChatMessage(role="user", content="b")])
    assert [c.type for c in first] == ["token", "done"]
    assert [c.type for c in second] == ["token", "done"]
    assert provider._call_count == 2


async def test_turns_script_a_tool_call_then_an_answer() -> None:
    call = ToolCall(id="1", name="web_search", arguments={"query": "weather"})
    provider = FakeProvider(
        turns=[FakeTurn(tool_calls=[call]), FakeTurn(tokens=["The", " answer."])]
    )
    first = await _run(provider, [ChatMessage(role="user", content="q")])
    assert [c.type for c in first] == ["tool_call", "done"]
    assert first[0].tool_call == call

    second = await _run(provider, [ChatMessage(role="user", content="q")])
    assert [c.type for c in second] == ["token", "token", "done"]
    assert "".join(c.delta for c in second if c.type == "token") == "The answer."


async def test_turns_record_the_messages_of_each_call() -> None:
    provider = FakeProvider(
        turns=[FakeTurn(tool_calls=[ToolCall(id="1", name="t")]), FakeTurn(tokens=["ok"])]
    )
    await _run(provider, [ChatMessage(role="user", content="first")])
    await _run(provider, [ChatMessage(role="user", content="second")])
    assert len(provider.calls) == 2
    assert provider.calls[0][0].content == "first"
    assert provider.calls[1][0].content == "second"


async def test_last_turn_repeats_once_exhausted() -> None:
    provider = FakeProvider(turns=[FakeTurn(tokens=["only"])])
    first = await _run(provider, [ChatMessage(role="user", content="a")])
    second = await _run(provider, [ChatMessage(role="user", content="b")])
    assert [c.type for c in first] == ["token", "done"]
    assert [c.type for c in second] == ["token", "done"]


async def test_turn_streams_thinking_then_tokens() -> None:
    provider = FakeProvider(turns=[FakeTurn(thinking=["reason"], tokens=["done"])])
    chunks = await _run(provider, [ChatMessage(role="user", content="a")])
    assert [c.type for c in chunks] == ["thinking", "token", "done"]


async def test_empty_turns_falls_back_to_the_script() -> None:
    # An empty turns list must not IndexError; it degrades to the default script.
    provider = FakeProvider(script=["hi"], turns=[])
    chunks = await _run(provider, [ChatMessage(role="user", content="a")])
    assert [c.type for c in chunks] == ["token", "done"]


def test_management_capabilities_default_to_a_registry_backend() -> None:
    # The fake stands in for the shipped provider, so its default must be the
    # registry shape - otherwise every management test would be the vLLM case.
    mgmt = FakeProvider().management_capabilities()
    assert (mgmt.can_pull, mgmt.can_delete, mgmt.can_list_installed) == (True, True, True)


def test_management_capabilities_are_scriptable() -> None:
    # The process-per-model runtime: nothing can be pulled or removed at run time.
    provider = FakeProvider(management=ManagementCapabilities())
    mgmt = provider.management_capabilities()
    assert (mgmt.can_pull, mgmt.can_delete, mgmt.can_list_installed) == (False, False, False)


async def test_runtime_version_is_scriptable() -> None:
    assert await FakeProvider().runtime_version() is None
    assert await FakeProvider(runtime_version="0.32.0").runtime_version() == "0.32.0"


async def test_default_capabilities_include_completion() -> None:
    # The fake IS the chat model wherever it stands in for a real provider, so
    # the flag that says "a conversation can run on this" has to be part of its
    # default report - otherwise every caller gating on it sees a deployment
    # with no chat model in it.
    caps = await FakeProvider().capabilities("fake")
    assert caps.completion is True
    assert (caps.tools, caps.embedding, caps.vision) == (True, True, False)


async def test_capabilities_are_scriptable_per_tag() -> None:
    # A real inventory mixes roles, so a fake that answers one fixed report for
    # every tag cannot stage the case a listing has to get right.
    provider = FakeProvider(
        capabilities_by_model={
            "llama3.1:8b": ModelCapabilities(completion=True, tools=True, context_length=131072),
            "nomic-embed-text:v1.5": ModelCapabilities(embedding=True),
        }
    )
    chat = await provider.capabilities("llama3.1:8b")
    assert (chat.completion, chat.embedding, chat.context_length) == (True, False, 131072)
    embedder = await provider.capabilities("nomic-embed-text:v1.5")
    assert (embedder.completion, embedder.embedding) == (False, True)
    # An unscripted tag still gets the fixed fallback.
    assert (await provider.capabilities("anything:1b")).completion is True
