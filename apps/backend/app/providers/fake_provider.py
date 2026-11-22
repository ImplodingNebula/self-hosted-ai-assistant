"""A deterministic ModelProvider for tests and offline development."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from app.providers.base import (
    ChatChunk,
    ChatMessage,
    EmbedOptions,
    GenerationOptions,
    ManagementCapabilities,
    ModelCapabilities,
    ModelInfo,
    ModelProvider,
    ModelResidency,
    ToolCall,
    Usage,
    installed_tag_matches,
)

FAKE_SERVING_ENDPOINT = "fake://in-process"
"""The host identity this fake serves from.

A real value rather than None because the fake IS a serving stack, just an
in-process one - and because None is the answer that certifies nothing, so a
suite built on it could never stage a valid probe certificate at all.
"""


@dataclass
class FakeTurn:
    """One ``chat_stream`` call's scripted output for a multi-call FakeProvider.

    The tool loop calls the provider more than once per turn - it asks, runs the
    tools the model requested, feeds the results back, and asks again - so a fake
    that replays one script on every call cannot exercise it. A list of these,
    passed as ``turns``, scripts each successive call independently: a turn that
    carries ``tool_calls`` makes the model "request" tools, and a later turn that
    carries ``tokens`` makes it answer.

    Attributes:
        tokens: Answer text streamed as ``token`` chunks this call.
        tool_calls: Tool calls requested this call, each a ``tool_call`` chunk.
        thinking: Reasoning streamed as ``thinking`` chunks before the tokens.
        finish_reason: The finish reason on this call's ``done`` chunk.
        completion_tokens: The count this call's usage reports; None (the
            default) counts the scripted tokens. Set it to stage a round whose
            count is not its visible text - a tool-call round writes tokens no
            ``tokens`` entry stands for.
        completion_ms: The writing time this call's usage reports; None (the
            default) mirrors a runtime that measures none.
    """

    tokens: list[str] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    thinking: list[str] = field(default_factory=list)
    finish_reason: str = "stop"
    completion_tokens: int | None = None
    completion_ms: float | None = None


class FakeProvider(ModelProvider):
    """Streams a scripted response; no network, fully reproducible.

    Failure modes are scriptable so error paths get the same deterministic
    treatment as the happy path. Each ``chat_stream`` call records its inputs
    (``last_messages``, ``last_model``, ``last_tools``, ``last_options``) so
    tests can assert on routing and prompt assembly.
    """

    def __init__(
        self,
        script: list[str] | None = None,
        *,
        thinking_script: list[str] | None = None,
        status: str | None = None,
        status_code: str = "loading_model",
        error: str | None = None,
        error_code: str | None = None,
        raise_exc: Exception | None = None,
        delay: float = 0.0,
        vision: bool = False,
        capabilities_by_model: dict[str, ModelCapabilities] | None = None,
        turns: list[FakeTurn] | None = None,
        models: list[ModelInfo] | None = None,
        management: ManagementCapabilities | None = None,
        runtime_version: str | None = None,
        serving_endpoint: str | None = FAKE_SERVING_ENDPOINT,
        residency: ModelResidency | None = None,
        resident_models: list[ModelResidency] | None = None,
        pull_script: list[dict] | None = None,
        pull_error: Exception | None = None,
        delete_error: Exception | None = None,
        completion_ms: float | None = None,
    ) -> None:
        """Create a fake provider that will stream ``script`` token-by-token.

        Args:
            script: Token deltas to stream; defaults to a small greeting. Ignored
                when ``turns`` is set.
            thinking_script: Reasoning deltas yielded before any tokens.
            status: When set, a status chunk (with ``status_code``) is yielded
                before everything else, mirroring a cold model load.
            status_code: Status code carried by the scripted status chunk.
            error: When set, an error chunk (with ``error_code``) is yielded
                after the scripted tokens and the stream stops (no done chunk).
            error_code: Error code carried by the scripted error chunk.
            raise_exc: When set, raised mid-stream after the first token to
                exercise the router's catch-all path.
            delay: Seconds slept before each streamed chunk, so tests can
                exercise SSE heartbeats and the per-tenant concurrency cap.
            vision: The vision capability this fake reports, so a test can drive
                the chat service's "this model cannot see images" path.
            capabilities_by_model: Per-tag capability reports, consulted by
                exact tag name before the fixed fallback. A real inventory is
                a set of tags with DIFFERENT capabilities - a chat model beside
                an embedder - and one fixed report for every tag cannot stage
                that at all, which is exactly the shape a listing derived from
                runtime reports has to be tested against.
            turns: Per-call scripts for the tool loop (see :class:`FakeTurn`).
                When set, the Nth ``chat_stream`` call plays the Nth turn (the
                last turn repeats once exhausted), so a single fake can request
                a tool on one call and answer on the next. Overrides ``script``.
            models: What ``list_models`` reports; defaults to one digestless
                synthetic model. A test seeds digest-bearing entries here to
                drive the digest-bound earned-verdict lookup.
            management: The management verbs this fake declares; defaults to
                all three (a registry-style backend, like the shipped one). An
                all-False value scripts the other kind of runtime - one that
                loads a model at process start and changes by redeployment -
                so a management surface can be tested against both.
            runtime_version: What ``runtime_version`` reports; None (the
                default) mirrors a runtime that exposes no version. Scriptable
                so a stale-verdict banner can be driven without a real upgrade.
            serving_endpoint: What ``serving_endpoint`` reports. Defaults to
                the fake's own constant rather than to None, because None means
                "certifies nothing" and a suite that could not stage a valid
                certificate would exercise only the clamped half of the earned
                resolution. Set it to another value to stage a verdict earned
                on a DIFFERENT server, and to None to stage a backend that
                names no host at all.
            residency: What ``residency`` reports; None (the default) mirrors a
                backend that reports no residency at all. Seeded so a probe
                that landed on the CPU - the reading that must never be trusted
                as a normal measurement - can be staged without a small GPU.
            resident_models: What ``resident_models`` reports; None (the
                default) mirrors a backend that reports no residency at all,
                where an empty list means "asked, and holding nothing".
            pull_script: Raw progress events ``pull_model`` yields; defaults to
                a two-event download. Set to drive byte-counter normalization.
            pull_error: Raised from ``pull_model`` after the scripted events,
                so a download failure mid-stream can be exercised.
            delete_error: Raised from ``delete_model`` instead of deleting, so
                the "the weights vanished between listing and deleting" race
                has a way to happen on purpose.
            completion_ms: The writing time the single-script ``done`` reports;
                None (the default) mirrors a runtime that measures none. A
                ``turns`` script sets its own per turn.
        """
        self._script = (
            script if script is not None else ["Hello", " from", " the", " fake", " model."]
        )
        self._thinking_script = thinking_script or []
        self._status = status
        self._status_code = status_code
        self._error = error
        self._error_code = error_code
        self._raise_exc = raise_exc
        self._delay = delay
        self._vision = vision
        self._capabilities_by_model = capabilities_by_model or {}
        self._turns = turns
        self._models = models
        self._management = (
            management
            if management is not None
            else ManagementCapabilities(
                can_pull=True, can_delete=True, can_list_installed=True, can_unload=True
            )
        )
        self._runtime_version = runtime_version
        self._serving_endpoint = serving_endpoint
        self._residency = residency
        self._resident_models = resident_models
        self._pull_script = pull_script
        self._pull_error = pull_error
        self._delete_error = delete_error
        self._completion_ms = completion_ms
        self._call_count = 0
        # Recorded management calls, the same convention as ``last_model``.
        self.pulled: list[str] = []
        self.deleted: list[str] = []
        self.unloaded: list[str] = []
        self.last_messages: list[ChatMessage] | None = None
        self.last_model: str | None = None
        self.last_tools: list[dict] | None = None
        self.last_options: GenerationOptions | None = None
        self.last_embed_options: EmbedOptions | None = None
        # The messages seen on EACH chat_stream call, in order - so a tool-loop
        # test can assert that the second call carried the first call's tool
        # result. ``last_messages`` stays the most recent, for single-call tests.
        self.calls: list[list[ChatMessage]] = []

    async def chat_stream(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        tools: list[dict] | None = None,
        options: GenerationOptions | None = None,
    ) -> AsyncIterator[ChatChunk]:
        """Yield scripted status, thinking, tokens, then an error or ``done`` chunk.

        With ``turns`` set, each call instead plays the next :class:`FakeTurn`
        (tool calls and/or answer tokens), which is what makes the tool loop
        testable.
        """
        self.last_messages = messages
        self.last_model = model
        self.last_tools = tools
        self.last_options = options
        self.calls.append(list(messages))
        if self._turns:
            turn = self._turns[min(self._call_count, len(self._turns) - 1)]
            self._call_count += 1
            async for chunk in self._play_turn(turn):
                yield chunk
            return
        self._call_count += 1
        if self._status is not None:
            await self._pause()
            yield ChatChunk(type="status", delta=self._status, status_code=self._status_code)
        for delta in self._thinking_script:
            await self._pause()
            yield ChatChunk(type="thinking", delta=delta)
        for index, token in enumerate(self._script):
            await self._pause()
            yield ChatChunk(type="token", delta=token)
            if self._raise_exc is not None and index == 0:
                raise self._raise_exc
        await self._pause()
        if self._error is not None:
            yield ChatChunk(type="error", error=self._error, error_code=self._error_code)
            return
        yield ChatChunk(
            type="done",
            finish_reason="stop",
            usage=Usage(
                prompt_tokens=0,
                completion_tokens=len(self._script),
                completion_ms=self._completion_ms,
            ),
        )

    async def embed(
        self,
        texts: list[str],
        *,
        model: str | None = None,
        options: EmbedOptions | None = None,
    ) -> list[list[float]]:
        """Return a deterministic hash-based vector for each text."""
        self.last_embed_options = options
        return [self._hash_vector(text) for text in texts]

    async def list_models(self) -> list[ModelInfo]:
        """Return the seeded models, or a single synthetic one."""
        if self._models is not None:
            return list(self._models)
        return [ModelInfo(name="fake", capabilities=ModelCapabilities(tools=True, embedding=True))]

    async def pull_model(self, name: str) -> AsyncIterator[dict]:
        """Emit the scripted progress events, then any scripted failure."""
        self.pulled.append(name)
        script = (
            self._pull_script
            if self._pull_script is not None
            else [{"status": "pulling manifest"}, {"status": "success"}]
        )
        for event in script:
            await self._pause()
            yield dict(event)
        if self._pull_error is not None:
            raise self._pull_error

    async def delete_model(self, name: str) -> None:
        """Record the call and drop the tag from the scripted inventory.

        Dropping it matters: a delete that left ``list_models`` unchanged would
        let a test assert a success while proving nothing about what the
        provider was actually asked to do.
        """
        self.deleted.append(name)
        if self._delete_error is not None:
            raise self._delete_error
        if self._models is not None:
            self._models = [
                info for info in self._models if not installed_tag_matches(name, {info.name})
            ]

    async def capabilities(self, model: str) -> ModelCapabilities:
        """Report the scripted capabilities for ``model``, else the fixed ones.

        The fallback declares ``completion`` because the fake IS the chat model
        wherever this provider stands in for a real one: a caller that gates on
        that flag - the model listing deciding which tags a conversation can
        run on - would otherwise find every fake tag unable to hold a
        conversation, and the whole existing suite would be asserting against a
        deployment with no chat model in it.
        """
        scripted = self._capabilities_by_model.get(model)
        if scripted is not None:
            return scripted
        return ModelCapabilities(tools=True, embedding=True, vision=self._vision, completion=True)

    def management_capabilities(self) -> ManagementCapabilities:
        """Report the scripted management verbs (all three unless overridden)."""
        return self._management

    def serving_endpoint(self) -> str | None:
        """Report the scripted serving host (the fake's own, unless overridden)."""
        return self._serving_endpoint

    async def runtime_version(self) -> str | None:
        """Report the scripted runtime version (None unless one was seeded)."""
        return self._runtime_version

    async def residency(self, model: str) -> ModelResidency | None:
        """Report the scripted residency (None unless one was seeded)."""
        return self._residency

    async def resident_models(self) -> list[ModelResidency] | None:
        """Report the scripted loaded-model list (None unless one was seeded).

        Seeded independently of ``residency`` because the two answer different
        questions: one is asked about a named tag, the other about whatever the
        runtime happens to be holding - including a tag nothing describes.
        """
        return self._resident_models

    async def unload_model(self, model: str) -> bool:
        """Record the unload and report whether this fake declares the verb.

        Recording it in call order beside ``pulled``/``deleted`` is what lets a
        test assert that an unload happened BEFORE the work it was clearing the
        way for, rather than merely that both happened.
        """
        self.unloaded.append(model)
        return self._management.can_unload

    async def _play_turn(self, turn: FakeTurn) -> AsyncIterator[ChatChunk]:
        """Stream one scripted turn: thinking, then tokens, then tool calls, then done."""
        for delta in turn.thinking:
            await self._pause()
            yield ChatChunk(type="thinking", delta=delta)
        for token in turn.tokens:
            await self._pause()
            yield ChatChunk(type="token", delta=token)
        for call in turn.tool_calls:
            await self._pause()
            yield ChatChunk(type="tool_call", tool_call=call)
        await self._pause()
        yield ChatChunk(
            type="done",
            finish_reason=turn.finish_reason,
            usage=Usage(
                prompt_tokens=0,
                completion_tokens=(
                    turn.completion_tokens
                    if turn.completion_tokens is not None
                    else len(turn.tokens)
                ),
                completion_ms=turn.completion_ms,
            ),
        )

    async def _pause(self) -> None:
        """Sleep the scripted delay so the stream visibly spans time in tests."""
        if self._delay > 0:
            await asyncio.sleep(self._delay)

    @staticmethod
    def _hash_vector(text: str, dim: int = 16) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()
        return [byte / 255.0 for byte in digest[:dim]]
