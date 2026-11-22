"""OllamaProvider tests over httpx.MockTransport - no real server needed."""

import base64
import json

import httpx
import pytest

from app.providers.base import (
    ChatChunk,
    ChatMessage,
    EmbedOptions,
    GenerationOptions,
    ImageContent,
    ModelNotFoundError,
    ProviderUnauthorizedError,
    ProviderUnavailableError,
    ToolCall,
)
from app.providers.ollama_provider import OllamaProvider, _eval_ms

MESSAGES = [ChatMessage(role="user", content="hi")]


def build_provider(handler) -> OllamaProvider:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return OllamaProvider(
        base_url="http://mock:11434", default_model="qwen3:4b-instruct", client=client
    )


def build_self_clienting_provider(handler, monkeypatch, **kwargs) -> OllamaProvider:
    """Provider that builds its OWN client, but over the mock transport.

    Credentials are applied when the provider constructs its client, so the
    injected-client helper above - which every other test here uses - would
    skip the code under test entirely.
    """
    real_client = httpx.AsyncClient

    def over_mock_transport(**client_kwargs) -> httpx.AsyncClient:
        return real_client(transport=httpx.MockTransport(handler), **client_kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", over_mock_transport)
    return OllamaProvider(base_url="http://mock:11434", default_model="qwen3:4b-instruct", **kwargs)


def ndjson_response(*objects: dict) -> httpx.Response:
    content = "\n".join(json.dumps(obj) for obj in objects)
    return httpx.Response(200, content=content.encode())


def happy_stream() -> httpx.Response:
    return ndjson_response(
        {"message": {"content": "ok"}},
        {"done": True, "done_reason": "stop", "prompt_eval_count": 5, "eval_count": 1},
    )


async def collect(provider: OllamaProvider, **kwargs) -> list[ChatChunk]:
    return [chunk async for chunk in provider.chat_stream(MESSAGES, **kwargs)]


async def test_404_yields_model_not_found_chunk() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "model 'missing:1b' not found"})

    chunks = await collect(build_provider(handler), model="missing:1b")
    assert len(chunks) == 1
    assert chunks[0].type == "error"
    assert chunks[0].error_code == "model_not_found"
    assert "missing:1b" in chunks[0].error


async def test_connect_error_yields_provider_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    chunks = await collect(build_provider(handler))
    assert chunks == [
        ChatChunk(
            type="error",
            error="Cannot reach the model server.",
            error_code="provider_unavailable",
        )
    ]


async def test_read_timeout_yields_provider_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow")

    chunks = await collect(build_provider(handler))
    assert chunks[0].error_code == "provider_unavailable"
    assert chunks[0].error == "The model timed out while generating."


async def test_500_yields_provider_unavailable_with_detail() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "model runner crashed"})

    chunks = await collect(build_provider(handler))
    assert chunks[0].error_code == "provider_unavailable"
    assert chunks[0].error == "model runner crashed"


CLOUDFLARE_524_PAGE = (
    '<!DOCTYPE html><html class="no-js" lang="en-US"><head>'
    "<title>runpod.net | 524: A timeout occurred</title></head><body>"
    + "The origin web server timed out responding to this request. " * 200
    + "</body></html>"
)


async def test_proxy_html_error_page_is_summarized_not_dumped() -> None:
    """A proxy's own error page must never reach a UI verbatim.

    Cloudflare's 524 page (the RunPod proxy's no-first-byte timeout) is ~8KB
    of markup; it landed whole in the tool-check UI on 2026-08-04. The detail
    must compress to the status code plus the page's own title.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            524,
            content=CLOUDFLARE_524_PAGE.encode(),
            headers={"content-type": "text/html"},
        )

    chunks = await collect(build_provider(handler))
    assert chunks[0].error_code == "provider_unavailable"
    assert "<" not in chunks[0].error
    assert "HTTP 524" in chunks[0].error
    assert "524: A timeout occurred" in chunks[0].error
    assert len(chunks[0].error) < 400


async def test_declared_html_is_summarized_even_without_leading_markup() -> None:
    """The declared content type is a signal in its own right.

    A markup sniff alone reads the first bytes, and an intermediary is free to
    open its page with a comment, a byte-order mark, or an XML declaration -
    at which point an 8KB document reaches a UI whole again. What the sender
    called the payload is the cheaper and more reliable of the two signals.
    """
    body = "<!-- generated by an edge cache -->\n" + ("<div>gateway timeout</div>" * 400)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            504, content=body.encode(), headers={"content-type": "text/html; charset=utf-8"}
        )

    chunks = await collect(build_provider(handler))
    assert "<" not in chunks[0].error
    assert "HTTP 504" in chunks[0].error
    assert len(chunks[0].error) < 200


async def test_plain_text_error_body_is_capped() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(502, content=b"upstream fell over " * 300)

    chunks = await collect(build_provider(handler))
    assert chunks[0].error_code == "provider_unavailable"
    assert chunks[0].error.endswith("[truncated]")
    assert len(chunks[0].error) < 350


async def test_plain_text_under_the_cap_passes_through_whole() -> None:
    """The clamp is a ceiling, not a rewrite: a short body reaches the reader."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            503, content=b"no runner available", headers={"content-type": "text/plain"}
        )

    chunks = await collect(build_provider(handler))
    assert chunks[0].error == "no runner available"


async def test_json_error_under_the_cap_passes_through_whole() -> None:
    """Ollama's own error shape is the common case and must survive untouched."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "model requires more system memory"})

    chunks = await collect(build_provider(handler))
    assert chunks[0].error == "model requires more system memory"
    assert "truncated" not in chunks[0].error


async def test_empty_error_body_falls_back_to_the_status_line() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(502, content=b"   ")

    chunks = await collect(build_provider(handler))
    assert chunks[0].error == "Model server returned HTTP 502."


async def test_oversized_json_error_is_capped_too() -> None:
    """Ollama's own error strings are short; a proxy impersonating the shape is not."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "y" * 5000})

    chunks = await collect(build_provider(handler))
    assert chunks[0].error.endswith("[truncated]")
    assert len(chunks[0].error) < 350


async def test_embed_proxy_html_error_is_summarized() -> None:
    """The sync extractor (embed, non-stream paths) applies the same summary."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(524, content=CLOUDFLARE_524_PAGE.encode())

    with pytest.raises(ProviderUnavailableError) as err:
        await build_provider(handler).embed(["hi"], model="qwen3-embedding:0.6b")
    assert "<" not in str(err.value)
    assert "HTTP 524" in str(err.value)


async def test_400_other_yields_bad_request() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "invalid options"})

    chunks = await collect(build_provider(handler))
    assert chunks[0].error_code == "bad_request"
    assert chunks[0].error == "invalid options"


async def test_400_no_thinking_support_retries_without_think() -> None:
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        calls.append(payload)
        if "think" in payload:
            return httpx.Response(400, json={"error": '"qwen2:0.5b" does not support thinking'})
        return happy_stream()

    chunks = await collect(build_provider(handler), options=GenerationOptions(think=True))
    assert len(calls) == 2
    assert calls[0]["think"] is True
    assert "think" not in calls[1]
    assert [c.type for c in chunks] == ["token", "done"]


async def test_thinking_deltas_map_to_thinking_chunks() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return ndjson_response(
            {"message": {"thinking": "hmm"}},
            {"message": {"thinking": " more"}},
            {"message": {"content": "answer"}},
            {"done": True, "done_reason": "stop", "prompt_eval_count": 5, "eval_count": 3},
        )

    chunks = await collect(build_provider(handler))
    assert [c.type for c in chunks] == ["thinking", "thinking", "token", "done"]
    assert chunks[0].delta == "hmm"
    assert chunks[2].delta == "answer"
    assert chunks[3].usage.completion_tokens == 3
    # A server that reports no writing time leaves it unmeasured.
    assert chunks[3].usage.completion_ms is None


async def test_the_servers_writing_time_arrives_in_milliseconds() -> None:
    # eval_duration is nanoseconds of decode time for eval_count tokens: the
    # one figure a speed can honestly be divided from.
    def handler(request: httpx.Request) -> httpx.Response:
        return ndjson_response(
            {"message": {"content": "answer"}},
            {
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 5,
                "eval_count": 30,
                "eval_duration": 1_500_000_000,
            },
        )

    usage = (await collect(build_provider(handler)))[-1].usage
    assert (usage.completion_tokens, usage.completion_ms) == (30, 1500.0)


def test_an_unusable_writing_time_reads_as_unmeasured() -> None:
    # Zero would divide into an infinite speed; a bool is an int in Python.
    for raw in (None, 0, -1, "1500000000", True, [1]):
        assert _eval_ms(raw) is None, raw


async def test_think_false_sent_top_level() -> None:
    captured: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return happy_stream()

    await collect(build_provider(handler), options=GenerationOptions(think=False))
    assert captured[0]["think"] is False


async def test_think_omitted_when_unset() -> None:
    captured: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return happy_stream()

    await collect(build_provider(handler), options=GenerationOptions())
    assert "think" not in captured[0]


async def test_generation_knobs_map_to_native_fields() -> None:
    captured: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return happy_stream()

    options = GenerationOptions(temperature=0.2, num_ctx=8192, max_tokens=256, keep_alive="5m")
    await collect(build_provider(handler), options=options)
    payload = captured[0]
    assert payload["options"] == {"temperature": 0.2, "num_ctx": 8192, "num_predict": 256}
    assert payload["keep_alive"] == "5m"


async def test_in_stream_error_yields_error_chunk() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": [{"name": "qwen3:4b-instruct"}]})
        return ndjson_response({"error": "model runner has unexpectedly stopped"})

    chunks = await collect(build_provider(handler))
    assert chunks[0].type == "error"
    assert chunks[0].error_code == "provider_unavailable"


async def test_status_chunk_when_model_not_loaded() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": []})
        return happy_stream()

    chunks = await collect(build_provider(handler))
    assert [c.type for c in chunks] == ["status", "token", "done"]
    assert chunks[0].status_code == "loading_model"
    assert "qwen3:4b-instruct" in chunks[0].delta


async def test_no_status_chunk_when_model_loaded() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": [{"name": "qwen3:4b-instruct"}]})
        return happy_stream()

    chunks = await collect(build_provider(handler))
    assert [c.type for c in chunks] == ["token", "done"]


async def test_ps_probe_failure_skips_status_quietly() -> None:
    # /api/ps errors must never block or pollute the chat stream.
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/ps":
            return httpx.Response(500, json={"error": "boom"})
        return happy_stream()

    chunks = await collect(build_provider(handler))
    assert [c.type for c in chunks] == ["token", "done"]


async def test_transport_error_yields_provider_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.RemoteProtocolError("peer closed connection")

    chunks = await collect(build_provider(handler))
    assert chunks == [
        ChatChunk(
            type="error",
            error="Lost the connection to the model server.",
            error_code="provider_unavailable",
        )
    ]


async def test_malformed_ndjson_yields_provider_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b'{"message": {"content": "ok"}}\n{truncated')

    chunks = await collect(build_provider(handler))
    assert [c.type for c in chunks] == ["token", "error"]
    assert chunks[1].error_code == "provider_unavailable"


async def test_list_models_malformed_body_raises_provider_unavailable() -> None:
    # A 200 with a non-JSON body (intercepting proxy, half-up server) must
    # surface as the seam-level error, never a raw parse exception.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>proxy error page</html>")

    with pytest.raises(ProviderUnavailableError):
        await build_provider(handler).list_models()


async def test_list_models_unexpected_shape_raises_provider_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=["not", "a", "mapping"])

    with pytest.raises(ProviderUnavailableError):
        await build_provider(handler).list_models()


# --- embed ---


async def test_embed_happy_path_payload_shape() -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2], [0.3, 0.4]]})

    vectors = await build_provider(handler).embed(["alpha", "beta"], model="qwen3-embedding:0.6b")
    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    assert captured[0].url.path == "/api/embed"
    payload = json.loads(captured[0].content)
    # The embedding tag is passed explicitly - never the chat default - and
    # no knobs leak into the payload when options are omitted.
    assert payload["model"] == "qwen3-embedding:0.6b"
    assert payload["input"] == ["alpha", "beta"]
    assert "options" not in payload
    assert "keep_alive" not in payload


async def test_embed_cpu_device_maps_to_zero_gpu_layers() -> None:
    captured: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={"embeddings": [[0.1]]})

    await build_provider(handler).embed(["a"], options=EmbedOptions(device="cpu"))
    assert captured[0]["options"] == {"num_gpu": 0}


async def test_embed_auto_device_omits_options_but_keeps_keep_alive() -> None:
    captured: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={"embeddings": [[0.1]]})

    await build_provider(handler).embed(["a"], options=EmbedOptions(device="auto", keep_alive="5m"))
    assert "options" not in captured[0]
    assert captured[0]["keep_alive"] == "5m"


async def test_embed_404_names_model_and_fix() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "model not found"})

    with pytest.raises(ProviderUnavailableError) as exc_info:
        await build_provider(handler).embed(["a"], model="qwen3-embedding:0.6b")
    # The message must be actionable: which model, and where to get it (the
    # deleted `make pull-models` target is gone; the Model Manager UI replaced it).
    assert "qwen3-embedding:0.6b" in str(exc_info.value)
    assert "/console/models" in str(exc_info.value)


async def test_embed_500_raises_provider_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "runner crashed"})

    with pytest.raises(ProviderUnavailableError):
        await build_provider(handler).embed(["a"])


async def test_embed_connect_error_raises_provider_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with pytest.raises(ProviderUnavailableError):
        await build_provider(handler).embed(["a"])


async def test_embed_malformed_body_raises_provider_unavailable() -> None:
    # A 200 without an "embeddings" key must surface as the seam-level error.
    def missing_key(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": []})

    with pytest.raises(ProviderUnavailableError):
        await build_provider(missing_key).embed(["a"])

    # ...and so must a vector count that does not match the input count.
    def wrong_length(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"embeddings": [[0.1]]})

    with pytest.raises(ProviderUnavailableError):
        await build_provider(wrong_length).embed(["a", "b"])


# --- tool calls across the seam (Phase 5, slice 2a) ---


def tool_handler(raw_call: dict):
    """Handler: model loaded on /api/ps, one tool call on the /api/chat done chunk."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": [{"name": "qwen3:4b-instruct"}]})
        return ndjson_response(
            {
                "message": {"tool_calls": [raw_call]},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 9,
                "eval_count": 0,
            },
        )

    return handler


async def test_tool_calls_parse_to_tool_call_chunks() -> None:
    # qwen3 on 0.32.0 returns the whole call on the done chunk (args a dict).
    raw = {
        "id": "abc123",
        "function": {"index": 0, "name": "get_weather", "arguments": {"city": "Paris"}},
    }
    chunks = await collect(build_provider(tool_handler(raw)))
    assert [c.type for c in chunks] == ["tool_call", "done"]
    assert chunks[0].tool_call == ToolCall(
        id="abc123", name="get_weather", arguments={"city": "Paris"}
    )


async def test_tool_call_arguments_as_json_string_are_parsed() -> None:
    # Some runtimes hand arguments back as a JSON string; normalize to a dict.
    raw = {"id": "x", "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}
    chunks = await collect(build_provider(tool_handler(raw)))
    assert chunks[0].tool_call.arguments == {"city": "Paris"}


async def test_tool_call_unparseable_arguments_degrade_to_empty_dict() -> None:
    raw = {"id": "x", "function": {"name": "f", "arguments": "not json"}}
    chunks = await collect(build_provider(tool_handler(raw)))
    assert chunks[0].tool_call.arguments == {}


async def test_tool_call_missing_id_is_minted() -> None:
    raw = {"function": {"name": "get_weather", "arguments": {}}}
    chunks = await collect(build_provider(tool_handler(raw)))
    assert chunks[0].tool_call.id  # non-empty minted id
    assert chunks[0].tool_call.name == "get_weather"


async def test_assistant_tool_calls_serialize_to_ollama_shape() -> None:
    captured: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return happy_stream()

    messages = [
        ChatMessage(role="user", content="weather?"),
        ChatMessage(
            role="assistant",
            content="",
            tool_calls=[ToolCall(id="1", name="get_weather", arguments={"city": "Paris"})],
        ),
        ChatMessage(role="tool", content="18C sunny", name="get_weather", tool_call_id="1"),
    ]
    provider = build_provider(handler)
    async for _ in provider.chat_stream(messages):
        pass
    sent = captured[0]["messages"]
    # Assistant tool call re-serialized into Ollama's {"function": {...}} shape.
    assert sent[1]["tool_calls"] == [
        {"id": "1", "function": {"name": "get_weather", "arguments": {"city": "Paris"}}}
    ]
    # Tool result keys on tool_name; the seam's tool_call_id stays off the wire.
    assert sent[2] == {"role": "tool", "content": "18C sunny", "tool_name": "get_weather"}
    # A plain message is unchanged from the old model_dump output.
    assert sent[0] == {"role": "user", "content": "weather?"}


async def test_message_images_serialize_to_ollama_base64() -> None:
    captured: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return happy_stream()

    pixels = b"\x89PNG\r\n\x1a\n" + bytes(range(16))
    messages = [
        ChatMessage(
            role="user",
            content="what is in this picture?",
            images=[ImageContent(data=pixels, media_type="image/png")],
        ),
        ChatMessage(role="assistant", content="a chart"),
    ]
    provider = build_provider(handler)
    async for _ in provider.chat_stream(messages):
        pass
    sent = captured[0]["messages"]
    # Raw seam bytes go out as Ollama's base64 images list; media_type stays
    # seam-side (Ollama sniffs the format from the bytes).
    assert sent[0]["images"] == [base64.b64encode(pixels).decode("ascii")]
    assert "media_type" not in json.dumps(sent[0])
    # An image-free message carries NO images key - the payload is unchanged.
    assert sent[1] == {"role": "assistant", "content": "a chart"}


def test_image_content_round_trips_through_json() -> None:
    # The confirmation flow checkpoints ChatMessages as JSON; pixels must
    # survive that round-trip byte-for-byte (base64, never utf-8 mangling).
    original = ChatMessage(
        role="user",
        content="see attached",
        images=[ImageContent(data=bytes(range(256)), media_type="image/png")],
    )
    # The checkpoint's EXACT pair: model_dump(mode="json") -> JSON column ->
    # model_validate on the parsed dict (chat_service checkpoint/restore).
    checkpointed = json.loads(json.dumps(original.model_dump(mode="json")))
    restored = ChatMessage.model_validate(checkpointed)
    assert restored.images is not None
    assert restored.images[0].data == bytes(range(256))
    assert restored.images[0].media_type == "image/png"
    # The direct JSON-string pair holds too.
    assert ChatMessage.model_validate_json(original.model_dump_json()) == original


def loaded_ps_handler(chat_response: httpx.Response):
    """Handler with the model loaded on /api/ps and a fixed /api/chat response."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": [{"name": "qwen3:4b-instruct"}]})
        return chat_response

    return handler


async def test_two_tool_calls_in_one_chunk_yield_two_chunks() -> None:
    # A model may request several tools in one turn; each becomes its own chunk.
    response = ndjson_response(
        {
            "message": {
                "tool_calls": [
                    {
                        "id": "a",
                        "function": {"name": "get_weather", "arguments": {"city": "Paris"}},
                    },
                    {"id": "b", "function": {"name": "search_web", "arguments": {"query": "news"}}},
                ]
            },
            "done": True,
            "done_reason": "stop",
        },
    )
    chunks = await collect(build_provider(loaded_ps_handler(response)))
    assert [c.type for c in chunks] == ["tool_call", "tool_call", "done"]
    assert [c.tool_call.name for c in chunks[:2]] == ["get_weather", "search_web"]
    assert [c.tool_call.id for c in chunks[:2]] == ["a", "b"]


async def test_non_dict_arguments_and_missing_name_degrade() -> None:
    raw = {"id": "x", "function": {"arguments": 42}}  # args not a str/dict, no name
    chunks = await collect(build_provider(tool_handler(raw)))
    assert chunks[0].tool_call.arguments == {}
    assert chunks[0].tool_call.name == ""


def test_management_capabilities_declare_the_registry_verbs() -> None:
    # A property of the backend, not of the server's health: no HTTP happens,
    # so the handler below is never reached.
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("management_capabilities must not perform IO")

    mgmt = build_provider(handler).management_capabilities()
    assert (mgmt.can_pull, mgmt.can_delete, mgmt.can_list_installed) == (True, True, True)
    # A hot-swapping runtime can drop a model from memory on request; a runtime
    # that reserves one model's memory at process start cannot, and says so.
    assert mgmt.can_unload is True


async def test_content_preceding_a_tool_call_is_not_swallowed() -> None:
    # If a runtime streams text and then a tool call on the same chunk, both appear.
    response = ndjson_response(
        {
            "message": {
                "content": "let me check",
                "tool_calls": [{"id": "1", "function": {"name": "f", "arguments": {}}}],
            },
            "done": True,
            "done_reason": "stop",
        },
    )
    chunks = await collect(build_provider(loaded_ps_handler(response)))
    assert [c.type for c in chunks] == ["token", "tool_call", "done"]
    assert chunks[0].delta == "let me check"


# --- management verbs: pull and delete ---


async def collect_pull(provider: OllamaProvider, name: str = "qwen3:4b") -> list[dict]:
    return [event async for event in provider.pull_model(name)]


async def test_pull_streams_every_progress_line() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/pull"
        assert json.loads(request.content) == {"model": "qwen3:4b", "stream": True}
        return ndjson_response(
            {"status": "pulling manifest"},
            {"status": "pulling sha256:a", "digest": "sha256:a", "total": 10, "completed": 4},
            {"status": "success"},
        )

    events = await collect_pull(build_provider(handler))
    assert [e["status"] for e in events] == ["pulling manifest", "pulling sha256:a", "success"]


async def test_pull_skips_a_truncated_progress_line() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b'{"status": "a"}\n{not json\n{"status": "b"}')

    assert [e["status"] for e in await collect_pull(build_provider(handler))] == ["a", "b"]


async def test_pull_404_raises_the_typed_not_found() -> None:
    # Not ProviderUnavailableError: the server answered perfectly well, and
    # reporting it as down would send someone to check the wrong thing.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "model not found"})

    with pytest.raises(ModelNotFoundError):
        await collect_pull(build_provider(handler), "ghost:1b")


async def test_pull_error_status_raises_provider_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "disk full"})

    with pytest.raises(ProviderUnavailableError, match="disk full"):
        await collect_pull(build_provider(handler))


async def test_pull_proxy_html_error_is_summarized() -> None:
    """Every path that lifts body text is clamped, not just the chat one.

    A download runs for minutes behind whatever proxy fronts the server, so it
    is at least as likely as a chat turn to be answered with a timeout page.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            524, content=CLOUDFLARE_524_PAGE.encode(), headers={"content-type": "text/html"}
        )

    with pytest.raises(ProviderUnavailableError) as err:
        await collect_pull(build_provider(handler))
    assert "<" not in str(err.value)
    assert "524: A timeout occurred" in str(err.value)


async def test_pull_transport_failure_never_leaks_httpx() -> None:
    # This feeds an SSE stream: a raw httpx error escaping here reaches a
    # browser as a dead connection with no event explaining it.
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with pytest.raises(ProviderUnavailableError):
        await collect_pull(build_provider(handler))


async def test_delete_sends_the_tag_in_the_body() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200)

    await build_provider(handler).delete_model("stray:1b")
    assert seen[0].method == "DELETE"
    assert seen[0].url.path == "/api/delete"
    assert json.loads(seen[0].content) == {"model": "stray:1b"}


async def test_delete_404_raises_the_typed_not_found() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "model 'ghost:1b' not found"})

    with pytest.raises(ModelNotFoundError, match="ghost:1b"):
        await build_provider(handler).delete_model("ghost:1b")


async def test_delete_error_status_raises_provider_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "cannot remove"})

    with pytest.raises(ProviderUnavailableError, match="cannot remove"):
        await build_provider(handler).delete_model("stray:1b")


async def test_delete_proxy_html_error_is_summarized() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            502, content=CLOUDFLARE_524_PAGE.encode(), headers={"content-type": "text/html"}
        )

    with pytest.raises(ProviderUnavailableError) as err:
        await build_provider(handler).delete_model("stray:1b")
    assert "<" not in str(err.value)
    assert "HTTP 502" in str(err.value)


async def test_delete_transport_failure_never_leaks_httpx() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with pytest.raises(ProviderUnavailableError):
        await build_provider(handler).delete_model("stray:1b")


# --- residency, and the deliberate unload ---


async def test_residency_reports_both_halves_of_where_a_model_sits() -> None:
    # `size` and `size_vram` are BOTH read: a VRAM figure alone cannot tell a
    # model that spilled entirely to the CPU from one that is not loaded at
    # all, and those two call for opposite reactions.
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/ps"
        return httpx.Response(
            200,
            json={
                "models": [
                    {"name": "other:1b", "size": 10, "size_vram": 10},
                    {"name": "qwen3:4b-instruct", "size": 4_000, "size_vram": 1_600},
                ]
            },
        )

    view = await build_provider(handler).residency("qwen3:4b-instruct")
    assert view is not None
    assert (view.loaded, view.total_bytes, view.vram_bytes) == (True, 4_000, 1_600)
    assert view.gpu_fraction == 0.4


async def test_a_model_the_server_does_not_hold_is_loaded_false_not_unknown() -> None:
    # A real answer, distinct from "this backend does not report residency" -
    # only the latter is None.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": []})

    view = await build_provider(handler).residency("qwen3:4b-instruct")
    assert view is not None and view.loaded is False


async def test_an_unreadable_ps_is_unmeasured_rather_than_zero() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    provider = build_provider(handler)
    assert await provider.residency("qwen3:4b-instruct") is None
    # ...and the narrow view derived from it says the same thing.
    assert await provider.resident_vram_bytes("qwen3:4b-instruct") is None


async def test_resident_models_lists_everything_loaded_including_the_uncatalogued() -> None:
    # The whole-runtime view exists so a page can show what is loaded without
    # asking about each catalogued tag in turn - a model nothing declares still
    # occupies the card, and is exactly the one worth seeing.
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/ps"
        return httpx.Response(
            200,
            json={
                "models": [
                    {"model": "qwen3:4b-instruct", "size": 4_000, "size_vram": 1_600},
                    {"model": "stray:1b", "size": 1_000, "size_vram": 0},
                ]
            },
        )

    readings = await build_provider(handler).resident_models()
    assert readings is not None
    assert [(r.model, r.vram_bytes) for r in readings] == [
        ("qwen3:4b-instruct", 1_600),
        ("stray:1b", 0),
    ]
    # A full CPU spill is a real reading of zero, not an absent one.
    assert readings[1].gpu_fraction == 0.0


async def test_resident_models_tells_holding_nothing_from_cannot_be_asked() -> None:
    def empty(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": []})

    def unreachable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    assert await build_provider(empty).resident_models() == []
    assert await build_provider(unreachable).resident_models() is None


async def test_unload_asks_for_a_zero_keep_alive() -> None:
    # There is no unload endpoint: a zero keep-alive on an empty generation is
    # how the runtime is told to release a model.
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"model": "qwen3:4b-instruct", "done": True})

    assert await build_provider(handler).unload_model("qwen3:4b-instruct") is True
    assert seen["path"] == "/api/generate"
    assert seen["body"]["keep_alive"] == 0
    assert "prompt" not in seen["body"]


async def test_a_failed_unload_reports_false_rather_than_raising() -> None:
    # Never the caller's error: an unload clears the way for work whose own
    # failure is the one worth reporting.
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    assert await build_provider(handler).unload_model("qwen3:4b-instruct") is False


# --- reaching a model server through somebody else's proxy ---

AUTH = {"Authorization": "Bearer changeme"}


def recording_handler(seen: list[httpx.Request], response_for_chat=None):
    """Handler that records every request; the model reads as already loaded."""

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": [{"name": "qwen3:4b-instruct"}]})
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": []})
        return response_for_chat or happy_stream()

    return handler


def request_for(seen: list[httpx.Request], path: str) -> httpx.Request:
    return next(r for r in seen if r.url.path == path)


async def test_configured_credentials_ride_on_every_call(monkeypatch) -> None:
    # The server authenticates nothing itself, so these are the credentials
    # whatever sits in front of it expects - and a call that forgets them is
    # simply refused.
    seen: list[httpx.Request] = []
    provider = build_self_clienting_provider(
        recording_handler(seen), monkeypatch, auth_headers=dict(AUTH)
    )
    await collect(provider)
    await provider.list_models()
    assert request_for(seen, "/api/chat").headers["authorization"] == "Bearer changeme"
    assert request_for(seen, "/api/tags").headers["authorization"] == "Bearer changeme"


async def test_no_credentials_configured_sends_none(monkeypatch) -> None:
    seen: list[httpx.Request] = []
    provider = build_self_clienting_provider(recording_handler(seen), monkeypatch)
    await collect(provider)
    await provider.list_models()
    assert "authorization" not in request_for(seen, "/api/chat").headers
    assert "authorization" not in request_for(seen, "/api/tags").headers


def redirect_handler(location: str = "https://pod-11434.proxy.example.net/api/chat"):
    """Every path answers a permanent redirect, as an http-only proxy would."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(301, headers={"location": location})

    return handler


def unauthorized_handler(status: int = 401):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"error": "token 'sk-live-abcdef' is not valid"})

    return handler


async def test_chat_redirect_names_the_address_and_the_setting() -> None:
    # Following it is not an option: the client would turn this POST into a
    # GET, so a redirect that "worked" would break chat silently.
    chunks = await collect(build_provider(redirect_handler()))
    assert [c.type for c in chunks] == ["error"]
    assert chunks[0].error_code == "provider_unavailable"
    assert "https://pod-11434.proxy.example.net/api/chat" in chunks[0].error
    assert "OLLAMA_URL" in chunks[0].error


async def test_list_models_redirect_raises_with_the_same_explanation() -> None:
    with pytest.raises(ProviderUnavailableError) as exc_info:
        await build_provider(redirect_handler()).list_models()
    message = str(exc_info.value)
    assert "https://pod-11434.proxy.example.net/api/chat" in message
    assert "OLLAMA_URL" in message


async def test_every_management_call_explains_a_redirect() -> None:
    # One helper, applied at each site: a verb that missed it would report a
    # perfectly healthy server as unreachable.
    provider = build_provider(redirect_handler())
    with pytest.raises(ProviderUnavailableError, match="OLLAMA_URL"):
        await provider.embed(["a"])
    with pytest.raises(ProviderUnavailableError, match="OLLAMA_URL"):
        await collect_pull(provider)
    with pytest.raises(ProviderUnavailableError, match="OLLAMA_URL"):
        await provider.delete_model("stray:1b")
    with pytest.raises(ProviderUnavailableError, match="OLLAMA_URL"):
        await provider.capabilities("qwen3:4b-instruct")


async def test_chat_unauthorized_gets_its_own_code_and_fix() -> None:
    chunks = await collect(build_provider(unauthorized_handler()))
    assert [c.type for c in chunks] == ["error"]
    assert chunks[0].error_code == "provider_unauthorized"
    assert "OLLAMA_AUTH_HEADERS" in chunks[0].error
    assert "401" in chunks[0].error
    # The upstream body can echo the credential it just refused, so none of it
    # is carried into a message a user reads.
    assert "sk-live-abcdef" not in chunks[0].error


async def test_list_models_unauthorized_raises_the_narrow_type() -> None:
    with pytest.raises(ProviderUnauthorizedError) as exc_info:
        await build_provider(unauthorized_handler(403)).list_models()
    message = str(exc_info.value)
    assert "OLLAMA_AUTH_HEADERS" in message
    assert "403" in message
    assert "sk-live-abcdef" not in message


async def test_every_management_call_explains_a_refused_credential() -> None:
    provider = build_provider(unauthorized_handler())
    with pytest.raises(ProviderUnauthorizedError, match="OLLAMA_AUTH_HEADERS"):
        await provider.embed(["a"])
    with pytest.raises(ProviderUnauthorizedError, match="OLLAMA_AUTH_HEADERS"):
        await collect_pull(provider)
    with pytest.raises(ProviderUnauthorizedError, match="OLLAMA_AUTH_HEADERS"):
        await provider.delete_model("stray:1b")
    with pytest.raises(ProviderUnauthorizedError, match="OLLAMA_AUTH_HEADERS"):
        await provider.capabilities("qwen3:4b-instruct")


def test_unauthorized_is_a_kind_of_unavailable() -> None:
    # Every existing "except ProviderUnavailableError" keeps working; only a
    # caller that can say something more useful narrows to the new type.
    assert issubclass(ProviderUnauthorizedError, ProviderUnavailableError)


async def test_an_ordinary_upstream_error_still_carries_its_detail() -> None:
    # The two new cases must not swallow what the server actually said about
    # any other status: the redirect and credential checks run BEFORE the
    # existing handling and must leave it alone.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "model runner crashed"})

    provider = build_provider(handler)
    with pytest.raises(ProviderUnavailableError, match="model runner crashed"):
        await provider.delete_model("stray:1b")
    with pytest.raises(ProviderUnavailableError, match="model runner crashed"):
        await provider.embed(["a"])
    chunks = await collect(provider)
    assert chunks[0].error_code == "provider_unavailable"
    assert chunks[0].error == "model runner crashed"


# --- capabilities(): the runtime's own report about one tag ---


def show_handler(payload: dict):
    """Answer /api/show with ``payload``; any other path is a test bug."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/show"
        return httpx.Response(200, json=payload)

    return handler


async def test_capabilities_read_the_flags_and_the_native_window() -> None:
    # The shape a live Ollama answers with: a flat capabilities array, and a
    # context length under a key named after the model's architecture.
    provider = build_provider(
        show_handler(
            {
                "capabilities": ["completion", "tools"],
                "model_info": {
                    "general.architecture": "llama",
                    "llama.context_length": 131072,
                },
            }
        )
    )
    caps = await provider.capabilities("llama3.1:8b")
    assert (caps.completion, caps.tools) == (True, True)
    assert (caps.vision, caps.embedding, caps.thinking) == (False, False, False)
    assert caps.context_length == 131072


async def test_context_length_key_is_built_from_the_reported_architecture() -> None:
    # The key is family-scoped, so reading it means reading the architecture
    # first. A hardcoded prefix would answer None for every other family, which
    # is indistinguishable from a runtime that reports no window at all - the
    # decoy key below is what a wrong prefix would pick up instead.
    provider = build_provider(
        show_handler(
            {
                "capabilities": ["completion", "tools", "thinking"],
                "model_info": {
                    "general.architecture": "qwen3",
                    "qwen3.context_length": 40960,
                    "llama.context_length": 131072,
                },
            }
        )
    )
    caps = await provider.capabilities("qwen3:4b")
    assert caps.context_length == 40960
    assert caps.thinking is True


async def test_an_embedding_only_model_does_not_report_completion() -> None:
    # The flag that separates a tag a conversation can run on from one that
    # can only turn text into vectors.
    provider = build_provider(show_handler({"capabilities": ["embedding"]}))
    caps = await provider.capabilities("qwen3-embedding:0.6b")
    assert caps.embedding is True
    assert caps.completion is False


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"model_info": "not a mapping"},
        {"model_info": {"llama.context_length": 131072}},
        {"model_info": {"general.architecture": "", ".context_length": 4096}},
        {"model_info": {"general.architecture": "llama"}},
        {"model_info": {"general.architecture": "llama", "llama.context_length": "131072"}},
        {"model_info": {"general.architecture": "llama", "llama.context_length": 0}},
        {"model_info": {"general.architecture": "llama", "llama.context_length": True}},
    ],
    ids=[
        "absent",
        "not-a-mapping",
        "no-architecture",
        "empty-architecture",
        "no-window-key",
        "string-window",
        "zero-window",
        "boolean-window",
    ],
)
async def test_an_unreadable_window_is_unknown_not_a_number(payload: dict) -> None:
    # A window is a positive integer or it is unknown: there is no zero-token
    # model, and a bogus one would size every context budget on the turn.
    caps = await build_provider(show_handler(payload)).capabilities("odd:1b")
    assert caps.context_length is None


# --- the descriptive fields, and the template's reasoning plumbing ---

# Abridged from the real thing, keeping the parts that decide the answer. This
# is the GO template the reasoning sibling of the dev pair ships: the server
# binds .IsThinkSet / .Think / .Thinking for it, and the template acts on them.
QWEN3_GO_TEMPLATE = """
{{- if .Messages }}
{{- if or .System .Tools }}<|im_start|>system
{{ .System }}
{{- if .Tools }}
# Tools
{{ range .Tools }}{{ . }}{{ end }}
{{- end }}<|im_end|>
{{ end }}
{{- range $i, $_ := .Messages }}
{{- if eq .Role "assistant" }}<|im_start|>assistant
{{- if and .Thinking $.IsThinkSet }}
<think>{{ .Thinking }}</think>
{{- end }}
{{ .Content }}<|im_end|>
{{ end }}
{{- end }}
{{- if and $.IsThinkSet (not $.Think) }}<think>

</think>
{{ end }}
{{- end }}
"""

# The trap, abridged from the real instruct tag: a JINJA template whose only
# <think> mentions STRIP reasoning out of history. Zero Go think actions, so
# it has no way to act on the flag - and matching the marker would read this
# never-reasoning model as a reasoning one.
QWEN3_INSTRUCT_JINJA_TEMPLATE = """
{%- if tools %}
    {{- '<|im_start|>system\\n' }}
{%- endif %}
{%- for message in messages %}
    {%- set content = message.content %}
    {%- if '<think>' in content %}
        {%- set content = content.split('</think>')[-1].lstrip('\\n') %}
    {%- endif %}
    {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' }}
{%- endfor %}
"""


async def test_the_details_map_fills_the_descriptive_fields() -> None:
    provider = build_provider(
        show_handler(
            {
                "capabilities": ["completion", "tools"],
                "details": {
                    "family": "qwen3",
                    "parameter_size": "4.0B",
                    "quantization_level": "Q4_K_M",
                },
            }
        )
    )
    caps = await provider.capabilities("qwen3:4b-instruct")
    assert (caps.family, caps.parameter_size, caps.quantization) == ("qwen3", "4.0B", "Q4_K_M")


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"details": "not a mapping"},
        {"details": {}},
        {"details": {"quantization_level": "   "}},
        {"details": {"quantization_level": 4}},
    ],
    ids=["absent", "not-a-mapping", "empty", "blank-string", "not-a-string"],
)
async def test_an_unreadable_detail_is_none_not_an_empty_string(payload: dict) -> None:
    caps = await build_provider(show_handler(payload)).capabilities("odd:1b")
    assert caps.quantization is None


async def test_a_go_template_wired_to_the_think_flag_reads_as_thinking() -> None:
    provider = build_provider(
        show_handler({"capabilities": ["completion"], "template": QWEN3_GO_TEMPLATE})
    )
    caps = await provider.capabilities("qwen3:4b")
    assert caps.template_thinking is True


async def test_a_jinja_template_that_only_strips_think_tags_reads_as_not_thinking() -> None:
    # The whole reason this reads the Go ACTIONS and not the <think> marker.
    # The instruct tag mentions <think> three times and cannot reason at all.
    provider = build_provider(
        show_handler(
            {
                # Ollama advertises the capability for this tag anyway, which is
                # exactly why the flag is not the evidence used here.
                "capabilities": ["completion", "tools", "thinking"],
                "template": QWEN3_INSTRUCT_JINJA_TEMPLATE,
            }
        )
    )
    caps = await provider.capabilities("qwen3:4b-instruct")
    assert "<think>" in QWEN3_INSTRUCT_JINJA_TEMPLATE
    assert caps.thinking is True
    assert caps.template_thinking is False


async def test_a_stub_template_with_a_builtin_renderer_is_unknown_not_a_no() -> None:
    # The shape the current generation of tags actually ships (verified live on
    # qwen3.5:9b): the template is a one-line stub and the real rendering lives
    # in code the modelfile names. There is nothing in that text to read, so
    # scanning it for think plumbing finds none and would report a model whose
    # think flag demonstrably toggles as unable to reason at all - a WRONG
    # answer where unknown is merely an unhelpful one.
    provider = build_provider(
        show_handler(
            {
                # Advertised, and deliberately NOT used as the fallback: the
                # flag describes the serving integration, not the weights.
                "capabilities": ["completion", "tools", "thinking", "vision"],
                "template": "{{ .Prompt }}",
                "modelfile": "FROM /root/.ollama/blobs/sha256-abc\nRENDERER qwen3.5\nPARSER qwen3.5\nPARAMETER temperature 1",
            }
        )
    )
    caps = await provider.capabilities("qwen3.5:9b")
    assert caps.thinking is True
    assert caps.template_thinking is None


async def test_a_stub_template_alone_is_unknown_even_without_a_renderer_line() -> None:
    # Either half of the evidence is enough to make the template unreadable:
    # a runtime that stops emitting the modelfile must not flip the answer
    # from unknown back to a confident "cannot reason".
    provider = build_provider(
        show_handler({"capabilities": ["completion"], "template": "{{ .Prompt }}"})
    )
    assert (await provider.capabilities("stub:1b")).template_thinking is None


async def test_a_readable_template_beside_a_renderer_line_is_still_unknown() -> None:
    # A built-in renderer means the TEMPLATE is not the artifact that would act
    # on a think flag, whatever the template happens to contain.
    provider = build_provider(
        show_handler(
            {
                "capabilities": ["completion"],
                "template": QWEN3_INSTRUCT_JINJA_TEMPLATE,
                "modelfile": "FROM x\nRENDERER qwen3.5\n",
            }
        )
    )
    assert (await provider.capabilities("odd:1b")).template_thinking is None


async def test_a_jinja_template_driving_reasoning_by_variable_reads_as_thinking() -> None:
    # The hybrid Qwen3 family ships this shape: same dialect as the instruct
    # tag above, but the reasoning is switched by a VARIABLE the caller sets.
    # Reading only Go actions would call this never-reasoning, which is worse
    # than saying nothing - the picker renders it as "answers directly".
    template = """
{%- if enable_thinking is defined and enable_thinking is false %}
    {{- '<think>\\n\\n</think>\\n\\n' }}
{%- endif %}
"""
    provider = build_provider(show_handler({"capabilities": ["completion"], "template": template}))
    assert (await provider.capabilities("qwen3:8b")).template_thinking is True


async def test_a_template_that_opens_the_reasoning_block_itself_reads_as_thinking() -> None:
    # An always-reasoning checkpoint has no flag to act on: its prompt simply
    # starts the block. The marker is in an OUTPUT expression here, which is
    # exactly what separates it from the instruct tag's conditions.
    template = "{%- for m in messages %}{{ m.content }}{%- endfor %}{{ '<think>\\n' }}"
    provider = build_provider(show_handler({"capabilities": ["completion"], "template": template}))
    assert (await provider.capabilities("deepseek-r1:7b")).template_thinking is True


@pytest.mark.parametrize(
    "payload",
    [{}, {"template": ""}, {"template": "   \n "}, {"template": 42}],
    ids=["absent", "empty", "blank", "not-a-string"],
)
async def test_no_template_is_unknown_not_a_no(payload: dict) -> None:
    # None and False are different diagnoses: one is a runtime that does not
    # expose templates, the other is a model that cannot reason.
    caps = await build_provider(show_handler(payload)).capabilities("odd:1b")
    assert caps.template_thinking is None
