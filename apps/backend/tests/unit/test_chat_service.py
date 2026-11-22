"""Chat service tests: prompt assembly, capping, routing, conversation resolution."""

from __future__ import annotations

import json
from uuid import uuid4

import pytest

from app.core.config import Settings
from app.core.errors import BadRequestError, NotFoundError
from app.core.tenancy import DEFAULT_TENANT_ID, DEFAULT_USER_ID, TenantContext
from app.models.entities import ModelPolicy
from app.providers.base import (
    ChatChunk,
    ModelCapabilities,
    ModelInfo,
    ProviderUnavailableError,
    ToolCall,
)
from app.providers.fake_provider import FakeProvider, FakeTurn
from app.providers.rag.embedder import FakeEmbedder
from app.repositories.workspace_settings import WorkspaceSettingsRepository
from app.schemas.chat import ChatRequest
from app.services.chat_service import stream_chat
from app.services.context_budget import approx_tokens
from app.services.model_facts import clear_facts_cache
from app.services.model_policy import clear_policy_cache
from app.services.prompts import DEFAULT_SYSTEM_PROMPT
from app.services.titles import derive_title
from tests.conftest import make_memory_store

TENANT = TenantContext(tenant_id=DEFAULT_TENANT_ID, user_id=DEFAULT_USER_ID, role="admin")

# The dev tier's configured chat pair. A fake that lists these is a normally
# provisioned machine: the turn routes to the configured model and no rescue
# path runs at all.
DEV_PAIR = ["qwen3:4b-instruct", "qwen3:4b"]


@pytest.fixture(autouse=True)
def _fresh_facts_cache():
    # Capability reports are cached per (provider name, tag) for five minutes,
    # which comfortably outlives a test session; each test stages its own.
    clear_facts_cache()
    yield
    clear_facts_cache()


class ListedFakeProvider(FakeProvider):
    """FakeProvider with a scripted inventory and scripted capability reports.

    Every listed tag reports it can complete unless named in ``embedders``;
    a tag in ``unprobeable`` refuses to answer at all, which is how "the
    runtime would not say" is staged (distinct from "it said no").
    """

    def __init__(
        self,
        installed: list[str],
        *,
        embedders: tuple[str, ...] = (),
        unprobeable: tuple[str, ...] = (),
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._installed = installed
        self._embedders = embedders
        self._unprobeable = unprobeable
        self.listed = 0

    async def list_models(self) -> list[ModelInfo]:
        self.listed += 1
        return [ModelInfo(name=name) for name in self._installed]

    async def capabilities(self, model: str) -> ModelCapabilities:
        if model in self._unprobeable:
            raise ProviderUnavailableError("The model server is unavailable.")
        return ModelCapabilities(
            completion=model not in self._embedders,
            embedding=model in self._embedders,
            tools=True,
        )


class UnlistableFakeProvider(FakeProvider):
    """FakeProvider whose model listing always fails (provider unreachable)."""

    async def list_models(self) -> list[ModelInfo]:
        raise ProviderUnavailableError("The model server is unavailable.")


def installed_provider(**kwargs: object) -> ListedFakeProvider:
    """A provider holding exactly the configured pair - the normal machine."""
    return ListedFakeProvider(list(DEV_PAIR), **kwargs)


def make_settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, model_provider="fake", **overrides)


async def run_stream(
    req: ChatRequest, provider: FakeProvider, settings: Settings, sessionmaker, **kwargs
) -> list[ChatChunk]:
    gen = await stream_chat(req, provider, settings, sessionmaker, TENANT, **kwargs)
    return [chunk async for chunk in gen]


async def stored_workspace_settings(sessionmaker) -> dict:
    """Read the tenant's workspace defaults back through a fresh session."""
    async with sessionmaker() as session:
        return await WorkspaceSettingsRepository(session, TENANT).load()


async def declare_thinking_pair(sessionmaker, *, model: str, thinking_model: str) -> None:
    """Store an operator-declared thinking sibling for ``model``.

    Stands in for the deleted tier profile's pairing: with no models.yaml, the
    empty-override rung of `resolve_thinking_sibling` falls through to a stored
    Model Manager declaration rather than a profile default, so a test of that
    rung needs one on the books.
    """
    async with sessionmaker() as session:
        session.add(ModelPolicy(provider="fake", model=model, thinking_model=thinking_model))
        await session.commit()
    clear_policy_cache()


# --- title derivation ---


def test_derive_title_collapses_whitespace() -> None:
    assert derive_title("  hello\n\tworld  ") == "hello world"


def test_derive_title_short_message_unchanged() -> None:
    text = "x" * 60
    assert derive_title(text) == text


def test_derive_title_truncates_at_word_boundary_with_ellipsis() -> None:
    title = derive_title("word " * 30)
    assert title.endswith("...")
    body = title.removesuffix("...")
    assert len(body) <= 60
    # No mid-word chop: the kept text is whole "word" repetitions.
    assert set(body.split()) == {"word"}


def test_derive_title_hard_cuts_single_long_word() -> None:
    title = derive_title("a" * 100)
    assert title == "a" * 60 + "..."


def test_derive_title_whitespace_only_is_none() -> None:
    # A null title is contract-legal; an empty string would be a third state
    # every client has to special-case (blank sidebar rows).
    assert derive_title(" \n\t ") is None


# --- prompt assembly ---


async def test_system_prompt_injected_exactly_once_and_first(db_sessionmaker) -> None:
    provider = installed_provider()
    req = ChatRequest(
        message="hi",
        history=[
            {"role": "user", "content": "earlier question"},
            {"role": "assistant", "content": "earlier answer"},
        ],
    )
    await run_stream(req, provider, make_settings(), db_sessionmaker)

    messages = provider.last_messages
    assert messages is not None
    assert messages[0].role == "system"
    assert messages[0].content == DEFAULT_SYSTEM_PROMPT
    assert sum(1 for m in messages if m.role == "system") == 1


async def test_legacy_history_in_order_before_new_message(db_sessionmaker) -> None:
    provider = installed_provider()
    req = ChatRequest(
        message="third",
        history=[
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "second"},
        ],
    )
    await run_stream(req, provider, make_settings(), db_sessionmaker)

    contents = [m.content for m in provider.last_messages if m.role != "system"]
    assert contents == ["first", "second", "third"]


async def test_disabled_system_prompt_is_omitted(db_sessionmaker) -> None:
    provider = installed_provider()
    await run_stream(
        ChatRequest(message="hi"), provider, make_settings(system_prompt=""), db_sessionmaker
    )
    assert all(m.role != "system" for m in provider.last_messages)


# --- history capping ---


async def test_history_capped_at_max_messages_keeping_newest(db_sessionmaker) -> None:
    provider = installed_provider()
    settings = make_settings()
    history = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"turn-{i}"} for i in range(30)
    ]
    await run_stream(
        ChatRequest(message="now", history=history), provider, settings, db_sessionmaker
    )

    non_system = [m for m in provider.last_messages if m.role != "system"]
    kept_history = non_system[:-1]
    assert len(kept_history) == settings.history_max_messages
    # Oldest dropped first: the survivors are the newest turns, in order.
    assert kept_history[0].content == "turn-10"
    assert kept_history[-1].content == "turn-29"


async def test_history_capped_by_token_budget_keeping_newest(db_sessionmaker) -> None:
    provider = installed_provider()
    # Budget 4000 tokens ~ 16000 chars: two 9000-char turns exceed it.
    history = [
        {"role": "user", "content": "a" * 9000},
        {"role": "assistant", "content": "b" * 9000},
        {"role": "user", "content": "c" * 9000},
    ]
    await run_stream(
        ChatRequest(message="now", history=history), provider, make_settings(), db_sessionmaker
    )

    non_system = [m for m in provider.last_messages if m.role != "system"]
    kept_history = non_system[:-1]
    assert len(kept_history) == 1
    assert kept_history[0].content.startswith("c")


async def test_very_long_history_is_truncated_not_rejected(db_sessionmaker) -> None:
    # A 250-turn transcript must validate (no schema-level cap) and degrade
    # gracefully to the newest turns instead of failing the request.
    provider = installed_provider()
    settings = make_settings()
    history = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"turn-{i}"} for i in range(250)
    ]
    req = ChatRequest(message="now", history=history)
    assert len(req.history) == 250
    await run_stream(req, provider, settings, db_sessionmaker)

    non_system = [m for m in provider.last_messages if m.role != "system"]
    kept_history = non_system[:-1]
    assert len(kept_history) == settings.history_max_messages
    assert kept_history[-1].content == "turn-249"


# --- model routing ---


async def test_thinking_on_routes_to_thinking_model(db_sessionmaker) -> None:
    # With no tier profile to supply a pair, the pair is now configured
    # EXPLICITLY through settings - there is no other source left.
    provider = installed_provider()
    await run_stream(
        ChatRequest(message="hi", thinking="on"),
        provider,
        make_settings(default_chat_model="qwen3:4b-instruct", thinking_chat_model="qwen3:4b"),
        db_sessionmaker,
    )
    assert provider.last_model == "qwen3:4b"
    assert provider.last_options is not None
    assert provider.last_options.think is True


async def test_thinking_off_routes_to_default_model(db_sessionmaker) -> None:
    provider = installed_provider()
    await run_stream(
        ChatRequest(message="hi", thinking="off"), provider, make_settings(), db_sessionmaker
    )
    assert provider.last_model == "qwen3:4b-instruct"
    # The instruct model never thinks (catalog: thinking false), so the flag is
    # omitted entirely instead of sent as False.
    assert provider.last_options.think is None


async def test_explicit_model_wins_but_think_flag_still_passed(db_sessionmaker) -> None:
    provider = installed_provider()
    await run_stream(
        ChatRequest(message="hi", model="qwen3:8b", thinking="on"),
        provider,
        make_settings(),
        db_sessionmaker,
    )
    assert provider.last_model == "qwen3:8b"
    # qwen3:8b is not in the dev-tier catalog -> mode unknown -> flag passes
    # through untouched.
    assert provider.last_options.think is True


async def test_thinking_without_pair_model_uses_default(db_sessionmaker) -> None:
    # The literal "none" disables pair-routing outright (empty means "use the
    # profile" since compose forwards an unset key as empty).
    provider = installed_provider()
    await run_stream(
        ChatRequest(message="hi", thinking="on"),
        provider,
        make_settings(thinking_chat_model="none"),
        db_sessionmaker,
    )
    assert provider.last_model == "qwen3:4b-instruct"
    # The requested thinking cannot happen on a never-thinking model; the flag
    # is dropped so the request is not rejected for carrying it.
    assert provider.last_options.think is None


async def test_thinking_with_empty_override_uses_the_declared_pair(db_sessionmaker) -> None:
    # "" is UNSET, not "disabled": with no tier profile left to supply a
    # sibling, the next rung down is a stored operator declaration (a Model
    # Manager pairing), so a thinking turn still finds its partner there.
    provider = installed_provider()
    await declare_thinking_pair(
        db_sessionmaker, model="qwen3:4b-instruct", thinking_model="qwen3:4b"
    )
    await run_stream(
        ChatRequest(message="hi", thinking="on"),
        provider,
        make_settings(default_chat_model="qwen3:4b-instruct", thinking_chat_model=""),
        db_sessionmaker,
    )
    assert provider.last_model == "qwen3:4b"
    assert provider.last_options.think is True


# --- missing-model routing and think normalization ---


async def test_missing_default_routes_to_thinking_pair_with_status(db_sessionmaker) -> None:
    # Only the thinking pair member is pulled (the other machine's failure
    # mode): a non-thinking turn must still answer - on that model, with its
    # reasoning parsed onto the thinking channel, and a status note first
    # among the narration (the meta frame naming the thread precedes it).
    # The pair is configured EXPLICITLY through settings - there is no tier
    # profile left to supply it - and only its thinking half is installed.
    provider = ListedFakeProvider(["qwen3:4b"])
    chunks = await run_stream(
        ChatRequest(message="hi", thinking="off"),
        provider,
        make_settings(default_chat_model="qwen3:4b-instruct", thinking_chat_model="qwen3:4b"),
        db_sessionmaker,
    )
    assert provider.last_model == "qwen3:4b"
    # Always-thinking model: think forced True so reasoning is parsed, never leaked.
    assert provider.last_options.think is True
    assert chunks[0].type == "meta"
    assert chunks[1].type == "status"
    assert chunks[1].status_code == "model_substituted"
    assert "qwen3:4b-instruct" in chunks[1].delta


async def test_missing_thinking_model_routes_to_default_with_status(db_sessionmaker) -> None:
    # Same explicit pair, this time with only its default half installed.
    provider = ListedFakeProvider(["qwen3:4b-instruct"])
    chunks = await run_stream(
        ChatRequest(message="hi", thinking="on"),
        provider,
        make_settings(default_chat_model="qwen3:4b-instruct", thinking_chat_model="qwen3:4b"),
        db_sessionmaker,
    )
    assert provider.last_model == "qwen3:4b-instruct"
    assert provider.last_options.think is None
    assert chunks[0].type == "meta"
    assert chunks[1].type == "status"
    assert chunks[1].status_code == "model_substituted"


async def test_explicit_model_is_never_substituted(db_sessionmaker) -> None:
    provider = ListedFakeProvider(["qwen3:4b-instruct"])
    chunks = await run_stream(
        ChatRequest(message="hi", model="ghost:1b"), provider, make_settings(), db_sessionmaker
    )
    assert provider.last_model == "ghost:1b"
    assert all(c.status_code != "model_substituted" for c in chunks)


async def test_neither_pair_member_installed_answers_on_what_the_server_holds(
    db_sessionmaker,
) -> None:
    # The configured pair is missing entirely and the server holds models the
    # catalog never heard of - the remote-host shape. The turn answers on the
    # first chat-capable tag the server listed and says so, rather than failing
    # on weights nobody has. Two tags, so the first-run seed cannot fire (it
    # only adopts a model when there is exactly one).
    # The pair is configured EXPLICITLY (no tier profile left to supply it),
    # and the server holds neither half.
    provider = ListedFakeProvider(["unrelated:1b", "other:3b"])
    chunks = await run_stream(
        ChatRequest(message="hi", thinking="off"),
        provider,
        make_settings(default_chat_model="qwen3:4b-instruct", thinking_chat_model="qwen3:4b"),
        db_sessionmaker,
    )
    assert provider.last_model == "unrelated:1b"
    note = next(c for c in chunks if c.status_code == "model_substituted")
    assert "qwen3:4b-instruct" in note.delta and "unrelated:1b" in note.delta


async def test_an_unprobeable_inventory_keeps_the_configured_model(db_sessionmaker) -> None:
    # The server lists models but will not say what any of them can do. Unknown
    # is not evidence: substituting onto an unknown tag, or refusing the turn,
    # would both be guesses - keep the configured model and let the generation
    # call report what is actually wrong. A model must be configured for there
    # to be anything to keep - an unconfigured deployment has no tag route
    # could substitute FROM.
    provider = ListedFakeProvider(
        ["mystery:1b", "mystery:3b"], unprobeable=("mystery:1b", "mystery:3b")
    )
    chunks = await run_stream(
        ChatRequest(message="hi", thinking="off"),
        provider,
        make_settings(default_chat_model="qwen3:4b-instruct"),
        db_sessionmaker,
    )
    assert provider.last_model == "qwen3:4b-instruct"
    assert all(c.status_code != "model_substituted" for c in chunks)


async def test_bare_tag_matches_installed_latest(db_sessionmaker) -> None:
    # Providers list bare names as name:latest; a configured bare tag must not
    # be treated as missing (which would trigger a bogus substitution).
    provider = ListedFakeProvider(["mistral:latest", "qwen3:4b"])
    chunks = await run_stream(
        ChatRequest(message="hi", thinking="off"),
        provider,
        make_settings(default_chat_model="mistral"),
        db_sessionmaker,
    )
    assert provider.last_model == "mistral"
    assert all(c.status_code != "model_substituted" for c in chunks)


async def test_pair_routing_disabled_still_reaches_the_general_fallback(db_sessionmaker) -> None:
    # THINKING_CHAT_MODEL=none means there is no partner at all, so the falsy
    # arm of the pair guard runs - and the turn must fall through to the
    # general fallback rather than stopping there with a model nobody has.
    provider = ListedFakeProvider(["unrelated:1b", "other:3b"])
    chunks = await run_stream(
        ChatRequest(message="hi", thinking="off"),
        provider,
        make_settings(thinking_chat_model="none"),
        db_sessionmaker,
    )
    assert provider.last_model == "unrelated:1b"
    assert any(c.status_code == "model_substituted" for c in chunks)


async def test_unlistable_provider_keeps_configured_model(db_sessionmaker) -> None:
    # If the provider cannot even be listed, routing must not guess; the chat
    # call itself will surface the connection failure. A model must be
    # configured for there to be anything to keep - with nothing configured
    # the resolved tag is itself blank, which is a different (first-run) case.
    provider = UnlistableFakeProvider()
    chunks = await run_stream(
        ChatRequest(message="hi", thinking="off"),
        provider,
        make_settings(default_chat_model="qwen3:4b-instruct"),
        db_sessionmaker,
    )
    assert provider.last_model == "qwen3:4b-instruct"
    assert all(c.status_code != "model_substituted" for c in chunks)


async def test_the_turn_lists_the_server_exactly_once(db_sessionmaker) -> None:
    # Three decisions need the inventory (honouring a stored pick, routing, the
    # first-run seed) and a listing can now cross the network, so the turn must
    # fetch it once and hand it around. This is the test that dies if any of
    # them starts listing for itself again.
    provider = ListedFakeProvider(["unrelated:1b", "other:3b"])
    await run_stream(
        ChatRequest(message="hi", thinking="off"),
        provider,
        # Deterministic keeps the earned-verdict lookup (which lists for the
        # digest, with its own cache) out of the count entirely.
        make_settings(agency_mode="deterministic"),
        db_sessionmaker,
    )
    assert provider.listed == 1


# --- nothing to chat with (the first-run failure) ---


async def test_an_empty_server_ends_the_turn_with_a_typed_error(db_sessionmaker) -> None:
    provider = ListedFakeProvider([])
    chunks = await run_stream(ChatRequest(message="hi"), provider, make_settings(), db_sessionmaker)
    # Still names its thread first: the question was saved, and a client that
    # cannot bind the failure to a conversation cannot retry in it either.
    assert chunks[0].type == "meta"
    assert chunks[0].conversation_id
    assert chunks[-1].type == "error"
    assert chunks[-1].error_code == "no_chat_model_configured"
    assert "Pull a chat model" in (chunks[-1].error or "")


async def test_an_embedder_only_server_ends_the_turn_with_a_typed_error(db_sessionmaker) -> None:
    provider = ListedFakeProvider(
        ["nomic-embed:latest", "bge-m3:latest"],
        embedders=("nomic-embed:latest", "bge-m3:latest"),
    )
    chunks = await run_stream(ChatRequest(message="hi"), provider, make_settings(), db_sessionmaker)
    assert chunks[-1].type == "error"
    assert chunks[-1].error_code == "no_chat_model_configured"
    # Nothing was adopted on the way out: an embedder is not a chat default.
    assert await stored_workspace_settings(db_sessionmaker) == {}


# --- the first-run seed of the workspace default ---


async def test_a_single_chat_model_becomes_the_workspace_default(db_sessionmaker) -> None:
    # Nothing configured that this server can run, and exactly one model that
    # can chat: the choice is not a choice, so it is made and remembered - and
    # THIS turn already runs on it, not the one after it.
    provider = ListedFakeProvider(
        ["llama3.1:8b", "nomic-embed:latest"], embedders=("nomic-embed:latest",)
    )
    await run_stream(ChatRequest(message="hi"), provider, make_settings(), db_sessionmaker)
    assert provider.last_model == "llama3.1:8b"
    # Read back through a fresh session: a merge that mutated the JSON column
    # in place would pass every in-session assertion and persist nothing.
    assert await stored_workspace_settings(db_sessionmaker) == {"model": "llama3.1:8b"}


async def test_two_chat_models_seed_nothing(db_sessionmaker) -> None:
    # With more than one candidate, picking would be a ranking - which this
    # platform does not do. The turn still answers (routing substitutes), but
    # nothing is written.
    provider = ListedFakeProvider(["llama3.1:8b", "llama3.2:3b"])
    await run_stream(ChatRequest(message="hi"), provider, make_settings(), db_sessionmaker)
    assert await stored_workspace_settings(db_sessionmaker) == {}


async def test_an_unreadable_model_seeds_nothing(db_sessionmaker) -> None:
    # One model answered, one would not say. "One of the ones we could ask
    # about" is a guess, and a guessed default outlives the turn that guessed.
    provider = ListedFakeProvider(["llama3.1:8b", "mystery:3b"], unprobeable=("mystery:3b",))
    await run_stream(ChatRequest(message="hi"), provider, make_settings(), db_sessionmaker)
    assert await stored_workspace_settings(db_sessionmaker) == {}


async def test_an_env_default_suppresses_the_seed(db_sessionmaker) -> None:
    provider = ListedFakeProvider(["llama3.1:8b"])
    await run_stream(
        ChatRequest(message="hi"),
        provider,
        make_settings(default_chat_model="qwen3:4b-instruct"),
        db_sessionmaker,
    )
    assert await stored_workspace_settings(db_sessionmaker) == {}


async def test_a_stored_workspace_default_suppresses_the_seed(db_sessionmaker) -> None:
    async with db_sessionmaker() as session:
        await WorkspaceSettingsRepository(session, TENANT).merge_defaults({"model": "chosen:1b"})
        await session.commit()
    provider = ListedFakeProvider(["llama3.1:8b"])
    await run_stream(ChatRequest(message="hi"), provider, make_settings(), db_sessionmaker)
    # The stored pick stands and is honoured, even though no catalog entry
    # describes it - the server does not hold it either, so the turn is
    # routed onto what the server does have, without rewriting the default.
    assert await stored_workspace_settings(db_sessionmaker) == {"model": "chosen:1b"}


async def test_a_configured_model_the_server_holds_suppresses_the_seed(db_sessionmaker) -> None:
    # Half the configured pair is pulled: the substitution already answers this
    # turn on the half that exists, and a seed here would outlive the missing
    # pull it was working around. The seed's gate is now simply "is anything
    # configured at all" (workspace default, DEFAULT_CHAT_MODEL, or a real
    # THINKING_CHAT_MODEL) - so the pair has to be configured for the gate to
    # close, even though only its thinking half is what the server holds.
    provider = ListedFakeProvider(["qwen3:4b"])
    await run_stream(
        ChatRequest(message="hi", thinking="off"),
        provider,
        make_settings(default_chat_model="qwen3:4b-instruct", thinking_chat_model="qwen3:4b"),
        db_sessionmaker,
    )
    assert await stored_workspace_settings(db_sessionmaker) == {}


async def test_the_seed_writes_once_and_then_stays_out_of_the_way(db_sessionmaker) -> None:
    provider = ListedFakeProvider(["llama3.1:8b"])
    settings = make_settings()
    await run_stream(ChatRequest(message="one"), provider, settings, db_sessionmaker)
    async with db_sessionmaker() as session:
        # A key another code version wrote, which the second turn must not eat.
        await WorkspaceSettingsRepository(session, TENANT).merge_defaults({"thinking": "on"})
        await session.commit()
    await run_stream(ChatRequest(message="two"), provider, settings, db_sessionmaker)
    assert await stored_workspace_settings(db_sessionmaker) == {
        "model": "llama3.1:8b",
        "thinking": "on",
    }


# --- context usage on done ---


async def test_done_reports_the_window_and_what_was_sent(db_sessionmaker) -> None:
    provider = installed_provider(script=["ok"])
    chunks = await run_stream(
        ChatRequest(message="how full is this"),
        provider,
        make_settings(agency_mode="deterministic"),
        db_sessionmaker,
    )
    usage = chunks[-1].context_usage
    assert usage is not None
    # The dev catalog's declared window for the routed model.
    assert usage.num_ctx == 8192
    # Exactly the prompt that went over the wire - no tools rode this turn.
    expected = sum(approx_tokens(m.content) for m in provider.calls[-1])
    assert usage.prompt_tokens == expected


async def test_a_tool_turn_reports_the_prompt_its_results_grew(db_sessionmaker) -> None:
    # The number must describe the LAST prompt, not the first: a tool loop
    # appends its results to the transcript, and reporting the opening prompt
    # would show a window emptying out exactly as it fills up.
    provider = installed_provider(
        turns=[
            FakeTurn(
                tool_calls=[
                    ToolCall(id="call-1", name="search_knowledge", arguments={"query": "refunds"})
                ]
            ),
            FakeTurn(tokens=["done"]),
        ]
    )
    chunks = await run_stream(
        ChatRequest(message="what do the records say about refunds"),
        provider,
        # A configured model is what gives the pre-routing capability lookup a
        # real tag to probe: with nothing configured, that lookup's candidate
        # is blank, its report is None, and the ceiling floors to
        # deterministic before routing ever substitutes a real tag in.
        make_settings(default_chat_model="qwen3:4b-instruct"),
        db_sessionmaker,
        vector_store=make_memory_store(),
        embedder=FakeEmbedder(dim=32),
    )
    assert len(provider.calls) == 2, "the tool loop did not re-enter"
    usage = chunks[-1].context_usage
    assert usage is not None
    # The schemas ride the wire alongside the messages whenever tools are
    # offered, and the runtime renders them into the prompt, so they count.
    assert provider.last_tools is not None
    schema_tokens = approx_tokens(json.dumps(provider.last_tools))
    first_round = sum(approx_tokens(m.content) for m in provider.calls[0]) + schema_tokens
    expected = sum(approx_tokens(m.content) for m in provider.calls[-1]) + schema_tokens
    assert usage.prompt_tokens == expected
    assert usage.prompt_tokens > first_round


async def test_an_error_terminal_carries_no_context_usage(db_sessionmaker) -> None:
    provider = installed_provider(script=[], error="model exploded", error_code="bad_request")
    chunks = await run_stream(
        ChatRequest(message="hi"),
        provider,
        make_settings(agency_mode="deterministic"),
        db_sessionmaker,
    )
    assert chunks[-1].type == "error"
    assert chunks[-1].context_usage is None


# --- chunk passthrough and done enrichment ---


async def test_chunks_reyielded_with_enriched_done(db_sessionmaker) -> None:
    # A configured model, so the turn routes directly with no substitution
    # status frame in the way - this test is about passthrough, not routing.
    provider = installed_provider(script=["a", "b"])
    chunks = await run_stream(
        ChatRequest(message="hi"),
        provider,
        make_settings(default_chat_model="qwen3:4b-instruct"),
        db_sessionmaker,
    )
    assert [c.type for c in chunks] == ["meta", "token", "token", "done"]
    done = chunks[-1]
    assert done.usage is not None
    # The done chunk carries the persisted ids for the SSE layer.
    assert done.conversation_id
    assert done.message_id


# --- conversation resolution (prepare phase) ---


async def test_malformed_conversation_id_raises_bad_request(db_sessionmaker) -> None:
    with pytest.raises(BadRequestError):
        await stream_chat(
            ChatRequest(message="hi", conversation_id="not-a-uuid"),
            installed_provider(),
            make_settings(),
            db_sessionmaker,
            TENANT,
        )


async def test_unknown_conversation_id_raises_not_found(db_sessionmaker) -> None:
    with pytest.raises(NotFoundError):
        await stream_chat(
            ChatRequest(message="hi", conversation_id=str(uuid4())),
            installed_provider(),
            make_settings(),
            db_sessionmaker,
            TENANT,
        )
