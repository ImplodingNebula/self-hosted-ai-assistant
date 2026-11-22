"""Unit tests for settings parsing and system-prompt semantics."""

import pytest
from pydantic import ValidationError

from app.core.config import Settings, ollama_url_host, ollama_url_is_remote
from app.schemas.settings import MIN_CONTEXT_CEILING
from app.services.prompts import (
    DEFAULT_SYSTEM_PROMPT,
    append_tool_guidance,
    compose_system_prompt,
    tool_guidance,
)


def make_settings(**overrides: object) -> Settings:
    # _env_file=None keeps tests hermetic from any local .env file.
    return Settings(_env_file=None, **overrides)


def test_the_default_context_window_is_floored_like_the_workspace_ceiling() -> None:
    """Both numbers feed one slot, so one floor has to govern both.

    ``DEFAULT_CONTEXT_WINDOW`` and the stored workspace ceiling are the two
    ways a deployment names the window an unsized tag runs at, and only the
    ceiling had a floor. Below it a turn assembles a prompt the runtime then
    truncates from the head - silently - so a mistyped ``819`` has to be a
    refused boot rather than a deployment that quietly answers from nothing.

    The literal lives in ``core.config`` (settings must not import a schema
    module) and the equality is asserted here, so lowering one floor without
    the other fails rather than drifting.
    """
    with pytest.raises(ValidationError):
        make_settings(default_context_window=MIN_CONTEXT_CEILING - 1)
    assert make_settings(default_context_window=MIN_CONTEXT_CEILING).default_context_window == (
        MIN_CONTEXT_CEILING
    )


def test_cors_comma_separated_string() -> None:
    settings = make_settings(cors_allow_origins="http://a.com, http://b.com ,,")
    assert settings.cors_allow_origins == ["http://a.com", "http://b.com"]


def test_cors_json_array_string() -> None:
    settings = make_settings(cors_allow_origins='["http://a.com", "http://b.com"]')
    assert settings.cors_allow_origins == ["http://a.com", "http://b.com"]


def test_cors_plain_single_origin() -> None:
    # The exact value shipped in infra/.env.example.
    settings = make_settings(cors_allow_origins="http://localhost:3000")
    assert settings.cors_allow_origins == ["http://localhost:3000"]


def test_cors_from_env_comma_form(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CORS_ALLOW_ORIGINS", "http://a.com,http://b.com")
    assert Settings(_env_file=None).cors_allow_origins == ["http://a.com", "http://b.com"]


def test_cors_from_env_json_form(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CORS_ALLOW_ORIGINS", '["http://a.com"]')
    assert Settings(_env_file=None).cors_allow_origins == ["http://a.com"]


def test_cors_default_untouched() -> None:
    assert make_settings().cors_allow_origins == ["http://localhost:3000"]


def test_model_provider_typo_rejected() -> None:
    with pytest.raises(ValidationError):
        make_settings(model_provider="olama")


def test_model_provider_accepts_fake() -> None:
    assert make_settings(model_provider="fake").model_provider == "fake"


def test_thinking_model_empty_string_becomes_none() -> None:
    # "" reads as UNSET (defer to a stored declaration), never "disabled":
    # compose forwards an unset key as an empty string. Disabling is the
    # literal "none", interpreted by model_resolution.resolve_thinking_sibling.
    assert make_settings(thinking_chat_model="").thinking_chat_model is None


def test_chat_model_overrides_default_to_unset() -> None:
    # Both default to unset, and that is the first-run state rather than a hole
    # to be filled with a literal: with nothing stored either,
    # model_resolution.resolve_chat_models answers the empty string, and the
    # routing rescue adopts whatever chat model the server actually holds.
    assert make_settings().default_chat_model == ""
    assert make_settings().thinking_chat_model is None


def test_the_reranker_model_ships_a_default_and_reads_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The sidecar's weights are named here, because only its operator knows them.

    A rerank call has to carry a model name and nothing can look up what a
    sidecar was started with, so the tag is deployment configuration beside
    ``RERANKER_URL``. The default matches the ``--model-id`` the shipped
    compose service starts with; a deployment serving other weights says so.
    """
    assert make_settings().reranker_model == "bge-reranker-v2-m3"
    monkeypatch.setenv("RERANKER_MODEL", "bge-reranker-base")
    assert Settings(_env_file=None).reranker_model == "bge-reranker-base"


def test_an_upgraded_env_file_keeps_booting_with_its_retired_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A deployment upgrading into this release still has the old keys set.

    ``MODEL_TIER`` and ``MODELS_YAML_PATH`` were required configuration for
    several releases, so every existing ``.env`` and every running compose
    stack still carries them. They name nothing now - which has to mean they
    are IGNORED, not that they fail the boot: a strict settings model would
    turn a routine upgrade into a container that will not start, over two lines
    that describe something the code no longer has.

    The absence is asserted on the model itself as well, because the
    compose-environment audit reads ``Settings.model_fields`` to decide which
    keys must reach a service - a field left behind here would be demanded
    there.
    """
    monkeypatch.setenv("MODEL_TIER", "server-a10")
    monkeypatch.setenv("MODELS_YAML_PATH", "/etc/models.yaml")
    settings = Settings(_env_file=None)
    assert not hasattr(settings, "model_tier")
    assert not hasattr(settings, "models_yaml_path")
    assert "model_tier" not in Settings.model_fields
    assert "models_yaml_path" not in Settings.model_fields


def test_retention_defaults_change_nothing_for_files_a_person_sent() -> None:
    # The asymmetry is the policy: generated files expire, uploads do not until
    # somebody says so. An existing deployment upgrading into this must not lose
    # a single uploaded file to a default.
    settings = make_settings()
    assert settings.generated_file_retention_days == 30
    assert settings.attachment_retention_days is None


def test_a_blank_retention_window_reads_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    # Uncommenting the template line and leaving it empty is an ordinary way to
    # write "no policy". Read literally it is not a number, and it would fail the
    # boot of the API and the worker over a knob being turned OFF.
    monkeypatch.setenv("ATTACHMENT_RETENTION_DAYS", "")
    monkeypatch.setenv("GENERATED_FILE_RETENTION_DAYS", "  ")
    settings = Settings(_env_file=None)
    assert settings.attachment_retention_days is None
    assert settings.generated_file_retention_days == 30


def test_retention_windows_are_read_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GENERATED_FILE_RETENTION_DAYS", "0")
    monkeypatch.setenv("ATTACHMENT_RETENTION_DAYS", "365")
    settings = Settings(_env_file=None)
    # 0 is the documented off switch for the generated-file class, and it has to
    # survive as 0 rather than being folded back into the default by the
    # blank-value validator above.
    assert settings.generated_file_retention_days == 0
    assert settings.attachment_retention_days == 365


@pytest.mark.parametrize("key", ["GENERATED_FILE_RETENTION_DAYS", "ATTACHMENT_RETENTION_DAYS"])
def test_a_negative_retention_window_is_refused(key: str, monkeypatch: pytest.MonkeyPatch) -> None:
    # There is no honest reading of a negative window, and the alternative to
    # refusing it is a cutoff in the future that expires everything at once.
    monkeypatch.setenv(key, "-1")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


@pytest.mark.parametrize("key", ["GENERATED_FILE_RETENTION_DAYS", "ATTACHMENT_RETENTION_DAYS"])
def test_an_absurd_retention_window_is_refused_at_boot(
    key: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A fat-fingered digit past roughly a million days cannot even be built into
    # a timedelta, so unbounded it would raise OverflowError inside the hourly
    # sweep - taking the whole garbage-collection pass down, every hour, for a
    # value that plainly meant "never". Refused at boot, where it is one loud
    # error instead of a silent recurring one.
    monkeypatch.setenv(key, "1000000000")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_a_century_of_retention_is_still_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    # The ceiling has to sit well above any policy anyone would really write.
    monkeypatch.setenv("ATTACHMENT_RETENTION_DAYS", "36500")
    assert Settings(_env_file=None).attachment_retention_days == 36500


def test_auth_headers_unset_means_no_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    # Compose forwards an unset key as an EMPTY STRING, which is not JSON: the
    # whole backend would refuse to boot on a deployment that never set one.
    monkeypatch.setenv("OLLAMA_AUTH_HEADERS", "")
    assert Settings(_env_file=None).ollama_auth_headers == {}
    monkeypatch.setenv("OLLAMA_AUTH_HEADERS", "   ")
    assert Settings(_env_file=None).ollama_auth_headers == {}
    monkeypatch.delenv("OLLAMA_AUTH_HEADERS")
    assert Settings(_env_file=None).ollama_auth_headers == {}


def test_auth_headers_parse_from_json_object(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_AUTH_HEADERS", '{"Authorization": "Bearer changeme", "X-Key": "k"}')
    headers = Settings(_env_file=None).ollama_auth_headers
    assert headers == {"Authorization": "Bearer changeme", "X-Key": "k"}


def test_auth_headers_malformed_json_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    # Silently dropping a mistyped credential would leave the deployment
    # talking to a public model server with no credential and no complaint.
    monkeypatch.setenv("OLLAMA_AUTH_HEADERS", "Bearer changeme")
    with pytest.raises(ValidationError) as excinfo:
        Settings(_env_file=None)
    # The boot error goes to stderr and the container log; the value is a
    # credential, so the error must describe the SHAPE and withhold the value
    # (hide_input_in_errors plus the validator's un-chained ValueError - either
    # alone still leaks it through one path).
    assert "changeme" not in str(excinfo.value)


def test_auth_headers_non_object_json_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_AUTH_HEADERS", '["Authorization", "Bearer changeme"]')
    with pytest.raises(ValidationError) as excinfo:
        Settings(_env_file=None)
    # Same secrecy rule as the malformed case: valid JSON of the wrong shape
    # is still a credential and must not be echoed into the log.
    assert "changeme" not in str(excinfo.value)


@pytest.mark.parametrize(
    "url",
    [
        "http://ollama:11434",
        "http://ollama-gpu:11434",
        "http://ollama-cpu:11434",
        "http://localhost:11434",
        "http://127.0.0.1:11434",
        "http://[::1]:11434",
        "http://host.docker.internal:11434",
        # No parseable host at all: degrade to the pre-check behaviour rather
        # than warning about a value nobody can act on.
        "",
        "not a url",
    ],
)
def test_local_model_server_urls_are_not_remote(url: str) -> None:
    assert ollama_url_is_remote(url) is False


@pytest.mark.parametrize(
    "url",
    [
        "https://abc123-11434.proxy.example.net",
        "https://abc123-11434.proxy.example.net:443/api",
        "http://10.0.0.5:11434",
        # Only the HOST decides - credentials in the userinfo change nothing.
        "https://user:secret@gpu.example.net:11434",
    ],
)
def test_off_machine_model_server_urls_are_remote(url: str) -> None:
    assert ollama_url_is_remote(url) is True


def test_host_helper_reports_the_hostname_alone() -> None:
    # Never the full URL and never a credential: this value is served over the
    # API, so nothing but the host may survive the parse.
    assert ollama_url_host("https://user:secret@gpu.example.net:11434/api") == "gpu.example.net"
    assert ollama_url_host("http://[::1]:11434") == "::1"
    assert ollama_url_host("HTTP://OLLAMA:11434") == "ollama"
    assert ollama_url_host("") == ""


def test_system_prompt_unset_composes_default() -> None:
    assert compose_system_prompt(make_settings(), workspace_prompt=None) == DEFAULT_SYSTEM_PROMPT


def test_system_prompt_empty_disables() -> None:
    assert compose_system_prompt(make_settings(system_prompt=""), workspace_prompt=None) == ""


def test_system_prompt_override_used_verbatim() -> None:
    assert (
        compose_system_prompt(make_settings(system_prompt="Be terse."), workspace_prompt=None)
        == "Be terse."
    )


# The whole precedence table in one place: the workspace layer sits ABOVE the
# environment, and the "" kill switch stays env-only. Each row names the two
# stored inputs and the persona that must come out.
@pytest.mark.parametrize(
    ("env_prompt", "workspace_prompt", "expected"),
    [
        # Workspace beats a set environment prompt - the chain's whole point.
        ("Be terse.", "Speak as the support desk.", "Speak as the support desk."),
        # Workspace beats the built-in default too.
        (None, "Speak as the support desk.", "Speak as the support desk."),
        # Blank and whitespace-only fall THROUGH; they are not an override and
        # they are not a kill switch. A textarea holding spaces must behave
        # exactly like an empty one.
        ("Be terse.", "", "Be terse."),
        ("Be terse.", "   \n\t ", "Be terse."),
        (None, "", DEFAULT_SYSTEM_PROMPT),
        (None, None, DEFAULT_SYSTEM_PROMPT),
        # The env kill switch, undisturbed while nothing above it speaks.
        ("", None, ""),
        ("", "  ", ""),
        # ...and overridden when something above it does. A workspace value is
        # a persona, never an instruction about whether prompts exist.
        ("", "Speak as the support desk.", "Speak as the support desk."),
        # Surrounding whitespace never reaches the prompt.
        (None, "  Speak as the support desk.\n", "Speak as the support desk."),
    ],
)
def test_workspace_prompt_precedence(
    env_prompt: str | None, workspace_prompt: str | None, expected: str
) -> None:
    settings = make_settings() if env_prompt is None else make_settings(system_prompt=env_prompt)
    assert compose_system_prompt(settings, workspace_prompt=workspace_prompt) == expected


def test_tool_guidance_rides_on_a_workspace_persona() -> None:
    # The layering rule, held against the layer that is new: an operator who
    # rewrites the persona from the console must not silently lose the
    # honest-refusal and never-invent-a-tool instructions. Asserted on the
    # composed pair, not on append_tool_guidance alone, so a composition that
    # returned the workspace text by some path that skips the append would fail.
    persona = "Speak as the support desk."
    composed = compose_system_prompt(
        make_settings(system_prompt="Be terse."), workspace_prompt=persona
    )
    layered = append_tool_guidance(composed, web=True, knowledge=True)
    assert layered.startswith(persona)
    assert "Never invent a tool" in layered
    assert "web_search" in layered and "search_knowledge" in layered
    # And the environment's persona is genuinely gone, not merely outranked in
    # the first line.
    assert "Be terse." not in layered


def test_a_disabled_prompt_carries_no_guidance_without_a_workspace_persona() -> None:
    # The kill switch still kills the whole thing: "" disables the system
    # prompt entirely, guidance included, whenever no workspace value stands
    # above it.
    composed = compose_system_prompt(make_settings(system_prompt=""), workspace_prompt=None)
    assert append_tool_guidance(composed, web=True, knowledge=True) == ""


def test_tool_guidance_lists_only_offered_tools() -> None:
    both = tool_guidance(web=True, knowledge=True)
    assert "web_search" in both and "search_knowledge" in both
    web_only = tool_guidance(web=True, knowledge=False)
    assert "web_search" in web_only
    # No bullet may instruct calling a tool the turn does not offer - the
    # guidance itself would be inventing a tool.
    assert "search_knowledge" not in web_only
    kb_only = tool_guidance(web=False, knowledge=True)
    assert "search_knowledge" in kb_only and "web_search" not in kb_only
    # The web-attribution sentence dangles off a missing capability on a
    # knowledge-only turn, so it comes and goes with the web tool.
    assert "Never write a URL" in web_only and "Never write a URL" not in kb_only
    assert tool_guidance(web=False, knowledge=False) == ""


def test_append_tool_guidance_layers_onto_any_persona() -> None:
    # The default persona and a custom override both get the layer; the persona
    # text itself is untouched.
    layered = append_tool_guidance(DEFAULT_SYSTEM_PROMPT, web=True, knowledge=True)
    assert layered.startswith(DEFAULT_SYSTEM_PROMPT)
    assert "Never invent a tool" in layered
    custom = append_tool_guidance("Be terse.", web=True, knowledge=True)
    assert custom.startswith("Be terse.")
    assert "Never invent a tool" in custom


def test_append_tool_guidance_respects_disabled_prompt() -> None:
    # "" disables the system prompt entirely, guidance included.
    assert append_tool_guidance("", web=True, knowledge=True) == ""
