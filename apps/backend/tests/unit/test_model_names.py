"""Tests for model-name validation - the SSRF and injection boundary.

Everything downstream of :func:`parse_model_name` builds a registry URL path or
hands the name to the model server, so this is the last point at which a typed
string can be refused. The rejection table below is deliberately exhaustive over
the reasons the parser knows about: a rule with no test is a rule that can be
deleted by accident.
"""

from __future__ import annotations

import pytest

from app.core.errors import UnprocessableEntityError
from app.services.model_names import (
    MAX_NAME_LENGTH,
    ModelName,
    ModelNameRejection,
    parse_model_name,
    require_model_name,
)


def accepted(raw: str) -> ModelName:
    parsed = parse_model_name(raw)
    assert isinstance(parsed, ModelName), f"expected {raw!r} to be accepted, got {parsed}"
    return parsed


def rejected(raw: str) -> ModelNameRejection:
    parsed = parse_model_name(raw)
    assert isinstance(parsed, ModelNameRejection), f"expected {raw!r} to be rejected"
    return parsed


# --- accepted names ---


@pytest.mark.parametrize(
    ("raw", "namespace", "repo", "tag"),
    [
        ("qwen3", "library", "qwen3", "latest"),
        ("qwen3:4b", "library", "qwen3", "4b"),
        ("qwen3:4b-instruct", "library", "qwen3", "4b-instruct"),
        ("qwen3-embedding:0.6b", "library", "qwen3-embedding", "0.6b"),
        ("library/qwen3:4b", "library", "qwen3", "4b"),
        ("some_org/some.model-2:q4_K_M", "some_org", "some.model-2", "q4_K_M"),
    ],
)
def test_library_names_split_into_their_parts(
    raw: str, namespace: str, repo: str, tag: str
) -> None:
    parsed = accepted(raw)
    assert (parsed.kind, parsed.namespace, parsed.repo, parsed.tag) == (
        "library",
        namespace,
        repo,
        tag,
    )
    assert parsed.warning is None


def test_surrounding_whitespace_is_trimmed_not_rejected() -> None:
    # A pasted name routinely arrives with a trailing space; refusing it would
    # be pedantry, but keeping it would make the in-flight claim key differ
    # from the same name typed cleanly.
    assert accepted("  qwen3:4b\t").text == "qwen3:4b"


def test_a_name_at_the_length_cap_is_accepted() -> None:
    assert accepted("a" * MAX_NAME_LENGTH).repo == "a" * MAX_NAME_LENGTH


@pytest.mark.parametrize(
    ("raw", "user", "repo", "tag"),
    [
        ("hf.co/user/repo", "user", "repo", "latest"),
        ("hf.co/user/repo:Q4_K_M", "user", "repo", "Q4_K_M"),
        ("huggingface.co/user/repo", "user", "repo", "latest"),
        # A browser copy-paste does not preserve case; the host part is a host.
        ("HF.CO/user/repo", "user", "repo", "latest"),
    ],
)
def test_huggingface_names_are_accepted_with_a_warning(
    raw: str, user: str, repo: str, tag: str
) -> None:
    parsed = accepted(raw)
    assert (parsed.kind, parsed.namespace, parsed.repo, parsed.tag) == (
        "huggingface",
        user,
        repo,
        tag,
    )
    # Accepted, but never silently: the runtime takes these names and the
    # registry cannot check them, so the caveat rides along with the parse.
    assert parsed.warning
    # The name is passed on exactly as typed - the parts are for the registry
    # lookup that will not happen, not a rewrite of what gets downloaded.
    assert parsed.text == raw.strip()


# --- rejected names ---


@pytest.mark.parametrize(
    ("raw", "code"),
    [
        ("", "empty"),
        ("   ", "empty"),
        ("\t\n", "empty"),
        ("a" * (MAX_NAME_LENGTH + 1), "too_long"),
        ("qwen\x003", "control_character"),
        ("qwen\n3", "control_character"),
        ("qwen\x7f3", "control_character"),
        ("http://registry.example/x", "scheme"),
        ("https://registry.example/x", "scheme"),
        ("file:///etc/passwd", "scheme"),
        ("gopher://x/y", "scheme"),
        # A schemeless host:port is refused too, just by the character set
        # rather than by the URL rule - the allowlist is what makes the
        # unanticipated shape fail closed.
        ("registry.example:8080//v2/x", "illegal_character"),
        ("some\\model", "backslash"),
        ("..", "path_traversal"),
        ("../../etc/passwd", "path_traversal"),
        ("library/../secret", "path_traversal"),
        ("qwen3..4b", "path_traversal"),
        ("qwen3:", "illegal_character"),
        ("qwen3:4b:extra", "illegal_character"),
        (":4b", "illegal_character"),
        ("a/b/c", "illegal_character"),
        ("/qwen3", "illegal_character"),
        ("qwen3/", "illegal_character"),
        ("-qwen3", "illegal_character"),
        (".qwen3", "illegal_character"),
        ("qwen 3", "illegal_character"),
        ("qwen3@sha256", "illegal_character"),
        ("qwen3?x=1", "illegal_character"),
        ("qwen3#frag", "illegal_character"),
        ("qwen3%2e%2e", "illegal_character"),
        ("hf.co/user", "illegal_character"),
        ("hf.co/a/b/c", "illegal_character"),
        ("hf.co/", "illegal_character"),
    ],
)
def test_unsafe_or_malformed_names_are_refused(raw: str, code: str) -> None:
    rejection = rejected(raw)
    assert rejection.code == code
    # Every refusal has to be actionable: a bare "invalid" sends people to edit
    # config by hand rather than fix what they typed.
    assert rejection.message.endswith((".", "'"))
    assert len(rejection.message) > 15


def test_a_url_is_blamed_on_being_a_url_not_on_its_slashes() -> None:
    # Ordering matters for the message a person reads: "that is a URL, and the
    # registry is not yours to choose" is the useful sentence, where "'/' is
    # not allowed there" would send them to delete a slash and try again.
    assert rejected("https://evil.example/v2/x/manifests/y").code == "scheme"


def test_require_raises_the_typed_422_error() -> None:
    with pytest.raises(UnprocessableEntityError) as excinfo:
        require_model_name("../../etc/passwd")
    assert excinfo.value.status == 422
    assert "'..'" in excinfo.value.detail


def test_require_returns_the_parsed_name_when_accepted() -> None:
    assert require_model_name(" qwen3:4b ").text == "qwen3:4b"
