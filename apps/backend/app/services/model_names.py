"""Model-name validation: the one place a typed-in name becomes safe to use.

Every management verb in this phase starts from free text somebody typed into a
box, and that text goes on to become a URL path segment on a request to a remote
registry and an argument to the local model server. Both of those are places
where a name carrying a scheme, a traversal, or a stray backslash stops being a
name and starts being an instruction - so the check happens ONCE, here, before
anything touches the network, and everything downstream consumes the parsed
result rather than the raw string.

The rule is an allowlist, not a blocklist. A conservative character set for
namespaces, repositories and tags accepts every name the model library actually
uses and rejects the rest by default, so a shape nobody anticipated fails closed
instead of being waved through because no specific rule happened to name it.

:func:`parse_model_name` is pure - no IO, no settings, no clock - which is what
makes it exhaustively testable. :func:`require_model_name` is the thin wrapper
that turns a rejection into the typed error a router renders.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from app.core.errors import UnprocessableEntityError

# Longest name accepted. Real tags run to a few dozen characters; the cap is
# generous enough never to bite one and small enough that nothing pathological
# reaches a URL builder or a log line.
MAX_NAME_LENGTH = 200

# Where a name points. "library" is the model registry's own namespace scheme
# (an optional publisher, a repository, an optional tag); "huggingface" is the
# hf.co passthrough the local runtime also accepts.
ModelNameKind = Literal["library", "huggingface"]

RejectionCode = Literal[
    "empty",
    "too_long",
    "control_character",
    "scheme",
    "backslash",
    "path_traversal",
    "illegal_character",
]

# The default namespace the registry serves un-namespaced names from.
LIBRARY_NAMESPACE = "library"
# The tag the registry resolves a name with no explicit tag to.
DEFAULT_TAG = "latest"

# Host prefixes that mean "this is a Hugging Face repository, not a library
# name". Matched case-insensitively because a browser copy-paste often is not.
_HUGGINGFACE_PREFIXES = ("hf.co/", "huggingface.co/")

# Anything that looks like a URL. Checked before the character set so the error
# names the real problem ("that is a URL") rather than blaming a slash.
_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.\-]*://")

# A namespace or repository: starts alphanumeric, then alphanumerics and the
# three separators the registry allows.
_SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
# A tag additionally allows a leading underscore, which some quantization
# suffixes use.
_TAG_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._-]*$")

# One sentence for every charset rejection: the reader needs the rule, not which
# of the three segments happened to break it first.
_CHARSET_MESSAGE = (
    "A model name may use letters, digits, '.', '_' and '-', with '/' between a "
    "publisher and a name and ':' before a version."
)

# What a Hugging Face name cannot be told about itself. Carried onto the
# preflight answer verbatim rather than being turned into a refusal: the local
# runtime does accept these names, and pretending otherwise would be a lie in
# the opposite direction from a fabricated size.
HUGGINGFACE_WARNING = (
    "This is a Hugging Face repository rather than a model-library tag, so it "
    "cannot be checked against the registry beforehand: whether it exists, how "
    "large the download is, and whether it was packaged with a working chat "
    "template are all unknown until it is here. Names packaged this way often "
    "arrive without the template that makes tool calling work, and nothing "
    "declares what such a model may do, so it runs at the safe floor."
)


@dataclass(frozen=True)
class ModelName:
    """A name that passed validation, split into the parts callers need.

    Attributes:
        text: The normalized name - what is sent to the model server and used
            as the identity of an in-flight download. Whitespace-trimmed and
            otherwise exactly as typed, so a round trip through this module
            never silently renames anything.
        kind: Which naming scheme the name follows.
        namespace: The publisher segment, defaulted to the registry's own
            library namespace for a bare name.
        repo: The repository segment.
        tag: The version segment, defaulted to the registry's ``latest``.
        warning: A caveat the caller must pass on to the user, or None.
    """

    text: str
    kind: ModelNameKind
    namespace: str
    repo: str
    tag: str
    warning: str | None = None


@dataclass(frozen=True)
class ModelNameRejection:
    """Why a name was refused, in terms the person who typed it can act on.

    Attributes:
        code: Stable machine-readable reason.
        message: A sentence naming what is wrong and what would be accepted.
    """

    code: RejectionCode
    message: str


def parse_model_name(raw: str) -> ModelName | ModelNameRejection:
    """Validate and classify a requested model name. Pure - no IO.

    Args:
        raw: The name exactly as it was received.

    Returns:
        A :class:`ModelName` when the name is acceptable, otherwise a
        :class:`ModelNameRejection` explaining which rule refused it.
    """
    text = (raw or "").strip()
    if not text:
        return ModelNameRejection("empty", "Enter a model name.")
    if len(text) > MAX_NAME_LENGTH:
        return ModelNameRejection(
            "too_long",
            f"A model name may be at most {MAX_NAME_LENGTH} characters; this one is {len(text)}.",
        )
    if any(ord(char) < 0x20 or ord(char) == 0x7F for char in text):
        return ModelNameRejection(
            "control_character",
            "A model name cannot contain control characters. Paste the name as plain text.",
        )
    if _SCHEME_RE.match(text) or "://" in text:
        return ModelNameRejection(
            "scheme",
            "Enter a model name, not a URL - for example 'qwen3:4b'. The registry a "
            "name is fetched from is fixed by this deployment's configuration.",
        )
    if "\\" in text:
        return ModelNameRejection(
            "backslash", "A model name cannot contain a backslash; segments are separated by '/'."
        )
    if ".." in text:
        return ModelNameRejection("path_traversal", "A model name cannot contain '..'.")
    for prefix in _HUGGINGFACE_PREFIXES:
        # Compare a slice of the ORIGINAL, lowercased, rather than lowercasing
        # the whole string first: a few code points change length under
        # ``lower()``, which would make the prefix length wrong for the slice.
        if text[: len(prefix)].lower() == prefix:
            return _parse_huggingface(text, prefix_length=len(prefix))
    return _parse_library(text)


def require_model_name(raw: str) -> ModelName:
    """Parse a name, raising the typed 422 error when it is rejected.

    Args:
        raw: The name exactly as it was received.

    Returns:
        The parsed :class:`ModelName`.

    Raises:
        UnprocessableEntityError: The name failed validation; the rejection's
            message is the response detail.
    """
    parsed = parse_model_name(raw)
    if isinstance(parsed, ModelNameRejection):
        raise UnprocessableEntityError(parsed.message)
    return parsed


def _split_tag(text: str) -> tuple[str, str] | None:
    """Split ``body:tag``, or return None when the tag part is malformed."""
    body, separator, tag = text.partition(":")
    if not separator:
        return body, DEFAULT_TAG
    if not tag or ":" in tag:
        return None
    return body, tag


def _parse_library(text: str) -> ModelName | ModelNameRejection:
    """Parse ``[namespace/]repo[:tag]`` against the registry's naming rules."""
    split = _split_tag(text)
    if split is None:
        return ModelNameRejection(
            "illegal_character",
            "A model name carries at most one ':' and must name a version after it, "
            "for example 'qwen3:4b'.",
        )
    body, tag = split
    segments = body.split("/")
    if len(segments) == 1:
        namespace, repo = LIBRARY_NAMESPACE, segments[0]
    elif len(segments) == 2:
        namespace, repo = segments
    else:
        return ModelNameRejection(
            "illegal_character",
            "A model name is either 'name' or 'publisher/name', with an optional "
            "':version' - it cannot have more parts than that.",
        )
    if not _SEGMENT_RE.match(namespace) or not _SEGMENT_RE.match(repo):
        return ModelNameRejection("illegal_character", _CHARSET_MESSAGE)
    if not _TAG_RE.match(tag):
        return ModelNameRejection("illegal_character", _CHARSET_MESSAGE)
    return ModelName(
        text=text, kind="library", namespace=namespace, repo=repo, tag=tag, warning=None
    )


def _parse_huggingface(text: str, *, prefix_length: int) -> ModelName | ModelNameRejection:
    """Parse ``hf.co/user/repo[:quant]``, which always carries the warning."""
    split = _split_tag(text[prefix_length:])
    if split is None:
        return ModelNameRejection(
            "illegal_character",
            "A Hugging Face name carries at most one ':' and must name a quantization "
            "after it, for example 'hf.co/user/repo:Q4_K_M'.",
        )
    body, tag = split
    segments = body.split("/")
    if len(segments) != 2:
        return ModelNameRejection(
            "illegal_character",
            "A Hugging Face name is 'hf.co/user/repository', with an optional ':quantization'.",
        )
    user, repo = segments
    if not _SEGMENT_RE.match(user) or not _SEGMENT_RE.match(repo) or not _TAG_RE.match(tag):
        return ModelNameRejection("illegal_character", _CHARSET_MESSAGE)
    return ModelName(
        text=text,
        kind="huggingface",
        namespace=user,
        repo=repo,
        tag=tag,
        warning=HUGGINGFACE_WARNING,
    )
