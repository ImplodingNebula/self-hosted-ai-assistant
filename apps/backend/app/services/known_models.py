"""A code-shipped ledger of per-model QUIRKS the runtime cannot report.

A quirk is a fact about specific weights that nothing about a deployment can
answer and no serving stack will volunteer: which of a family's tags always
reasons, which packaging channel strips the tool parser, what instruct prefix
an asymmetric embedder was trained against. It GATES NOTHING - it informs
priors and carries data for named consumers. A stale seed means an uninformed
model, never a broken one.

Entries record upstream-bug knowledge and measured facts, versioned with code;
every entry carries dated provenance. That is the whole contract: a reader who
disagrees with a line can go find the measurement it names, and a line nobody
can source does not belong here.

Only ``think_mode`` has a consumer today (:mod:`app.services.model_facts`).
The sibling tag, the embedder's query prefix and relevance floors, and the
advisory note are FORWARD DATA for the think-check, the embedder-check and the
management view, which read them in the slices that land next. They are
written now rather than later so each entry arrives with its provenance in one
review, instead of a fact and its source being added months apart.

Two things the seed is deliberately NOT. It is not a registry - it declares no
policy, no ceiling and no role, and a tag appearing here says nothing about
whether this deployment may run it. And it is not a registry mirror - it
describes model FAMILIES the ecosystem publishes, so it stays correct across
deployments that hold entirely different inventories.

One caveat the lookup cannot express, because it takes a tag and no provider:
some of these facts belong to a (model, RUNTIME) pair rather than to the
weights. A template with no think plumbing is a statement about how ONE serving
stack packaged a family, and the hf.co note is Ollama-packaging-specific
outright - under a runtime that loads from HuggingFace natively the trap does
not exist. Every such entry names its runtime in its provenance, and the whole
file is scoped to the one provider this deployment can select today. Adding a
second provider means splitting those entries by runtime before this module is
consulted for it, not widening the globs.

Consumers read it BELOW every declaration and ABOVE every guess: a policy
declaration outranks it (a human wrote that about this deployment), and it
outranks a template inference or a blank "unknown" (a measured family fact
beats reading a chat template's shape). The lookup is
pure and needs no I/O, so a caller with nothing but a tag string can consult
it - which is what makes the background jobs and the warm path smarter for
free.
"""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatchcase

from app.providers.base import tag_aliases
from app.services.model_resolution import ThinkMode

# The metacharacters that make a pattern a GLOB rather than an exact tag. Read
# off the pattern itself rather than declared per entry, so a family line and a
# specific-tag line cannot disagree about which kind they are.
_GLOB_CHARS = "*?["


@dataclass(frozen=True, kw_only=True)
class KnownModelQuirks:
    """What is known about one model tag, or one family of them.

    Attributes:
        pattern: An exact tag (``qwen3:4b``) or an fnmatch-style family glob
            (``qwen3.5:*``). Exact patterns are matched alias-aware; globs are
            matched against the raw tag string.
        think_mode: How these weights treat the think flag, when it is known
            from MEASUREMENT rather than from a template's shape. Measurement
            is the bar, not a preference: this outranks the runtime's own
            template report, so a guess here is a guess that silences a fact.
        thinking_model: The sibling tag that serves this one's thinking turns,
            for a pair-routed family where the two are different checkpoints.
        query_prefix: The instruct wrapper an asymmetric embedder's model card
            prescribes for retrieval QUERIES (documents embed raw).
        min_score: The absolute relevance floor calibrated for this embedder's
            similarity scale.
        rel_margin: The relative margin calibrated alongside ``min_score``.
            The two travel together: a floor calibrated on one prefix scale
            says nothing on another.
        note: Advisory text about this tag worth showing a human, when the
            quirk is something to warn about rather than something to apply.
        provenance: Where the entry's facts come from, dated. Required.
    """

    pattern: str
    provenance: str
    think_mode: ThinkMode | None = None
    thinking_model: str | None = None
    query_prefix: str | None = None
    min_score: float | None = None
    rel_margin: float | None = None
    note: str | None = None


# The instruct wrapper the qwen3-embedding card prescribes. Named once so
# nothing here and nothing added later can drift into two spellings of one
# prefix - a vector space is only comparable to itself, and a one-character
# difference silently re-scales every stored similarity.
_QWEN3_EMBED_PREFIX = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
)

# Declaration order matters only among GLOBS (an exact pattern always wins);
# the family lines therefore sit beneath the specific tags they would
# otherwise swallow.
KNOWN_MODELS: tuple[KnownModelQuirks, ...] = (
    KnownModelQuirks(
        pattern="qwen3:4b",
        think_mode="always",
        provenance="2507 thinking-only refresh; verified locally 2026-07-03 "
        "(think:false is a no-op)",
    ),
    KnownModelQuirks(
        pattern="qwen3:4b-instruct",
        think_mode="never",
        thinking_model="qwen3:4b",
        provenance="2507 instruct variant; Jinja template has zero IsThinkSet "
        "wiring (measured 2026-07-20)",
    ),
    KnownModelQuirks(
        pattern="qwen3:*",
        think_mode="hybrid",
        # Scoped to what was actually measured. Upstream remapped the
        # plain 30b-a3b and 235b tags to 2507 variants, whose toggle behaviour
        # nobody here has verified, so this line must not be read as covering
        # them - a family glob is the weakest kind of claim here and the
        # provenance has to say how weak.
        provenance="original qwen3 hybrids; 8b verified locally 2026-07-03 "
        "alongside the 4b pair, which overrides this line by exact match. The "
        "tags remapped to 2507 variants (30b-a3b, 235b) are UNVERIFIED here",
    ),
    KnownModelQuirks(
        pattern="qwen3.5:*",
        think_mode="hybrid",
        provenance="measured on qwen3.5:9b 2026-08-04: think toggles both ways",
    ),
    KnownModelQuirks(
        pattern="qwen3-vl:2b",
        think_mode="always",
        provenance="measured 2026-07: thinking model whose content channel stays empty",
    ),
    KnownModelQuirks(
        pattern="llama3.1:*",
        think_mode="never",
        provenance="no think plumbing in Ollama's llama3.1 templates; "
        "pod inventory verified 2026-08-03",
    ),
    KnownModelQuirks(
        pattern="llama3.2:*",
        think_mode="never",
        provenance="no think plumbing in Ollama's llama3.2 templates; "
        "pod inventory verified 2026-08-03",
    ),
    # gemma3 and deepseek-r1 lines were considered here and DELIBERATELY LEFT
    # OUT. Both would have been family guesses nobody in this repo measured,
    # and an unmeasured ``think_mode`` is not the harmless kind of stale: it
    # outranks the runtime's own template report, so a wrong "never" on a
    # family whose template DOES carry think plumbing omits the flag, and an
    # omitted flag is what a serving stack promotes to think:true on any model
    # advertising the capability - the dangerous triple, arrived at by a line
    # in this file. A tag with no entry falls through to the template
    # inference, which is the weaker answer and the honest one.
    #
    # The 0.6b line below repeats this prefix on purpose: an exact pattern
    # shadows the family glob outright (one entry answers, never a merge of
    # two), so a specific tag has to carry every axis it needs.
    KnownModelQuirks(
        pattern="qwen3-embedding:*",
        query_prefix=_QWEN3_EMBED_PREFIX,
        provenance="card-default instruct prefix; wraps all retrieval queries "
        "(grounding redesign 2026-07-09)",
    ),
    KnownModelQuirks(
        pattern="qwen3-embedding:0.6b",
        query_prefix=_QWEN3_EMBED_PREFIX,
        min_score=0.35,
        rel_margin=0.10,
        provenance="floors measured live 2026-07-09 on the prefixed scale; "
        "related 0.40-0.60, unrelated peaks ~0.31 (prefix repeated from the "
        "family line, which an exact match shadows)",
    ),
    KnownModelQuirks(
        pattern="hf.co/*",
        note="hf.co manifests for Ollama carry no renderer or parser: tool calls "
        "come out as prose with an empty tool_calls list and no error at all. "
        "Prefer the Ollama library tag for the same weights.",
        provenance="hf-vs-library A/B, 2026-07",
    ),
)


def quirks_for(model: str) -> KnownModelQuirks | None:
    """Return what is known about ``model``, or None when nothing is.

    Exact patterns are tried first and ALWAYS beat a glob, whatever the
    declaration order: a family line describes what its members usually do,
    and a specific tag is here precisely because it does something else. The
    2507 split is the case that makes this a rule rather than a preference -
    ``qwen3:4b`` and ``qwen3:4b-instruct`` sit inside a hybrid family and
    neither of them is hybrid.

    Exact matching is ALIAS-AWARE (:func:`app.providers.base.tag_aliases`),
    because both sides are written-down configuration: a deployment may spell
    a tag ``qwen3:4b`` or ``library/qwen3:4b`` and mean one model. Globs are
    matched against the raw tag string instead, so a family pattern says
    exactly what it looks like it says.

    Args:
        model: The exact model tag, as configured or as the runtime lists it.

    Returns:
        The first matching entry, or None for a tag the seed does not describe.
        At most ONE entry answers - the axes it leaves unset are simply
        unknown, never filled in from a broader line.
    """
    aliases = tag_aliases(model)
    if not aliases:
        return None
    for entry in KNOWN_MODELS:
        if not _is_glob(entry.pattern) and entry.pattern in aliases:
            return entry
    for entry in KNOWN_MODELS:
        if _is_glob(entry.pattern) and fnmatchcase(model, entry.pattern):
            return entry
    return None


def _is_glob(pattern: str) -> bool:
    """True when ``pattern`` is a family glob rather than one exact tag."""
    return any(char in pattern for char in _GLOB_CHARS)
