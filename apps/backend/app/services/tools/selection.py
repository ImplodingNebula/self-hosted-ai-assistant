"""Threshold-gated, in-process top-K tool selection.

Every offered tool rides in the prompt as its full JSON Schema, so the offered
set is paid for in context on EVERY turn. While the catalog is the two
in-process tools that cost is trivial; the moment MCP servers join the registry
the tool count can outgrow both a small model's window and its judgement (a 4B
model handed thirty schemas picks worse than one handed five). So selection
becomes retrieval: rank the tools against the turn's query and offer the best
few.

It is retrieval done IN PROCESS, over cached description vectors, reusing only
the :class:`~app.providers.rag.base.Embedder` seam. It deliberately does NOT use
the vector store. The per-turn cost is the QUERY embed, which a store would not
remove - it would only ADD a third ``CollectionKind`` across every seam method,
a fake ``tenant_id`` for data that is not tenant-scoped, and vectors that strand
on an embedder change. Ranking tens of tools in process is microseconds.

The threshold is what keeps this honest. Below it, selection is completely inert
- zero embed calls, zero IO, the tool list returned as it came in. That is not
an optimization, it is the point: a plain agentic turn pays NO embedding today
(the chat service short-circuits retrieval), and the dev embedder is CPU-pinned
where a cold load alone is 6-8s. Selection must never put an embed on a turn
that today pays none. Dev has 4 tools against a threshold of 8, so selection
never runs there at all; it wakes up only once a deployment has enough tools for
the ranking to be worth more than the query embed it costs.

Failure here is never fatal. A selection that cannot rank falls back to the
registry's own order: a less-relevant tool set is strictly better than no
answer. "Cannot rank" includes "did not rank in time": the candidate set comes
from REMOTE listings, so both its size and the embedder's speed are bounded here
rather than trusted.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
from collections.abc import Sequence

from app.providers.rag.base import Embedder
from app.providers.tools.base import Tool
from app.services.tools.local import SEARCH_KNOWLEDGE_TOOL, WEB_SEARCH_TOOL

logger = logging.getLogger(__name__)

# The tools that are always offered, whatever they rank. They are the generic
# reach for anything outside the model's own knowledge, so a turn that ranks
# them last is usually a turn whose query is a poor description match - not a
# turn that needs them less.
CORE_TOOL_NAMES: frozenset[str] = frozenset({WEB_SEARCH_TOOL, SEARCH_KNOWLEDGE_TOOL})

# Defensive cap on the description-vector cache (see ToolVectorCache).
_MAX_CACHE_ENTRIES = 512

# How many non-core tools may be RANKED (and so embedded) in one turn. The
# candidate set is whatever the registry's remote sources listed, so without
# this the first turn after a listing cache expires embeds the lot in one batch,
# on the turn, against the same CPU-pinned embedder the chat model contends
# with. Past the cap the tail simply keeps registry order and is never a
# candidate - it could not have been OFFERED anyway (only top_k of them are), so
# the cost is relevance among the tail, not capability.
_MAX_RANKED_TOOLS = 128

# Deadline for the whole ranking round trip (the query embed plus whatever
# descriptions are not cached). Generous enough for a COLD CPU-pinned embedder
# (a load alone is 6-8s) plus a batch, tight enough that a wedged embedder costs
# a few seconds of a turn rather than the turn. Past it, selection degrades
# exactly as an embedder failure does - registry order.
_EMBED_TIMEOUT_SECONDS = 15.0


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Return the cosine similarity of two vectors, 0.0 if either has no length.

    Pure stdlib on purpose: ranking tens of short vectors does not justify a
    numpy dependency. A zero-norm vector has no direction, so it has no
    similarity to anything - 0.0 is the honest answer and it keeps a degenerate
    embedding (or a vector the cache could not produce) from dividing by zero.

    Args:
        left: One vector.
        right: The other vector; a length mismatch is truncated to the shorter.

    Returns:
        The cosine similarity, or 0.0 when either vector is empty or zero.
    """
    dot = 0.0
    left_sq = 0.0
    right_sq = 0.0
    for a, b in zip(left, right, strict=False):
        dot += a * b
        left_sq += a * a
        right_sq += b * b
    if left_sq <= 0.0 or right_sq <= 0.0:
        return 0.0
    return dot / math.sqrt(left_sq * right_sq)


def _embed_text(tool: Tool) -> str:
    """Return the text that represents a tool in the vector space.

    The description is what the model reads to choose a tool, so it is what the
    ranking matches against. An empty description would embed the empty string
    and rank arbitrarily forever, so it falls back to the tool NAME - a weak
    signal, but a real one ("create_issue" is not nothing).
    """
    spec = tool.spec
    return spec.description.strip() or spec.name


def _digest(text: str) -> str:
    """Fingerprint the embedded text for the cache key."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ToolVectorCache:
    """Caches tool DESCRIPTION vectors across turns.

    Descriptions are stable across turns while queries are not, so the document
    side of this ranking is computed once and reused: the steady-state per-turn
    cost is the single query embed and nothing else.
    """

    def __init__(self) -> None:
        """Start empty; the cache fills lazily on the first selection turn."""
        # Insertion-ordered by dict contract - see the FIFO trim in _store.
        self._vectors: dict[tuple[str, str, str], list[float]] = {}

    async def vectors_for(
        self, tools: Sequence[Tool], embedder: Embedder
    ) -> dict[str, list[float]]:
        """Return a description vector per tool name, embedding only what is missing.

        Args:
            tools: The tools to vectorize; names are unique per the registry.
            embedder: The embedder defining the vector space. Descriptions are
                the DOCUMENT side of this ranking, so they embed raw via
                ``embed_texts`` - only the query takes the embedder's instruct
                prefix (``embed_query``).

        Returns:
            A vector per tool name. A name is absent only when the embedder
            returned fewer vectors than texts (a broken embedder); the caller
            scores an absent tool 0.0 rather than failing the turn.

        Raises:
            Exception: Whatever the embedder raises - the caller catches it and
                falls back, so a selection failure never reaches the user.
        """
        keys: dict[str, tuple[str, str, str]] = {}
        missing: dict[tuple[str, str, str], str] = {}
        for tool in tools:
            text = _embed_text(tool)
            # Three-part key, each part earning its place:
            #   embedder.model - a different model is a different vector SPACE;
            #     a cached vector from the old embedder is not merely stale, it
            #     is meaningless against the new query vector.
            #   tool name - the identity the model calls by.
            #   description hash - a server can change a tool's description
            #     under a stable name (the listing is remote, untrusted and
            #     mutable), and a stale vector would silently misrank it
            #     forever, with nothing in the output to hint at why.
            key = (embedder.model, tool.spec.name, _digest(text))
            keys[tool.spec.name] = key
            if key not in self._vectors:
                missing[key] = text
        if missing:
            # ONE batched call, not one per tool: the embedder round-trip
            # dominates, and the cold CPU-pinned embedder on dev makes a
            # per-tool fan-out pathological.
            missing_keys = list(missing)
            vectors = await embedder.embed_texts([missing[key] for key in missing_keys])
            for key, vector in zip(missing_keys, vectors, strict=False):
                self._store(key, vector)
        return {name: self._vectors[key] for name, key in keys.items() if key in self._vectors}

    def _store(self, key: tuple[str, str, str], vector: list[float]) -> None:
        """Insert a vector, trimming the oldest entries past the cap.

        Bounded rather than unbounded: the working set really is tens of entries
        keyed by content hash, but a misbehaving or hostile MCP server can rotate
        its descriptions on every listing, and each rotation mints a new key. An
        unbounded cache would grow for the life of the process on remote input
        alone. FIFO, not LRU: under description churn the oldest entries ARE the
        rotated-away ones, and the honest steady-state working set never reaches
        the cap, so the extra bookkeeping would buy nothing.
        """
        while len(self._vectors) >= _MAX_CACHE_ENTRIES:
            self._vectors.pop(next(iter(self._vectors)))
        self._vectors[key] = vector


async def select_tools(
    tools: list[Tool],
    *,
    query: str,
    embedder: Embedder | None,
    threshold: int,
    top_k: int,
    cache: ToolVectorCache,
) -> list[Tool]:
    """Narrow the offered tool set to the core tools plus the best ``top_k`` others.

    Args:
        tools: The registry's catalog for this turn, in registry order.
        query: The turn's query, matched against tool descriptions.
        embedder: The embedder to rank with; None (not wired) ranks by registry
            order instead of failing.
        threshold: At or below this many tools, selection is INERT - the list is
            returned as-is with no embed and no IO.
        top_k: How many NON-CORE tools to offer. The core tools are offered on
            top of it, so the returned set holds up to ``len(core) + top_k``
            tools. Zero or less offers the core tools only.
        cache: The process-wide description-vector cache.

    Returns:
        The tools to offer, always a new list: the core tools in registry order,
        followed by the best-ranked non-core tools.
    """
    # The inert fast path, FIRST and before anything else touches the embedder
    # or the cache. This is the dev path (4 tools, threshold 8) and it must cost
    # literally nothing - see the module docstring.
    if len(tools) <= threshold:
        return list(tools)

    core = [tool for tool in tools if tool.spec.name in CORE_TOOL_NAMES]
    rest = [tool for tool in tools if tool.spec.name not in CORE_TOOL_NAMES]
    if top_k <= 0:
        return core
    if embedder is None:
        return core + rest[:top_k]
    # An empty query has no direction to rank against - embedding it would order
    # the tools arbitrarily but confidently. Registry order is the honest answer.
    if not query.strip():
        return core + rest[:top_k]

    # Ranking is paid for per candidate, and the candidates come from remote
    # listings - so the batch is bounded before it is embedded, not after.
    candidates = rest[:_MAX_RANKED_TOOLS]
    if len(rest) > _MAX_RANKED_TOOLS:
        logger.warning(
            "%d non-core tools offered; ranking only the first %d and leaving the "
            "rest in registry order",
            len(rest),
            _MAX_RANKED_TOOLS,
        )

    try:
        # The QUERY side takes the embedder's instruct prefix (embed_query); the
        # DOCUMENT side embeds raw (embed_texts, inside the cache). That
        # asymmetry is the embedder seam's contract, not an oversight here.
        # One deadline covers both: an embedder that hangs must cost a slower
        # turn, never a turn that never starts. TimeoutError is an Exception, so
        # it lands in the same degrade-to-registry-order handler below.
        async with asyncio.timeout(_EMBED_TIMEOUT_SECONDS):
            query_vector = await embedder.embed_query(query)
            description_vectors = await cache.vectors_for(candidates, embedder)
        # Scoring stays INSIDE the try: an embedder can return a structurally
        # valid list whose elements are not numbers (a proxy answering
        # {"embeddings": [null, null]} clears the provider's length check), and
        # that only turns into a TypeError here, at the arithmetic. Ranking
        # outside the guard would let it escape and kill the turn - which is
        # exactly what the degradation below promises it cannot do.
        scores = [
            cosine(query_vector, description_vectors.get(tool.spec.name, [])) for tool in candidates
        ]
        # Sort on (-score, index) so equal scores keep REGISTRY order. Python's
        # sort is already stable, but the tie-break is spelled out because the
        # property is load-bearing: a tool set that flapped between identical
        # turns would be miserable to debug and would break prompt-prefix caching.
        order = sorted(range(len(candidates)), key=lambda index: (-scores[index], index))
        ranked = [candidates[index] for index in order[:top_k]]
    except Exception as exc:
        # Any failure - the provider being down, the store being down, a
        # malformed vector - degrades to registry order. A selection failure
        # must NEVER fail the turn: the worst case is a less-relevant tool set,
        # which is strictly better than no answer.
        logger.warning(
            "Tool selection failed (%s); offering the first %d non-core tools in registry order",
            type(exc).__name__,
            top_k,
        )
        return core + rest[:top_k]
    return core + ranked
