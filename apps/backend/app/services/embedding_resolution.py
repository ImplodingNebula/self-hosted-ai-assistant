"""Which embedder this deployment actually queries with, and how it decided.

The embedding model DEFINES the vector space: collections are named per model,
so changing it points every query at a fresh, empty space and leaves the old
one full and unread. That makes "which embedder" a deployment fact worth
resolving in exactly one place, the way the chat default already is.

It used to be one place - a committed file naming one embedder per hardware
profile - and that is the thing being retired. Such a file ships identically to
every machine, so it could describe what a profile was BUILT with and never
what a particular box had moved on to. Every rung here is instead something the
deployment itself can answer:

1. **The workspace choice.** Read through the tenant-bound settings
   repository, exactly like the default chat model. This is the rung an admin
   WRITES, and the one a console switch sets.
2. **The observed corpus.** What wrote the stored passages, sampled through
   :func:`app.services.vector_spaces.observed_embed_model`. Nothing has to be
   configured for this to be true, which is precisely its value: an upgraded
   deployment that never touched a settings page keeps embedding with the
   model its own vectors were built by, instead of silently starting a second
   vector space beside them.
3. **The only installed embedder.** When the model server holds exactly ONE
   tag that can embed, that is not a choice anybody has to make - an inventory
   is observed reality rather than curation. It is what makes a fresh box work
   with no setup beyond pulling an embedder: pull it, it is unambiguously the
   one, it is active.
4. **Several installed, and nothing selects one.** REFUSED, deliberately.
   Picking between two embedders would be a ranking, and the wrong pick writes
   vectors into a space the other model can never read. The surfaces that need
   an embedder say so and name the workspace defaults as the place to choose.
5. **Nothing.** Retrieval is unconfigured, and the same surfaces say so.

Two properties this chain has to keep, because both were paid for elsewhere:

- **The evidence rung never fails a request.** A store that cannot be
  enumerated, has nothing stored, or holds points without provenance simply
  falls through with a debug line. It rides the stranded-vectors alarm's own
  cached enumeration, so consulting it costs no extra round trip on a warm
  process and at most one bounded one on a cold path.
- **Placement is not part of the chain at all.** Where embedding work RUNS is
  a fact about the machine - how much card there is, and what else wants it -
  so ``EMBEDDING_DEVICE`` decides it outright rather than arbitrating with
  anything.

``dim`` and ``query_prefix`` describe the MODEL rather than the deployment, so
they come from the model's own sources: the runtime's report for the width (the
store also learns it from the first vectors written), and the known-model seed
for the instruct wrapper an asymmetric embedder's card prescribes.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from app.core.config import Settings
from app.providers.base import ModelProvider
from app.providers.rag.base import VectorStore
from app.schemas.settings import WorkspaceDefaults
from app.services.embedder_check import resolve_query_prefix
from app.services.model_facts import runtime_capabilities_many
from app.services.model_resolution import EmbeddingConfig
from app.services.vector_spaces import observed_embed_model, refresh_vector_spaces

logger = logging.getLogger(__name__)

# Where the answer came from. Reported rather than inferred: "nothing can
# embed", "several could and nobody chose" and "the corpus decided" are
# different states with different fixes, and only the resolution knows which
# one it took.
EmbeddingSource = Literal["workspace", "observed", "installed", "ambiguous", "none"]

# What to tell someone whose deployment resolves no embedder at all. Named once
# so the endpoint's refusal, the worker's boot error and anything else that has
# to explain it cannot drift into three slightly different instructions.
UNCONFIGURED_DETAIL = (
    "No embedding model is configured for this workspace, so retrieval has no "
    "vector space to work in. Choose one in the workspace defaults, and make "
    "sure the model server holds it."
)

# The same failure with a different cause, and a different fix: the weights are
# here, the choice is not. Refusing beats guessing - the candidates write into
# vector spaces neither can read, so a wrong guess is a corpus quietly split in
# half.
AMBIGUOUS_DETAIL = (
    "This model server holds more than one model that can create embeddings, "
    "and nothing selects which one retrieval uses - so it stays off rather "
    "than guess, because that choice decides which stored passages a query can "
    "ever reach. Name one in the workspace defaults."
)


@dataclass(frozen=True)
class EmbeddingChoice:
    """The active embedding configuration, and which rung supplied it.

    Attributes:
        config: What to embed with, or None when nothing resolved. None is the
            honest "retrieval is unavailable" answer, never a guess.
        source: The rung that answered.
        candidates: The installed embedding-capable tags, when the resolution
            saw the inventory. Carried so a refusal can NAME what there was to
            choose between instead of leaving somebody to go and look.
    """

    config: EmbeddingConfig | None
    source: EmbeddingSource
    candidates: tuple[str, ...] = ()

    @property
    def model(self) -> str | None:
        """The active embedding tag, or None when nothing resolved."""
        return self.config.model if self.config is not None else None

    @property
    def detail(self) -> str | None:
        """One sentence explaining an unresolved embedder, or None when one resolved."""
        if self.config is not None:
            return None
        return AMBIGUOUS_DETAIL if self.source == "ambiguous" else UNCONFIGURED_DETAIL


def config_for(tag: str, settings: Settings, *, dim: int | None = None) -> EmbeddingConfig:
    """Assemble the embedding configuration for a tag, whoever chose it.

    Args:
        tag: The embedding model to describe.
        settings: The active settings, which decide placement.
        dim: The vector width, when a caller already read one off the runtime.
            Unknown is a legitimate answer: the store learns a collection's
            width from the first vectors written into it, so a dimension is a
            cross-check rather than something retrieval needs told.

    Returns:
        The :class:`~app.services.model_resolution.EmbeddingConfig` retrieval
        and ingestion both run through.
    """
    return EmbeddingConfig(
        model=tag,
        dim=dim,
        device=settings.embedding_device,
        query_prefix=resolve_query_prefix(tag).prefix,
    )


async def installed_embedders(provider: ModelProvider, provider_name: str) -> tuple[str, ...]:
    """Return the installed tags that can embed and cannot complete, sorted.

    The "and cannot complete" half is the precedence every other surface
    applies: a model that can ALSO hold a conversation is listed as a chat
    model, so counting it here would make one tag a candidate for two roles and
    turn a single-embedder box into an ambiguous one.

    Sorted rather than left in the server's order, because the answer decides
    which vector space a deployment writes into: a listing that came back in a
    different order must not silently move the corpus.

    Args:
        provider: The model provider to ask.
        provider_name: Its configured name, for the facts cache key.

    Returns:
        The candidate tags, empty when nothing installed can embed or when the
        server could not be listed - a fall-through, never a failure.
    """
    # Function-local, as workspace_defaults does for the same reason:
    # model_manager imports resolve_embedding from this module, so a top-level
    # import back at it would be a cycle. The call itself is the point - the
    # chat-wins-over-embedding precedence has ONE home, and a local copy of the
    # predicate here would silently keep an old rule the day derive_role grows
    # a new axis.
    from app.services.model_manager import derive_role

    try:
        models = await provider.list_models()
    except Exception:  # noqa: BLE001 - an unlistable server simply skips this rung
        logger.debug("Could not list the model server to find an embedder", exc_info=True)
        return ()
    names = [info.name for info in models]
    reports = await runtime_capabilities_many(provider, provider_name, names)
    return tuple(
        sorted(
            name
            for name, report in zip(names, reports, strict=True)
            if report is not None and derive_role(report) == "embedding"
        )
    )


async def resolve_embedding(
    *,
    settings: Settings,
    stored: WorkspaceDefaults | None = None,
    store: VectorStore | None = None,
    provider: ModelProvider | None = None,
    provider_name: str | None = None,
    candidates: Sequence[str] | None = None,
    refresh: bool = False,
) -> EmbeddingChoice:
    """Resolve the embedder this deployment queries and ingests with.

    Args:
        settings: The active settings (placement, and nothing else).
        stored: The tenant's workspace defaults, already read through the
            tenant-bound repository. None for the genuinely tenant-free
            callers (a boot check, a maintenance probe), which then resolve
            the deployment's answer rather than a workspace's.
        store: The vector-store seam, for the evidence rung. None skips that
            rung entirely, which is what a caller with no store to hand should
            do - it is a fall-through, not a failure.
        provider: The model seam, for the inventory rung. None skips it, and a
            caller with no provider then resolves as it did before that rung
            existed.
        provider_name: The provider's configured name, for the facts cache.
            Read only alongside ``provider``.
        candidates: The embedding-capable installed tags, when the caller has
            already probed the inventory (the management view has). Given, they
            are used verbatim and ``provider`` is not consulted - the point is
            that a page and a turn resolve against ONE reading of the server.
        refresh: Re-enumerate the store rather than reuse a cached reading.
            Only a surface an operator is looking at should pay this; a turn
            never does.

    Returns:
        The :class:`EmbeddingChoice`, whose ``config`` is None when no rung
        answered.
    """
    if store is not None and refresh:
        # Refreshed BEFORE the rungs rather than inside the one that reads the
        # inventory, because a stored choice short-circuits past that rung -
        # and the caller asking for a fresh reading has a SECOND reader of the
        # same inventory (the stranded-vectors verdict) that would then be
        # served a cached one up to fifteen minutes old. That is how a delete
        # guard comes to judge a corpus written since the page was drawn.
        await refresh_vector_spaces(store)
    if stored is not None and stored.embedding_model:
        return EmbeddingChoice(
            config=config_for(stored.embedding_model, settings), source="workspace"
        )
    if store is not None:
        observed = await observed_embed_model(store)
        if observed:
            return EmbeddingChoice(config=config_for(observed, settings), source="observed")
        logger.debug("No embedder could be observed from the stored vectors.")
    if candidates is None and provider is not None and provider_name is not None:
        candidates = await installed_embedders(provider, provider_name)
    if candidates is None:
        return EmbeddingChoice(config=None, source="none")
    found = tuple(candidates)
    if len(found) == 1:
        return EmbeddingChoice(
            config=config_for(found[0], settings), source="installed", candidates=found
        )
    if found:
        return EmbeddingChoice(config=None, source="ambiguous", candidates=found)
    return EmbeddingChoice(config=None, source="none")
