"""The Model Manager view: a request-time join, never a stored copy.

An INVENTORY, first and last: one row per tag the model server actually holds.
There is no second list of models a deployment might have downloaded but has
not - such a row has no digest, no measurement, no verb worth offering and
nothing a reader can act on, and a page full of them buries the ones somebody
came for.

Four sources answer four different questions about each held tag, and the view
is simply what they say together:

- the PROVIDER says what is installed right now, and which management verbs it
  supports at all;
- the RUNTIME REPORT says what the serving stack knows about those weights -
  family, parameter count, quantization, the native window, and the mechanical
  capabilities (can the template take a tool list, do the weights see);
- the PROBE LOG says what was MEASURED on them - and, read once more without
  the digest filter, what was measured on weights that are no longer here,
  which is the only way a reader learns that a re-pull left a model's verdicts
  behind;
- the STORED DECLARATIONS and the SETTINGS say what an operator decided and how
  far the deployment clamps everything down.

Where a declaration and the runtime both have an answer, the declaration's
wins: it is a human decision about this deployment, digest-bound so a re-pull
retires it. Where only the runtime has one the runtime answers, because the
alternative is not caution but a blank card over a working model.

Two more sources answer questions about the same models from beside them. The
VECTOR STORE says whether what is already stored is still reachable by the
embedding model in use, reported here because this is the page where an
embedder is looked at. The STORED SETTINGS - the workspace default and the
tenant's conversation pins - say what is still set to USE each model, which is
what decides whether a row may offer a Delete at all.

Nothing here is persisted and no new table exists. A stored mirror of a live
provider listing would only be something to drift from, and the join is cheap -
one provider call, one capability report per tag (concurrent, and cached for
minutes by ``app.services.model_facts``), a couple of probe queries per chat
model, and two small reads for what is in use.

Two things this module deliberately does NOT do:

- **It never predicts fit.** The only sizes it reports are the provider's
  on-disk blob size and whatever a probe OBSERVED. Estimating whether a model
  will fit a card has already been measured wrong by multiples on this
  project's own hardware, and a confident wrong answer is worse than an honest
  "unmeasured".
- **It never calls the registry.** Whether a tag that is not installed could
  be downloaded is a question for the pull path, not for rendering a list.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.core.tenancy import TenantContext
from app.providers.base import (
    ManagementCapabilities,
    ModelCapabilities,
    ModelInfo,
    ModelProvider,
    installed_tag_matches,
    tag_aliases,
)
from app.providers.rag.base import VectorStore
from app.repositories.conversations import ConversationRepository
from app.schemas.models import (
    AgencyReport,
    ManagerRow,
    MeasuredFacts,
    ModelManagerResponse,
    ProviderInfo,
    RoleGap,
    RoleGroup,
    ThinkModeNote,
    WeightsChangeNote,
)
from app.schemas.rag import StrandedVectorsReport
from app.services.agency import agency_rank, parse_agency_mode, resolve_agency
from app.services.earned import (
    EarnedExplanation,
    ProbeView,
    declared_thinking_with_tools,
    endpoint_certifies,
    explain_earned,
    measured_think_for,
)
from app.services.embedding_resolution import resolve_embedding
from app.services.model_facts import (
    effective_agency_ceiling,
    effective_num_ctx,
    effective_think_mode,
    effective_tools_mode,
    effective_vision,
    runtime_capabilities_many,
)
from app.services.model_policy import policy_for
from app.services.model_resolution import (
    AgencyMode,
    ChatModelPair,
    ThinkMode,
    ToolsMode,
    resolve_chat_models,
    thinking_sibling_for,
)
from app.services.tool_check import CASES, clamping_cases, harness_version
from app.services.vector_spaces import vector_space_report
from app.services.workspace_defaults import load_defaults, reranker_availability

# How much a UI may do to each role, and why when the answer is "nothing".
#
# The embedding lock is the load-bearing one: the embedding model DEFINES the
# vector space, so swapping it leaves every stored vector unreachable by the
# new one's queries. The lock is the PREVENTIVE half of that protection; the
# detective half is the ``vectors`` report on the response, which catches a
# swap made outside this page (a changed workspace default, a delete). It is a
# lock on CHANGE, not on download: fetching an embedder strands nothing, and a
# fresh deployment has to be able to do it from here.
_EMBEDDING_LOCK = (
    "The embedding model defines the vector space stored passages live in: "
    "deleting or updating a model that vectors still reference would leave "
    "them unreachable, so those verbs are withheld here. Downloading, and "
    "deleting an embedder nothing references, stay available."
)
# The reranker is not served through the model seam at all - it runs in the
# Infinity sidecar - so there is no provider to ask whether it is installed,
# and no probe that measures it. A live health line needs a seam of its own
# (a sidecar health check); until then this row reports CONFIGURATION only.
_RERANKER_STATUS = (
    "The reranker runs in a separate sidecar rather than through the model "
    "provider, so it is reported here as configured status only."
)

_MANAGEMENT_BY_ROLE: dict[str, tuple[str, str | None]] = {
    "chat": ("manageable", None),
    "embedding": ("locked", _EMBEDDING_LOCK),
    "reranker": ("status_only", _RERANKER_STATUS),
}
# Groups render in this order; any other role follows, alphabetically. The
# empty role sorts last, which is where a tag the runtime describes as neither
# a chat model nor an embedder belongs: it is listed because the weights are
# here and occupy disk, not because anything is known about what it does.
_ROLE_ORDER = ("chat", "embedding")

# How many rows read the probe log at once. Each opens its own session, so the
# bound is about not asking the connection pool for one per tag when a server
# holds dozens - not about the cost of the reads themselves. The semaphore
# itself is built per request: an asyncio primitive binds to the loop it is
# first awaited on, and a module-level one would tie a page load to whichever
# loop happened to render the first.
_MAX_CONCURRENT_ROW_READS = 8

logger = logging.getLogger(__name__)


def article_for(word: str) -> str:
    """Pick "a" or "an" for a word interpolated into a refusal.

    Public and shared by every refusal that names a role, because these
    sentences are read by people: "a embedding model" reads as a bug in the
    product rather than as a message about their request.

    Args:
        word: The word the article will precede.

    Returns:
        ``"an"`` before a vowel, ``"a"`` otherwise.
    """
    return "an" if word[:1].lower() in "aeiou" else "a"


def derive_role(report: ModelCapabilities | None) -> str:
    """Name what a tag can serve, from the runtime's report and nothing else.

    ONE derivation, used by every surface that has to group, gate or refuse on
    a role: the management view's groups, the model listing the composer reads,
    the delete guard, the workspace-default writer, and the embedder rung
    (:func:`app.services.embedding_resolution.installed_embedders`). A second
    copy is how a page comes to offer a verb the API refuses. The one deliberate
    exception is :func:`app.services.model_verify._wrong_role_reason`, which must
    treat a MISSING report as "allowed" rather than as a role - absence of
    evidence is not grounds to refuse a check.

    The precedence is not symmetric, and deliberately: a model that can ALSO
    complete is a CHAT model everywhere, even when it can embed too. Counting
    such a tag as an embedder would make one model a candidate for two roles,
    put a chat model in the embedding group's lock, and turn a
    single-embedder server into an ambiguous one.

    A report that could not be taken answers ``""`` - "the runtime would not
    say" - which is a different claim from a report of no capabilities, and the
    callers read it that way: an empty role is manageable and offers no verbs
    it cannot justify, rather than being refused.

    Args:
        report: The runtime's report for this tag, or None.

    Returns:
        ``"chat"``, ``"embedding"``, or ``""``.
    """
    if report is None:
        return ""
    if report.completion:
        return "chat"
    return "embedding" if report.embedding else ""


def management_of(role: str) -> tuple[str, str | None]:
    """Return a role's management mode and lock reason.

    An empty role is manageable: nothing establishes what such a tag does, and
    refusing to manage it would be a guess in the more surprising direction
    than simply offering the provider's own verbs.

    Public because the delete path enforces exactly what this view renders. Two
    copies of "which roles are locked" is one copy too many: the day they drift,
    the UI hides a button the API still honours.

    Args:
        role: The role derived for a tag, or "" for none.

    Returns:
        The management mode and, when it is not ``manageable``, the reason.
    """
    return _MANAGEMENT_BY_ROLE.get(role, ("manageable", None))


def serves_chat_role(report: ModelCapabilities | None) -> bool:
    """Say whether a tag can actually answer a chat turn.

    Public because two surfaces must agree about it and one of them is a
    REFUSAL: the console offers thinking partners from this rule and the pair
    writer rejects on it, so a second definition would let the API accept a
    partner the UI would never have offered.

    The case this exists for is narrow and nasty: a vision-only tag can be
    measured ``always`` on the think flag and so passes every reasoning test,
    while its visible ``content`` channel comes back empty. Asking "can it
    reason" and never "can it reply" is what let such a tag be offered as a
    thinking partner.

    Args:
        report: The runtime's report for the tag, or None.

    Returns:
        True when a chat turn could route to it.
    """
    return derive_role(report) == "chat"


async def _installed_reports(
    provider: ModelProvider, provider_name: str, models: list[ModelInfo]
) -> dict[str, ModelCapabilities]:
    """Probe every installed tag once, concurrently, and keep what answered.

    A tag that does not answer is simply absent from the map, and every caller
    reads that as "the runtime would not say" - never as a report of no
    capabilities. The two look identical in a boolean and are opposite
    diagnoses: one is a server to fix, the other is a model to replace.

    The fan-out is bounded by the facts service, which owns that number for
    every surface that describes a whole inventory.
    """
    reports = await runtime_capabilities_many(
        provider, provider_name, [info.name for info in models]
    )
    return {info.name: caps for info, caps in zip(models, reports, strict=True) if caps is not None}


def _checkable(role: str) -> bool:
    """Decide whether a tool check may be run against this row right now.

    Mirrors what :func:`app.services.model_verify.begin_verification` actually
    refuses, so a control the page enables is one the endpoint honours: every
    row here holds installed weights, and a role the runtime reports as
    something other than chat rules the check out - measuring an embedder's
    tool-calling would hold the card for minutes to learn nothing.

    A tag whose probe FAILED has the empty role and is not offered a check
    either, and that is the one place this is stricter than the endpoint, which
    accepts an unreadable report. Enabling a control on a server that will not
    describe its own models would spend GPU minutes on a guess; the endpoint
    stays permissive so a deliberate request still goes through.
    """
    return role == "chat"


def _row_ceiling(*, role: str, ceiling: int | None) -> int | None:
    """The workspace context ceiling, for rows a CHAT turn could actually run on.

    The ceiling is a chat-turn budget: it bounds what one question may spend of
    the machine. An embedder is driven by the ingest path, and it never reads
    the setting - so a card showing it clamped would report a limit that does
    not apply to it, which is the same lie in the opposite direction from a
    card that ignores a limit that does.

    An UNKNOWN role takes the ceiling, and that is not caution - it is
    agreement with the turn. The turn path applies the ceiling without
    consulting any report at all, and the ceiling now decides an unsized tag's
    window rather than merely clamping it, so withholding it here on a blipped
    probe would make the card name a window a turn on the very same tag would
    not request. "The runtime would not say" is also precisely when this row
    might be one a turn lands on.

    Args:
        role: The row's runtime-derived role, empty when the probe said
            nothing.
        ceiling: The tenant's stored ceiling, or None when none is set.

    Returns:
        The ceiling to size this row's window with, or None to leave it alone.
    """
    return None if role == "embedding" else ceiling


def _embedding_gap(
    *,
    installed: set[str],
    candidates: Sequence[str],
    active_tag: str | None,
) -> RoleGap | None:
    """Report the embedding role's gap, or None when there is none.

    The failure this exists for is silent: with no usable embedder, uploads
    still accept files, knowledge search still returns (nothing), and web
    results still arrive - they simply never ground anything. Nothing errors,
    so nothing says so, which is why the state is worth a card of its own.

    Every branch is FATAL, including the one whose fix is a choice rather than
    a download. Retrieval embeds through the SELECTED model and nothing else,
    so an unselected embedder sitting on disk does not make the feature work
    less well - it makes it not work. The codes carry the difference that a
    reader can act on (see :class:`app.schemas.models.RoleGap`), and severity
    stays honest about the consequence.

    A suggestion is always the tag this deployment ALREADY resolved for itself,
    never a recommendation. Choosing between installed candidates would be a
    ranking, and naming a model nobody here holds would be a curated list by
    another name; the remedy for both is the free-text download the page
    already offers, preflighted against the registry.

    Args:
        installed: The tags this server holds.
        candidates: The installed tags that could embed, derived by the
            caller from the ONE inventory reading this page took (see
            :func:`_roles`) under the same rule the resolution chain's own
            inventory rung applies - so the banner, the cards and the embedder
            a retrieval runs on cannot describe different servers.
        active_tag: The embedder this workspace actually queries with, from the
            resolution chain. Passed in because the chain needs a workspace and
            a vector store, and this function is pure.
    """
    if active_tag and installed_tag_matches(active_tag, installed):
        return None

    dead = (
        "so uploaded files never become searchable, knowledge search finds nothing, "
        "and web results cannot ground an answer"
    )
    if active_tag:
        alternatives = (
            ""
            if not candidates
            else f" Installed models that could embed: {', '.join(candidates)} - but nothing "
            "selects one, because the embedding model defines the vector space and is "
            "therefore configuration rather than a choice made per request."
        )
        return RoleGap(
            role="embedding",
            severity="fatal",
            code="embedder_missing",
            message=(
                f"This server does not hold '{active_tag}', the embedding model this "
                f"deployment is configured to use, {dead}. Downloading it is the fix that "
                f"needs no other change.{alternatives}"
            ),
            suggested_model=active_tag,
        )
    if not candidates:
        return RoleGap(
            role="embedding",
            severity="fatal",
            code="embedder_missing",
            message=(
                f"No model on this server can create embeddings, {dead}. Download one - "
                "the box below takes any tag the registry serves - and it becomes the "
                "active embedder as soon as it is the only one here."
            ),
        )
    count = "one model" if len(candidates) == 1 else f"{len(candidates)} models"
    return RoleGap(
        role="embedding",
        severity="fatal",
        code="embedder_unselected",
        message=(
            f"No embedding model is selected, though this server holds {count} that "
            f"could embed ({', '.join(candidates)}), {dead}. With several to choose "
            "between, the choice is not one this deployment may guess at - it decides "
            "which stored passages a query can reach - so name one in the workspace "
            "defaults to turn retrieval on."
        ),
    )


def _tools_gap(rows: list[ManagerRow]) -> RoleGap | None:
    """Report why nothing is cleared to use tools, or None when something is.

    Two states that look the same from the outside and are fixed in opposite
    ways, which is the whole reason this is one function with two codes. A
    server whose models cannot take a tool list needs a DIFFERENT MODEL; a
    server whose capable models are all held at ``deterministic`` needs a
    DECLARATION, a passing check, or a change to the deployment's own clamp,
    and downloading anything at all would be wasted effort.

    Silent when nothing installed can hold a conversation: the deployment
    cannot answer at all in that state, and a note about tools underneath that
    would be noise on top of a much larger problem.
    """
    chat_capable = [row for row in rows if row.checkable]
    if not chat_capable:
        return None
    capable = [row for row in chat_capable if row.tools == "native"]
    if not capable:
        return RoleGap(
            role="chat",
            severity="note",
            code="tools_incapable",
            message=(
                "No installed model's chat template can take a tool list, so nothing here "
                "can search the web or the knowledge base on its own initiative. Answers "
                "are still grounded by the deterministic pre-fetch; what is missing is the "
                "model choosing when to look something up."
            ),
        )
    if all(row.agency is not None and row.agency.effective == "deterministic" for row in capable):
        return RoleGap(
            role="chat",
            severity="note",
            code="tools_clamped",
            message=(
                "Every installed model whose template could take tools is held at "
                f"deterministic by policy ({len(capable)} of them), so no turn is handed a "
                "tool. Each card says which axis clamps it - a declaration or a passing "
                "check lifts one, where downloading more of the same would not."
            ),
        )


def _measured_facts(
    explanation: EarnedExplanation, *, runtime: str | None, harness: str, endpoint: str | None
) -> MeasuredFacts | None:
    """Project the probe behind a verdict into the reported measurements.

    DEEP is preferred over FAST: it exercises every FAST case plus the guards,
    so where both exist it is the fuller measurement. None when no probe is
    bound to the installed weights - the honest "unmeasured", never an estimate.
    """
    view: ProbeView | None = explanation.deep or explanation.fast
    if view is None:
        return None
    return MeasuredFacts(
        depth=view.depth,
        tokens_per_second=view.tokens_per_second,
        vram_bytes=view.vram_bytes,
        measured_at=view.at,
        runtime_version=view.runtime_version,
        # Stale means CHANGED, never merely unknown: a probe that recorded no
        # version, or a runtime that exposes none, tells us nothing about
        # whether the serving stack moved - and a banner that cries staleness
        # at every such row would train a reader to ignore the real ones.
        runtime_stale=bool(view.runtime_version and runtime and view.runtime_version != runtime),
        harness_stale=view.harness_version != harness,
        # The host, and this one deliberately does NOT follow the "changed,
        # never merely unknown" rule above it. That rule fits a BANNER fact: an
        # unrecorded runtime version clamps nothing, so crying staleness over
        # it would be noise. The endpoint is a CERTIFICATE CONDITION - a row
        # that names no host is discounted exactly as one naming another host
        # is (:func:`~app.services.earned.endpoint_certifies`) - so a reader is
        # owed the same explanation in both cases, or the page shows a passing
        # deep probe beside "no check has proven this" and reads as a bug.
        #
        # The unnamed case is the COMMON one, not the rare one: every row
        # written before the column has it. Nothing else marks those rows
        # either - the harness fingerprint hashes the case specs and the served
        # prompt stack, and neither moved when this column arrived.
        #
        # Asked through the resolution's own helper rather than re-derived, so
        # the card cannot come to a different answer than the turn did.
        endpoint_stale=not endpoint_certifies(view.serving_endpoint, endpoint),
        # Carried so a reader can explain the numbers above rather than having
        # to re-derive it: a fully-spilled probe records zero VRAM and no
        # throughput, and those two alone look like a bug.
        representative=view.representative,
        gpu_fraction=view.gpu_fraction,
        # Read off the EXPLANATION rather than off the projected view, because
        # it is not a property of one row: it is what the binding rules make of
        # that row against the pair's installed thinking weights, and only the
        # resolver knows those. Taking `view.thinking_with_tools` instead would
        # report a raw column - true on a run measured against weights this
        # machine no longer holds, and meaningless on a FAST row, which never
        # runs the reasoning case at all.
        thinking_with_tools_passed=explanation.thinking_with_tools_measured,
    )


def _weights_change(
    explanation: EarnedExplanation | None, *, installed_digest: str | None
) -> WeightsChangeNote | None:
    """Report the verdicts these weights outran, or None when none were.

    The comparison the probe log cannot make for itself. Every digest-bound
    read drops a verdict measured on other weights, which is what keeps a
    verdict attached to WEIGHTS rather than to a mutable tag - but it also
    means a re-pull quietly empties a model's measured facts, an old failure
    clamp included. Nothing said so until this note.

    Purely a READING. It is derived per request from rows the reader already
    holds, it feeds nothing back into :mod:`app.services.earned`, and a turn
    resolves exactly as it did before it existed. It also needs no storage and
    no dismissal: the state it describes ends when a check runs against the
    installed weights, and the note ends with it.

    Two silences are deliberate. Weights the runtime cannot digest-identify
    have nothing to compare, so they raise no notice (the agency report already
    names that case under the hardened regime). And a row measured with NO
    digest recorded says "unknown weights", not "different weights" - warning
    about it would put a reader on the trail of a re-pull that never happened.

    Two suppressions, both of them "that depth is already covered":

    - A depth that STILL HAS a verdict bound to the installed weights has lost
      nothing, however new the row above it is. That happens whenever a tag
      comes back to weights a check already covered, and it is the case that
      would otherwise contradict the row itself - the reported measurement and
      the warning would name the same depth and disagree about it. Read off
      the very pair the measured facts come from, so the two cannot diverge.
    - An orphaned FAST that a bound DEEP at least as new already re-measured.
      DEEP runs every FAST case plus the guards, which is the same superseding
      the earned resolution applies.

    Args:
        explanation: The row's earned explanation, or None for a row that has
            none (any non-chat role).
        installed_digest: The weights the tag points at right now, or None when
            the runtime exposes none.

    Returns:
        The note, or None when nothing was left behind.
    """
    if explanation is None or installed_digest is None:
        return None
    bound = {"fast": explanation.fast, "deep": explanation.deep}
    orphaned = [
        view
        for view in explanation.latest_any_weights
        if view.digest is not None
        and view.digest != installed_digest
        and bound.get(view.depth) is None
    ]
    bound_deep = explanation.deep
    if bound_deep is not None:
        orphaned = [
            view for view in orphaned if not (view.depth == "fast" and bound_deep.at >= view.at)
        ]
    if not orphaned:
        return None
    # Last, not first: `latest_any_weights` runs weakest depth first, so this
    # is the fullest orphaned measurement - the same DEEP-over-FAST preference
    # the measured facts beside it use.
    leading = orphaned[-1]
    return WeightsChangeNote(
        depths=[view.depth for view in orphaned],
        earned=leading.earned,
        measured_at=leading.at,
    )


async def _measured_think(
    *,
    provider_name: str,
    model: str,
    digests: dict[str, str],
    sessionmaker: async_sessionmaker[AsyncSession],
) -> ThinkMode | None:
    """Read the think mode the probe log measured on the installed weights.

    Read through the SAME function the chat turn uses
    (:func:`app.services.earned.measured_think_for`) rather than off the earned
    explanation this row already holds, and the reason is the one rule this
    module keeps: a card must never describe a model differently from the turn
    that runs it. The explanation is the wrong source for this axis on two
    counts, both of which would show up as a silently missing note:

    - It is keyed on the governing entry's CHAT tag, because a pair shares its
      agency verdicts. It emphatically does not share a think mode - one
      sibling always reasons and the other never does.
    - It takes an early-out for a tag whose ceiling is already
      ``deterministic`` with no thinking-with-tools claim, which never reads
      the log at all. That is right for an agency verdict a probe could not
      change, and wrong here: the routing clamp applies to such a model just
      the same, so the card would show a declaration the turn is overriding.

    It costs the short-TTL cached read the turn pays, shared with it, and
    returns None the moment the row has no installed digest to bind to.

    Args:
        provider_name: The ``MODEL_PROVIDER`` selector value rows key on.
        model: The tag this row is about.
        digests: The installed name -> digest map the view already read.
        sessionmaker: Session factory for the probe-log read (cached).

    Returns:
        The measured mode, or None when nothing measured these weights.
    """
    found = await measured_think_for(
        provider_name=provider_name, model=model, digests=digests, sessionmaker=sessionmaker
    )
    return None if found is None else found.mode


def _think_note(
    *,
    measured: ThinkMode | None,
    declared: ThinkMode,
    report: ModelCapabilities | None,
    checkable: bool,
    thinking_with_tools: bool,
) -> ThinkModeNote | None:
    """Report a disagreement about how this model treats the think flag.

    Three sources, and the note names which pair of them fell out:

    - A MEASUREMENT that contradicts the declaration is the re-pull detector.
      A tag is mutable, so a check earned on one checkpoint and a declaration
      written for another can both be about "the same model" while describing
      different weights - and the failure is silent in both directions. It
      reads as reasoning markup in the answer, or as reasoning that vanished,
      neither of which raises anything.
    - The DANGEROUS TRIPLE, raised only when nothing has measured these weights
      at all: the declaration says the model does not reason, so a turn omits
      the flag entirely, while the runtime advertises think plumbing AND the
      model's own template carries it. A serving stack that reads an absent
      flag as "on" then reasons with nothing parsing the channel. It is a
      hazard rather than an observation, so a measurement - of ANY mode -
      retires it: either the measurement agrees and the hazard is not real, or
      it disagrees and the mismatch above is the better sentence.

    The suppression order matters and is not arbitrary: a measurement is
    evidence and a triple is a suspicion, so evidence is reported and the
    suspicion is only raised where there is none.

    Args:
        measured: What a check measured on the installed weights, or None.
        declared: The resolution WITHOUT the measurement - what the note is
            disagreeing with.
        report: The runtime's report for this tag, or None.
        checkable: Whether a chat turn could route to this row at all
            (:func:`_checkable`, the same answer the verify endpoint gives).
            The think flag is only ever sent to a model a turn runs, so a row
            that cannot be one has nothing to disagree about - and an embedder
            distilled from a chat family ships that family's template, which
            would otherwise raise the hazard note on a row no flag ever
            reaches.
        thinking_with_tools: The EFFECTIVE reason-and-call-tools capability -
            the declaration already folded against what a deep check measured.
            Read for one sentence and one only: a forced think flag on a model
            without the capability also strips that turn's tools, and a note
            that reported half of what the clamp does would be the page hiding
            the half a reader would notice.

            The FOLD and not the declaration, because the sentence is about
            what a turn does and the turn keys on the fold
            (:mod:`app.services.turn_plan`). A granted-but-unproven model reads
            as capable in the declaration and is offered no tools by the chat
            path, so the claim would have written exactly the wrong sentence.

    Returns:
        The note, or None when the sources agree (the ordinary case).
    """
    if not checkable:
        return None
    advertises = report is not None and report.thinking
    if measured is not None and measured != declared:
        clamped = measured == "always"
        if clamped:
            detail = (
                "It reasons even when the flag is cleared, so turns now force it ON - "
                "otherwise the reasoning arrives as raw markup inside the answer. The "
                "usual cause is the tag being re-pulled onto a differently-behaving "
                "checkpoint."
            )
            if not thinking_with_tools:
                # Said plainly, because it is the consequence a reader will
                # actually observe: the model stops calling tools. Reasoning
                # and tool-calling in one turn fails as an EMPTY answer on some
                # model-and-runtime pairs, so the turn keeps the reasoning and
                # drops the tools - and the grounding a tool would have done is
                # covered meanwhile by the pre-fetch that runs when no tool
                # survives.
                detail += (
                    " Nothing declares that this model can reason and call tools in the "
                    "same turn, so while the flag is forced it is offered no tools at "
                    "all - answers stay grounded through the usual pre-fetch and the "
                    "knowledge base instead. Confirming the mode and earning that "
                    "grant is what gives the tools back."
                )
        elif measured == "never":
            detail = (
                "It produced no reasoning either way, so the flag is doing nothing "
                "here. Turns still route on the declaration until this is confirmed."
            )
        else:
            detail = (
                "It reasons only when asked, so the flag genuinely toggles. Turns "
                "still route on the declaration until this is confirmed."
            )
        return ThinkModeNote(
            code="measured_mismatch",
            measured=measured,
            declared=declared,
            runtime_thinking=advertises,
            message=(
                f"A tool check measured this model as '{measured}' on the installed "
                f"weights, where the declaration says '{declared}'. {detail}"
            ),
            confirmable=True,
            routing_clamped=clamped,
        )
    if measured is None and declared in ("never", "unknown") and advertises:
        if report is None or not report.template_thinking:
            # Without template plumbing there is nothing for a promoted flag to
            # drive, so the advertised capability is a fact about the serving
            # integration and not a hazard. Measured live on this project's own
            # instruct tag, which advertises thinking and cannot use it.
            return None
        return ThinkModeNote(
            code="nil_promotion",
            measured=None,
            declared=declared,
            runtime_thinking=True,
            message=(
                "Nothing declares that this model reasons, so a turn sends no think "
                "flag at all - but the runtime advertises think plumbing and the "
                "model's own template carries it, which is the combination a serving "
                "stack can read as 'reason by default'. Reasoning would then arrive "
                "with nothing parsing it out. Run a tool check: it measures which of "
                "the two is true here."
            ),
            confirmable=False,
            routing_clamped=False,
        )
    return None


# The harness's own case order, so a tie in pass fraction is broken by the
# order the check RUNS them in rather than by whatever order the JSON column
# happened to serialize. Derived from the specs, never re-listed: a new case
# joins this map by existing. An import-time snapshot, like
# `tool_check.HARNESS_VERSION` and with the same caveat - a test that
# monkeypatches CASES must rebuild it rather than expect a live read.
_CASE_ORDER: dict[str, int] = {case.name: index for index, case in enumerate(CASES)}

# How many cases a clamp line names before it starts counting. Four fits the
# sentence; past that the line stops being a pointer and becomes the report.
_MAX_NAMED_CASES = 4


def _failed_case_detail(view: ProbeView | None) -> str | None:
    """Name the cases that held a clamping probe's verdict down, worst first.

    The half of the ``failed_case`` blame that makes it actionable: which
    cases, and how badly. Worst-first because a case that lost every sample is
    a different problem from one that lost a single sample to noise, and the
    reader wants the first one.

    Only cases that actually BLAME are named, and blame is asked of
    :func:`~app.services.tool_check.clamping_cases` rather than inferred from
    the numbers here. An imperfect tally is not the same thing as a failing
    one: the deciding cases are scored by majority at deep depth, so ``2/3``
    passes and holds nothing back, while a guard case blames on a single lost
    sample. Reading "imperfect" as "to blame" would print four cases that
    passed and push the one that caused the clamp into "+n more" - the same
    mistake as naming the wrong probe ROW, one level down.

    Args:
        view: The probe row the clamp rests on - the SAME row the earned
            resolution took its evidence from, never merely the newest one.
            None whenever that row is unavailable.

    Returns:
        The parenthetical's contents, or None when there is nothing honest to
        put there: a row predating tallies, or one whose recorded cases do not
        account for the clamp (a hand-written row, or one measured by a
        harness whose rules have since moved).
    """
    if view is None:
        return None
    # The `run > 0` filter is re-stated rather than inherited: the parser two
    # modules away guarantees it, but this function divides by it and takes a
    # value type any caller can construct.
    tallies = {name: (passed, run) for name, passed, run in view.imperfect_cases if run > 0}
    # Intersected with what the row actually HAS: a case the run never reached
    # blames too, and has no numbers to show for it.
    blamed = [name for name in clamping_cases(view.depth, tallies) if name in tallies]
    if not blamed:
        return None
    blamed.sort(
        key=lambda name: (tallies[name][0] / tallies[name][1], _CASE_ORDER.get(name, len(CASES)))
    )
    named = blamed[:_MAX_NAMED_CASES]
    parts = [f"{name} {tallies[name][0]}/{tallies[name][1]}" for name in named]
    if len(blamed) > len(named):
        parts.append(f"+{len(blamed) - len(named)} more")
    return ", ".join(parts)


def _clamp_blame(
    *,
    effective: AgencyMode,
    ceiling: AgencyMode,
    earned: AgencyMode,
    env_mode: AgencyMode | None,
    explanation: EarnedExplanation | None,
    tools_native: bool,
    declared_by_policy: bool = False,
) -> tuple[str | None, str | None]:
    """Name the axis holding a model back, most actionable first.

    Priority: a measured FAILURE, then a chat template that cannot take a tool
    list at all, then "no check has proven this much for these weights", then
    the human declaration, then the deployment clamp. The order is about what a
    reader can DO next - a failing probe names a specific case to look at,
    while ``AGENCY_MODE`` is a line in an env file.

    The template floor sits ABOVE "unproven" rather than below it, which is a
    change the re-anchored grace forced and a correction either way. Almost
    every un-probed row is now unproven, so leaving that first would have
    answered a toolless model with "run a deep check" - a check that can never
    lift it, because no verdict overrules a template that takes no tool list.
    A measured failure still outranks both: something ran, and it says which
    case to look at.

    "Unproven" reads as the blame far more often than it used to, and the
    wording has to survive that: the default regime now carries a reviewed
    declaration only as far as ``assisted``, so a model declared ``agentic``
    that nobody has deep-checked sits one rung below its ceiling with no
    failure behind it. That is not "a measurement said no" - it is "nothing has
    measured this yet", and the sentence says so in both regimes rather than
    naming a switch only one of them has set.
    """
    if effective == "agentic":
        return None, None
    reason = explanation.reason if explanation is not None else None
    if reason is not None and reason.code == "failed_case":
        # The clamping row itself, addressed by the depth the resolution
        # recorded - never "the newest probe" and never the DEEP-preferred
        # row the measured facts show. A model can carry a clean verdict at
        # one depth and the failure at another, and naming the wrong row's
        # cases would send a reader to a case that passed.
        clamped = {"fast": explanation.fast, "deep": explanation.deep}.get(reason.depth or "")
        detail = _failed_case_detail(clamped)
        return (
            "failed_case",
            f"A {reason.depth} tool-check on the installed weights did not pass"
            # Absent rather than empty when the row has no tallies to read:
            # an old row is not evidence that nothing failed, and "()" would
            # be a broken sentence either way.
            f"{'' if detail is None else f' ({detail})'}, "
            f"so the measured verdict clamps this model to {earned}.",
        )
    if not tools_native:
        return (
            "tools_none",
            "The model server reports that this model's chat template cannot take a "
            "tool list, so no turn can hand it tools however much a declaration "
            "allows.",
        )
    if reason is not None:
        if reason.code == "no_digest":
            return (
                "unverified_weights",
                "The runtime exposes no weights digest, so no tool-check verdict "
                f"can be bound to what is installed, and this model runs at {earned}.",
            )
        if reason.code == "endpoint_mismatch":
            # Split from "unverified" because it sends a reader somewhere else:
            # the check has been run and passed, it simply was not run HERE, so
            # the fix is a re-check on this server rather than a first one.
            return (
                "endpoint_mismatch",
                f"A {reason.depth} tool-check passed on these weights, but it was measured "
                "on a different or unrecorded server, so its verdict does not carry to this "
                f"one - this model runs at {earned} until a check runs against the model "
                "server being served from now.",
            )
        return (
            "unverified_weights",
            f"No tool-check certifies {ceiling} for the installed weights, "
            f"so this model runs at {earned} until a deep check passes.",
        )
    if agency_rank(ceiling) < agency_rank("agentic"):
        if declared_by_policy:
            # Ranked first: a stored declaration is what a reader would change
            # to lift this model, and telling them "nothing declares it" would
            # send them to make a decision somebody already made.
            return (
                "declared_ceiling",
                f"A stored declaration sets a ceiling of {ceiling} for this model.",
            )
        # Nothing declares it, so the runtime's own report stands in - and it
        # stops one rung short of the top by design. Worth spelling out,
        # because the row beside this says the template takes a tool list, and
        # a bare "undeclared" would read as a flat refusal rather than as the
        # one rung a report cannot climb.
        #
        # Phrased as a CEILING, never as what the model is running at: this
        # branch is reached before the env clamp is considered, so a
        # deployment-wide clamp may be holding the model lower still, and a
        # sentence asserting the running mode would contradict the `effective`
        # field beside it on the same card.
        return (
            "uncatalogued",
            "Nothing declares what this model may do, so the model server's own "
            f"report stands in and its ceiling is {ceiling} - tools may be offered "
            "alongside the usual grounding. Driving a turn on its own initiative "
            "needs a declaration.",
        )
    if env_mode is not None and agency_rank(env_mode) < agency_rank("agentic"):
        return ("env_clamp", f"AGENCY_MODE={env_mode} clamps every turn on this deployment.")
    return None, None


def _agency_report(
    *,
    ceiling: AgencyMode,
    explanation: EarnedExplanation | None,
    env_mode: AgencyMode | None,
    tools: ToolsMode,
    declared_by_policy: bool = False,
) -> AgencyReport:
    """Build one row's agency report from its own axes.

    ``effective`` comes from the SAME resolver a turn uses, over exactly the
    axes reported beside it - so a reader can re-derive the number, and the
    grid can never display an autonomy the resolver would not grant.

    That last guarantee is why the tool-support floor is applied HERE and not
    left to the turn: a turn passes ``runtime_floors=("deterministic",)`` for a
    model whose template takes no tool list
    (:func:`app.services.turn_plan.plan_turn`), and this floor is knowable from
    the same report the row already carries. The remaining per-turn axes (the
    request clamp, whether any tool survived the user's switches, whether the
    context window has room for a result) are NOT knowable here and are left
    out rather than guessed; each can only clamp down, so the reported mode
    stays a ceiling on the real one.
    """
    earned: AgencyMode = explanation.earned if explanation is not None else ceiling
    tools_native = tools == "native"
    effective = resolve_agency(
        ceiling=ceiling,
        earned=earned,
        env_mode=env_mode,
        runtime_floors=() if tools_native else ("deterministic",),
    )
    code, detail = _clamp_blame(
        effective=effective,
        ceiling=ceiling,
        earned=earned,
        env_mode=env_mode,
        explanation=explanation,
        tools_native=tools_native,
        declared_by_policy=declared_by_policy,
    )
    return AgencyReport(
        ceiling=ceiling,
        earned=earned,
        env_clamp=env_mode,
        effective=effective,
        clamp_code=code,  # type: ignore[arg-type]  # built from the same literal set
        clamp_detail=detail,
    )


def _row_verbs(
    *, mgmt: ManagementCapabilities, installed: bool, management: str
) -> tuple[bool, bool]:
    """Derive (can_pull, can_delete) for one row.

    Server-derived rather than left to the client: what may be done to a model
    depends on the provider's verbs, on whether the weights are already here,
    and on how much its role allows - three facts the server holds and a UI
    would have to re-implement (and eventually re-implement wrongly).

    A MANAGEABLE row keeps its pull even once installed: a pull on installed
    weights fetches whatever the registry now serves under the tag, which is
    how a model takes updates (and how a digest change re-clamps its verdict
    to unverified, exactly as the measurement log intends).

    The LOCKED role's pull is DOWNLOAD-ONLY. Fetching a missing embedder
    strands nothing - a fresh clone has to be able to do it from here - but an
    UPDATE to an installed one is the silent version of the swap this lock
    exists to prevent: changed weights under an unchanged tag shift the vector
    space while the provenance (which records the model NAME, not the digest)
    keeps the stranded-vectors alarm quiet. Its blanket "no delete" is only
    the STARTING answer: the build re-derives embedding rows' delete per row
    against the stranded-vectors report (see :func:`embedding_delete_block`),
    so an installed embedder nothing references gets the verb back. A
    ``status_only`` role offers nothing at all: it is not served through this
    provider, so there is no verb to honour.
    """
    if management == "status_only":
        return (False, False)
    if management == "locked":
        return (mgmt.can_pull and not installed, False)
    return (mgmt.can_pull, mgmt.can_delete and installed)


async def load_model_usage(
    *,
    settings: Settings,
    session: AsyncSession,
    tenant: TenantContext,
    provider_name: str,
    sessionmaker: async_sessionmaker[AsyncSession],
    digests: dict[str, str],
) -> ModelUsage:
    """Read every rung currently set to use a chat model.

    A few cheap queries, all on deliberate actions (rendering the management
    page, deleting a model) rather than on a chat turn.

    The stored PAIRING is resolved in here rather than handed in, and that is
    the whole point of the signature. It was a parameter first, and the delete
    endpoint promptly forgot to pass it - so the management page disabled Delete
    on a declared partner while the endpoint behind the button happily removed
    it. A guard that two callers have to remember to arm is a guard that is
    armed on one of them; doing the read here makes the omission
    unrepresentable.

    Args:
        settings: The active settings (the deployment's own chat tag).
        session: Database session for the stored rungs.
        tenant: The principal whose workspace and conversations are read.
        provider_name: The ``MODEL_PROVIDER`` selector value declarations key on.
        sessionmaker: Session factory for the declaration read (cached).
        digests: The installed name -> digest map the caller already read.

    Returns:
        The assembled :class:`ModelUsage`.
    """
    workspace = await load_defaults(session, tenant)

    async def pair_for(default_tag: str, workspace_model: str | None) -> tuple[str | None, bool]:
        """Resolve one routable tag's thinking sibling, and whether it was declared.

        Keyed on the tag whose pairing is being asked about, which is the whole
        point: a declaration is stored per model, so reading it under one tag
        and applying it to another's pair is how the guard came to protect the
        wrong sibling.
        """
        policy = await policy_for(
            provider_name=provider_name,
            model=default_tag,
            digests=digests,
            sessionmaker=sessionmaker,
        )
        declared = None if policy is None else policy.thinking_model
        pair = resolve_chat_models(
            settings, workspace_model=workspace_model, declared_thinking_model=declared
        )
        # Compared against the RESOLVED sibling, not just asserted: an
        # environment override outranks a declaration, so a stored pairing that
        # lost to one must not be named as the place to change what a turn
        # actually routes to.
        return pair.thinking, bool(declared) and pair.thinking == declared

    # Two tags a turn can route through, and BOTH of their pairs matter.
    #
    # ``active`` means "the pair this DEPLOYMENT is configured for" - the rung a
    # request cannot repair - which is what makes its Delete an outright
    # refusal. The workspace default is deliberately NOT folded into it: that
    # one IS repairable, by naming a successor, and collapsing it in here would
    # turn the whole hand-over flow into a flat "no".
    #
    # But its SIBLING is not repairable that way. A thinking turn on the
    # workspace default routes there, and deleting it breaks that turn with no
    # successor to name - so it is protected like any other active sibling.
    # Reading one pairing and applying it to the other pair left exactly that
    # tag exposed while blocking a tag nothing routed to.
    binding_default = resolve_chat_models(settings).default
    routed_default = resolve_chat_models(settings, workspace_model=workspace.model).default
    binding_thinking, binding_declared = await pair_for(binding_default, None)
    active = ChatModelPair(default=binding_default, thinking=binding_thinking)
    if routed_default == binding_default:
        # The ordinary case: no stored default, or one that resolves to the
        # same tag. One pair, one read - the second would be the same query.
        routed_thinking, routed_declared = binding_thinking, binding_declared
    else:
        routed_thinking, routed_declared = await pair_for(routed_default, workspace.model)
    siblings = {tag for tag in (binding_thinking, routed_thinking) if tag}
    declared_siblings = {
        tag
        for tag, declared in (
            (binding_thinking, binding_declared),
            (routed_thinking, routed_declared),
        )
        if tag and declared
    }
    return ModelUsage(
        active=active,
        thinking_siblings=frozenset(siblings),
        declared_siblings=frozenset(declared_siblings),
        workspace_model=workspace.model,
        context_ceiling=workspace.context_ceiling,
        routed_default=routed_default,
        pin_counts=await ConversationRepository(session, tenant).pinned_model_counts(),
    )


async def build_manager_view(
    *,
    provider: ModelProvider,
    provider_name: str,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
    tenant: TenantContext,
    store: VectorStore | None = None,
) -> ModelManagerResponse:
    """Join the inventory, the probe log, the declarations and the settings.

    Args:
        provider: The model seam, for the installed listing and its verbs.
        provider_name: The ``MODEL_PROVIDER`` selector value probe rows key on.
        settings: The active settings (the ``AGENCY_MODE`` clamp and regime).
        sessionmaker: Session factory for the probe-log and stored-rung reads.
        tenant: The principal whose workspace and conversations are consulted
            for what is still set to use each model.
        store: The vector-store seam, for the stranded-vectors report. Omitted
            leaves ``vectors`` null - "not asked", never "nothing wrong".

    Returns:
        The assembled :class:`ModelManagerResponse`.

    Raises:
        ProviderUnavailableError: When the provider cannot be listed.
            Deliberately NOT caught: this view IS the inventory, and an empty
            page because the server is momentarily down reads as a machine
            holding nothing rather than as a machine that did not answer.
    """
    models = await provider.list_models()
    mgmt = provider.management_capabilities()
    runtime = await provider.runtime_version()
    # ONE reading of where these models are served from, for the whole page:
    # every row's certificate is judged against it and every row's staleness
    # explained by it, so two rows can never disagree about which host answered.
    endpoint = provider.serving_endpoint()
    harness = harness_version()
    env_mode = parse_agency_mode(settings.agency_mode)
    # What the RUNTIME says about each held tag - the source of every capability
    # column, and of the ROLE each row is grouped under.
    reports = await _installed_reports(provider, provider_name, models)
    digests = {info.name: info.digest for info in models if info.digest}
    # One reading of the reranker gate for the whole page, from the same
    # function the retrieval dependency uses. The sidecar is not a model this
    # provider serves, so it has no row of its own here.
    reranker = reranker_availability(settings)
    # What is still SET to use each model: the active pair withdraws Delete
    # outright, while pin counts and the workspace-default flag ride on the
    # rows so the confirm dialog can disclose what a delete will change. The
    # stored pairing is resolved inside, so this view and the delete endpoint
    # cannot arm the guard differently.
    async with sessionmaker() as session:
        usage = await load_model_usage(
            settings=settings,
            session=session,
            tenant=tenant,
            provider_name=provider_name,
            sessionmaker=sessionmaker,
            digests=digests,
        )
        stored_defaults = await load_defaults(session, tenant)

    # The tags this server holds that could embed, derived through the SAME
    # function the resolution chain's inventory rung uses - so the banner below
    # and the embedder a retrieval actually runs on read one answer about one
    # snapshot, rather than probing the server twice and disagreeing.
    embedders = tuple(sorted(name for name, role in _roles(models, reports) if role == "embedding"))

    # The embedder this page is about, resolved through the chain a retrieval
    # runs through - the banner and the stranded-vectors verdict below both
    # hang on it, and a page that judged them against a different answer would
    # be describing a deployment that does not exist.
    active_embedding = await resolve_embedding(
        settings=settings,
        stored=stored_defaults,
        store=store,
        candidates=embedders,
        refresh=True,
    )

    # Computed BEFORE the rows, not only for the response's status banner: the
    # same report is the embedding role's DELETE POLICY (see
    # embedding_delete_block), so the verbs and the banner read one answer.
    # Recomputed on every page load rather than served from the cache - this
    # page is precisely where an operator who just changed the embedder looks,
    # and it is off the turn path, so a handful of bounded round trips buys an
    # answer that is true right now. Storing the fresh result also warms the
    # cache the chat path reads.
    #
    # ``refresh`` sits on the RESOLUTION above rather than here: both read the
    # same enumeration, and asking twice would mean judging a corpus against an
    # embedder resolved from a different reading of it.
    vectors = (
        await vector_space_report(store=store, refresh=False, active_model=active_embedding.model)
        if store is not None
        else None
    )

    reads = asyncio.Semaphore(_MAX_CONCURRENT_ROW_READS)

    async def row_for(info: ModelInfo) -> ManagerRow:
        async with reads:
            return await _row(
                info,
                report=reports.get(info.name),
                settings=settings,
                mgmt=mgmt,
                env_mode=env_mode,
                usage=usage,
                provider_name=provider_name,
                sessionmaker=sessionmaker,
                digests=digests,
                runtime_version=runtime,
                serving_endpoint=endpoint,
                harness=harness,
                vectors=vectors,
            )

    # Concurrent, bounded, and ordered by the gather rather than by completion.
    # Every row reads the probe log and the declaration store, and the
    # deployment this page exists for - a remote server holding a large
    # inventory - is precisely the one where serialising would chain one round
    # trip per installed model onto every page load.
    rows: list[ManagerRow] = list(await asyncio.gather(*(row_for(info) for info in models)))

    grouped: dict[str, list[ManagerRow]] = {}
    for row in rows:
        grouped.setdefault(row.role, []).append(row)
    groups = [
        RoleGroup(
            role=role,
            management=management_of(role)[0],  # type: ignore[arg-type]  # from the literal set
            lock_reason=management_of(role)[1],
            rows=grouped[role],
        )
        for role in sorted(grouped, key=_role_sort_key)
    ]
    # Assembled from the finished rows rather than from the raw listing: a gap
    # is a statement about what the page is showing, so deriving it from
    # anything else is how a banner ends up contradicting the cards under it.
    found = (
        _embedding_gap(
            installed={info.name for info in models},
            candidates=embedders,
            active_tag=active_embedding.model,
        ),
        _tools_gap(rows),
    )
    # Worst first, so a client that shows one shows the one that matters.
    gaps = sorted(
        (gap for gap in found if gap is not None), key=lambda gap: gap.severity != "fatal"
    )
    logger.debug("Reranker for the management view: %s", reranker.detail)
    return ModelManagerResponse(
        provider=ProviderInfo(
            name=provider_name,
            runtime_version=runtime,
            can_pull=mgmt.can_pull,
            can_delete=mgmt.can_delete,
            can_list_installed=mgmt.can_list_installed,
        ),
        groups=groups,
        vectors=vectors,
        role_gaps=gaps,
    )


def _roles(
    models: Sequence[ModelInfo], reports: Mapping[str, ModelCapabilities]
) -> list[tuple[str, str]]:
    """Pair every held tag with the role the runtime's report derives for it."""
    return [(info.name, derive_role(reports.get(info.name))) for info in models]


async def _row(
    info: ModelInfo,
    *,
    report: ModelCapabilities | None,
    settings: Settings,
    mgmt: ManagementCapabilities,
    env_mode: AgencyMode | None,
    usage: ModelUsage,
    provider_name: str,
    sessionmaker: async_sessionmaker[AsyncSession],
    digests: dict[str, str],
    runtime_version: str | None,
    serving_endpoint: str | None,
    harness: str,
    vectors: StrandedVectorsReport | None,
) -> ManagerRow:
    """Build the row for one held tag.

    Every capability column comes from the RUNTIME, because against a server
    holding weights nobody has described those are the only facts there are - a
    blanket "no tools, no vision" over an inventory is not caution, it is a
    wrong answer that reads as a shipped limitation. The agency ceiling reads
    the same report and stops one rung short: a reported tool-taking template
    lifts a tag to ``assisted`` (tools offered, the deterministic pre-fetch
    still running), while ``agentic`` - handing the turn to the model - stays a
    human declaration. See
    :func:`app.services.model_facts.effective_agency_ceiling`.

    This is a render-time answer, not the authority: the delete endpoint
    re-derives the same policy per request, because a page cannot stay current
    with a pin somebody sets in another tab.
    """
    model = info.name
    role = derive_role(report)
    management = management_of(role)[0]
    can_pull, can_delete = _row_verbs(mgmt=mgmt, installed=True, management=management)
    # Resolved once and used twice: the agency report applies the no-tools
    # runtime floor from it, and the row displays it. Two calls would be two
    # chances for them to disagree.
    tools = effective_tools_mode(report)
    # What an operator declared about this tag, read through the SAME resolvers
    # the turn uses so a card and a turn cannot disagree. Read for every role,
    # not just chat: a think-mode confirmation is worth showing on any row, and
    # the read is cached.
    policy = await policy_for(
        provider_name=provider_name, model=model, digests=digests, sessionmaker=sessionmaker
    )
    # Resolved once, before the explanation, and handed to it: the turn
    # resolves the ceiling the same way, so a row explained against a different
    # one would be a card the chat path disagrees with.
    ceiling = effective_agency_ceiling(report, policy=policy)
    # The one ranking, asked once per row: which tag would serve THIS model's
    # thinking turns if a conversation pinned it. The same call the verify job
    # drives and the earned explanation binds against, off the same
    # declarations - so what this row DISPLAYS as the partner is the tag a run
    # measured and a certificate names, rather than a third answer computed
    # here.
    sibling = thinking_sibling_for(
        settings, model, declared=None if policy is None else policy.thinking_model
    )
    explanation = await explain_earned(
        provider_name=provider_name,
        model=model,
        settings=settings,
        require_verification=settings.agency_require_verification,
        sessionmaker=sessionmaker,
        digests=digests,
        serving_endpoint=serving_endpoint,
        effective_ceiling=ceiling,
        policy=policy,
    )
    # Resolved once and used three times (the row's own field, the gate on the
    # think note, and whether the measured read is worth taking at all): two
    # calls would be two chances for a card to raise a note about a row it also
    # says nothing can check.
    checkable = _checkable(role)
    # What a check WATCHED these weights do with the think flag, and what the
    # resolution says without it. Resolved as a pair so the card can report a
    # disagreement instead of quietly showing whichever won - and only for a
    # row a turn could route to, since nothing measures the think flag of a
    # model no turn ever sends it to.
    measured_think = (
        await _measured_think(
            provider_name=provider_name,
            model=model,
            digests=digests,
            sessionmaker=sessionmaker,
        )
        if checkable
        else None
    )
    declared_think = effective_think_mode(model, report, policy=policy)
    # Resolved once and used twice - the row's own field and the think note's
    # tool sentence - because both are statements about what a TURN does, and
    # two derivations would be two chances to disagree.
    ttw_effective = explanation.thinking_with_tools
    agency = (
        _agency_report(
            ceiling=ceiling,
            explanation=explanation,
            env_mode=env_mode,
            tools=tools,
            declared_by_policy=policy is not None and policy.ceiling is not None,
        )
        if role == "chat"
        else None
    )
    can_delete, delete_blocked = _clamp_in_use(model=model, usage=usage, can_delete=can_delete)
    # The locked role's blanket "no delete" narrows to a per-row answer: an
    # embedder nothing references strands nothing, and the report knows exactly
    # which ones those are.
    if role == "embedding" and mgmt.can_delete:
        blocked = embedding_delete_block(model, vectors)
        can_delete, delete_blocked = (blocked is None, blocked)
    return ManagerRow(
        model=model,
        provider=provider_name,
        role=role,
        installed=True,
        installed_size_bytes=info.size,
        digest=info.digest,
        num_ctx=effective_num_ctx(
            report,
            default_window=settings.default_context_window,
            ceiling=_row_ceiling(role=role, ceiling=usage.context_ceiling),
        ),
        native_context_length=report.context_length if report is not None else None,
        quantization=report.quantization if report is not None else None,
        parameter_size=report.parameter_size if report is not None else None,
        family=report.family if report is not None else None,
        tools=tools,
        # The full resolution, measured layer included, so the card shows the
        # mode a TURN will actually use rather than the one the declaration
        # alone would have produced.
        think=effective_think_mode(model, report, policy=policy, measured=measured_think),
        thinking_with_tools=declared_thinking_with_tools(policy),
        # What the DECLARATION becomes once the fold has had it. Off the same
        # explanation the agency verdict comes from, so the two halves of one
        # card were resolved against one snapshot.
        thinking_with_tools_effective=ttw_effective,
        thinking_with_tools_declared=(
            policy is not None and policy.thinking_with_tools is not None
        ),
        # Through the SAME ranking the turn applies, never a second copy of it:
        # the environment override, then a stored pairing. Reading the
        # declaration alone made this card name a partner that a deployment
        # with THINKING_CHAT_MODEL set never routes to.
        thinking_model=sibling,
        # Declared AND still winning. A pairing the environment overrode is not
        # what a turn uses, so marking it "declared here" would send a reader to
        # change something and watch nothing happen.
        thinking_model_declared=(
            policy is not None and bool(policy.thinking_model) and sibling == policy.thinking_model
        ),
        vision=effective_vision(model, report),
        agency=agency,
        measured=_measured_facts(
            explanation, runtime=runtime_version, harness=harness, endpoint=serving_endpoint
        ),
        weights_change=_weights_change(explanation, installed_digest=info.digest or None),
        think_mode=_think_note(
            measured=measured_think,
            declared=declared_think,
            report=report,
            checkable=checkable,
            thinking_with_tools=ttw_effective,
        ),
        can_pull=can_pull,
        can_delete=can_delete,
        checkable=checkable,
        delete_blocked_reason=delete_blocked,
        pinned_conversations=pinned_count_for(model, usage),
        workspace_default=is_workspace_default(model, usage),
        is_default=is_resolved_default(model, usage),
        ceiling_declared=policy is not None and policy.ceiling is not None,
    )


def _role_sort_key(role: str) -> tuple[int, str]:
    """Order the known roles as declared, then anything else alphabetically."""
    return (_ROLE_ORDER.index(role), "") if role in _ROLE_ORDER else (len(_ROLE_ORDER), role)


@dataclass(frozen=True)
class ModelUsage:
    """Everything currently SET to use a model, across the resolution tiers.

    One value assembled from the three places a turn's chat model can come from
    (see ``resolve_chat_models``), so the view and the delete endpoint reason
    over exactly the same facts instead of each remembering their own subset.

    Attributes:
        active: The DEPLOYMENT's resolved chat pair - the environment override
            or nothing at all on a box that has configured none, plus that
            tag's own stored pairing. Deliberately excludes the workspace default, which is
            repairable by naming a successor and must not read as unrepairable.
        thinking_siblings: Every tag a thinking turn could route to, across
            both routable defaults (the deployment binding and the workspace
            default). A sibling is protected if EITHER pair still reaches it -
            the two are different pairs and only checking one left a live
            sibling deletable while blocking a tag nothing routed through.
        declared_siblings: Which of those came from a stored declaration rather
            than from the environment. Carried only so a refusal can
            name the right place to change it: a message sending somebody to
            edit a YAML file that no longer decides the answer is worse than
            one that says nothing.
        workspace_model: The tenant's stored default chat model, if any.
        context_ceiling: The tenant's stored context ceiling, if any. Not a
            "what uses this model" fact like its siblings, but it rides here
            for the reason they do: the view needs the workspace's stored
            settings and this is the value already carrying them, so a card's
            window can be the window a turn requests instead of the one the
            model would grant unclamped.
        routed_default: The tag a turn with no model of its own ACTUALLY runs -
            the stored workspace default where there is a usable one, else the
            deployment binding. Distinct from ``workspace_model``, which is the
            stored string and is empty on most deployments: a grid wants to
            mark the one model that answers by default, and on a fresh install
            that is the binding rather than nothing at all.
        pin_counts: Trimmed pinned tag to the number of the tenant's
            conversations pinning it.
    """

    active: ChatModelPair
    thinking_siblings: frozenset[str] = frozenset()
    declared_siblings: frozenset[str] = frozenset()
    workspace_model: str | None = None
    context_ceiling: int | None = None
    routed_default: str = ""
    pin_counts: Mapping[str, int] = field(default_factory=dict)


def delete_block_reason(model: str, usage: ModelUsage) -> str | None:
    """Explain why deleting ``model`` is refused outright, or None.

    The single source of this policy. The management view renders the answer as
    a row's ``delete_blocked_reason`` and the delete endpoint raises it as a
    409, so the button a user sees disabled and the message they get for forcing
    it past a stale page say the same thing in the same words.

    Exactly one reason refuses outright now: the deployment's own chat pair,
    which is a CONFIG LINE (the environment's chat-model override) that no
    request can rewrite - deleting its weights would break the next
    turn with nothing in the product able to fix it. The two STORED tiers that
    used to refuse here are handled instead of guarded: conversation pins are
    cleared by the delete itself (the threads follow the workspace default),
    and the workspace default is replaced by a successor the caller names in
    the same request (see :mod:`app.services.model_delete`). Blocking on those
    made every model that anyone had ever picked effectively undeletable - an
    admin cannot walk every user's threads, and the change a delete forces on
    them is cheap and disclosed.

    Args:
        model: The tag being asked about.
        usage: What is currently set to use it.

    Returns:
        A sentence naming what to change, or None when nothing forbids it.
    """
    aliases = tag_aliases(model)
    role = next(
        (found for alias in sorted(aliases) if (found := active_chat_role(alias, usage.active))),
        None,
    )
    # Every sibling a thinking turn could route to, across BOTH routable
    # defaults - not just the deployment binding's. A thinking turn on the
    # workspace default lands on its partner, and that tag has no successor
    # flow to fall back on, so it is protected exactly like the binding's.
    declared_partner = False
    if role is None:
        matched = next(
            (tag for tag in usage.thinking_siblings if installed_tag_matches(tag, aliases)),
            None,
        )
        if matched is not None:
            role, declared_partner = "thinking", matched in usage.declared_siblings
    elif role == "thinking":
        declared_partner = any(
            installed_tag_matches(tag, aliases) for tag in usage.declared_siblings
        )
    if role is not None:
        # Where to change it, and only the places that actually decide it. A
        # thinking half an operator paired from the console is not governed by
        # deployment configuration at all, so naming it there would send somebody
        # to edit a file that no longer answers the question.
        where = (
            "the reasoning partner declared for it in the model manager"
            if declared_partner
            else (
                "the workspace default in the console, or the "
                "chat-model override in the environment"
            )
        )
        return (
            f"'{model}' is the deployment's active {role} model, so deleting it would "
            f"break the next turn. Point the deployment at another model first - through "
            f"{where} - and then delete this one."
        )
    return None


def pinned_count_for(model: str, usage: ModelUsage) -> int:
    """How many of the tenant's conversations pin any spelling of ``model``.

    Membership mirrors :func:`delete_block_reason`'s old pin check: a stored
    pin is a trimmed literal, so it matches when it is one of the tag's
    spellings. Not a guard any more - the count is what the confirm dialog
    discloses and what the delete clears - but view and delete path still
    count through this one function so they can never disagree.
    """
    aliases = tag_aliases(model)
    return sum(count for tag, count in usage.pin_counts.items() if tag in aliases)


def is_workspace_default(model: str, usage: ModelUsage) -> bool:
    """True when ``model`` (under any spelling) is the STORED workspace default.

    The delete path's question: deleting this needs a successor named in the
    same request. Deliberately not the same as :func:`is_resolved_default` -
    a deployment with nothing stored has no such row at all.
    """
    return (usage.workspace_model or "").strip() in tag_aliases(model)


def is_resolved_default(model: str, usage: ModelUsage) -> bool:
    """True when ``model`` is the tag an unpinned turn ACTUALLY runs.

    The grid's question rather than the delete path's, and the difference
    matters on every fresh deployment: with nothing stored, no row is the
    stored default and yet one model plainly answers by default. This resolves
    through the same chain a turn does, so exactly one row wears the mark
    whether or not anybody has ever set one.

    Args:
        model: The tag being asked about.
        usage: What is currently set to use each model.

    Returns:
        True for the one tag a default turn routes to.
    """
    return bool(usage.routed_default) and installed_tag_matches(
        usage.routed_default, tag_aliases(model)
    )


def embedding_delete_block(model: str, report: StrandedVectorsReport | None) -> str | None:
    """Explain why deleting embedding-role weights is refused, or None.

    The single source of the embedding role's delete policy, mirroring
    :func:`delete_block_reason` for the chat tiers: the management view renders
    the answer as a row's ``delete_blocked_reason`` and the delete endpoint
    raises it as a 409, so the disabled button and the refusal say the same
    thing in the same words.

    Three refusals, most-actionable-first, and one deliberate asymmetry: with
    no report (no store wired, or the store not answering) the answer FAILS
    CLOSED. Every other unknown on this page renders as an honest "unknown";
    this one gates a destructive verb whose damage - stored passages nothing
    can search - is exactly what the missing report would have warned about.

    Args:
        model: The embedding-role tag being asked about.
        report: The stranded-vectors report, or None when no store is wired.

    Returns:
        A sentence naming what to change, or None when nothing depends on it.
    """
    aliases = tag_aliases(model)
    if report is None or not report.known:
        return (
            f"Whether stored passages still depend on '{model}' is unknown - the vector "
            "store did not answer - so deleting an embedding model stays off until it does."
        )
    active = (report.active_model or "").strip()
    if active and installed_tag_matches(active, aliases):
        return (
            f"'{model}' is the active embedding model: every new chunk and every query "
            "embeds through it, so deleting it would break ingestion and retrieval "
            "outright. Choose another embedder in the workspace defaults - with a "
            "re-index - and then delete this one."
        )
    points = sum(
        space.points
        for space in report.spaces
        if space.points > 0
        and any(installed_tag_matches(stored, aliases) for stored in space.embed_models)
    )
    if points:
        passages = "1 stored passage was" if points == 1 else f"{points} stored passages were"
        return (
            f"{passages} embedded by '{model}', and a query can only reach them through "
            "the same weights. Deleting it would strand them - re-index them under the "
            "active embedder first (session-scope passages also expire on their own)."
        )
    return None


def _clamp_in_use(*, model: str, usage: ModelUsage, can_delete: bool) -> tuple[bool, str | None]:
    """Withdraw the delete verb from the deployment's own active pair.

    Applied to EVERY row. On most deployments the served model is one a
    request cannot repair, so this is the modal row rather than
    a corner case, and a view that offered Delete there would render an enabled
    button whose every press returns 409. The stored tiers no longer withdraw
    the verb at all: pins and the workspace default are handled BY the delete
    (cleared and replaced respectively), so their rows keep an enabled Delete
    and disclose the consequences in the confirm dialog instead.

    The reason rides beside the verb rather than inside the agency report: "why
    can I not delete this" and "how much may this model drive a turn" are
    different questions with different audiences, and a row whose delete is
    withdrawn may have no agency report at all (an embedding row) - which is
    exactly where a note folded into the report would vanish.

    Args:
        model: The row's tag.
        usage: What is currently set to use it.
        can_delete: The verb derived from the provider and the role.

    Returns:
        The possibly-withdrawn verb, and the reason when it was withdrawn.
    """
    if not can_delete:
        return can_delete, None
    reason = delete_block_reason(model, usage)
    return (can_delete and reason is None), reason


def active_chat_role(model: str, active: ChatModelPair) -> str | None:
    """Name the active chat role this tag fills, or None.

    ``DEFAULT_CHAT_MODEL`` / ``THINKING_CHAT_MODEL`` are honoured VERBATIM by
    the chat path - a deployment may point them at any tag - so the model a
    turn is about to request can legitimately show up here with nothing else
    describing it.

    Public because the delete path refuses exactly what this view refuses to
    offer: the tag the next turn is about to request must not be deletable, and
    the client's ``can_delete`` flag is a render-time snapshot that a second tab
    or an env change can invalidate before the button is pressed.

    Matching is SYMMETRIC (via :func:`tag_aliases`), because here both sides are
    written-down configuration: a deployment may configure ``qwen3:latest``
    while a caller names ``qwen3``, and expanding only one side would let one of
    the two spellings walk past the guard.

    Scope, stated exactly: this answers for the DEPLOYMENT-WIDE pair alone - the
    environment overrides alone. It knows nothing about
    the two STORED tiers above them in ``resolve_chat_models`` (a workspace
    default, a conversation's pinned model), which live in the database rather
    than in configuration. Those are the delete path's job - it clears the pins
    and hands the default to a successor (see
    :mod:`app.services.model_delete`) - not this function's.

    Args:
        model: The tag being asked about.
        active: The deployment's resolved chat pair.

    Returns:
        "chat", "thinking", or None.
    """
    aliases = tag_aliases(model)
    if active.default and installed_tag_matches(active.default, aliases):
        return "chat"
    if active.thinking and installed_tag_matches(active.thinking, aliases):
        return "thinking"
    return None
