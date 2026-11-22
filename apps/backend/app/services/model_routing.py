"""The last step between a resolved chat model and the model that runs.

:mod:`app.services.model_resolution` answers "which tag does the configuration
point at"; this module answers "is that tag actually on this machine". They are
separate questions because configuration is written down and the installed set
is observed, and only the second one needs to reach the provider.

It lives on its own so every caller that generates from the chat pair applies
the SAME rescue. A background job that resolved the model one step short of
what the turn ran would work everywhere the configured set is fully pulled and
fail silently everywhere it is not - the kind of divergence that only shows up
on someone else's machine.

There are two ways a tag arrives here and they get opposite treatment.
CONFIGURED routing is rescued - :func:`route_around_missing` substitutes
another model when the configured one is absent, because a deployment whose
configuration and whose server disagree still has to answer, and so does one
that has not chosen a chat model at all. A name somebody TYPED is never
substituted: :func:`match_installed_name` only ever resolves the name to what
it already named, and refuses when it cannot do that unambiguously.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from app.providers.base import (
    ModelInfo,
    ModelProvider,
    ProviderUnavailableError,
    installed_tag_matches,
    tag_aliases,
)
from app.services.model_facts import runtime_capabilities
from app.services.model_resolution import ChatModelPair

logger = logging.getLogger(__name__)


class NoChatModelConfiguredError(Exception):
    """Raised when the model server holds nothing this deployment can chat with.

    A first-run state, not a fault: a fresh deployment pointed at an empty
    server, or at one whose whole inventory is embedders. It is a TYPED
    exception rather than a substitution because there is no honest answer to
    substitute - every caller either tells the user what to do about it (the
    chat turn's terminal error) or steps aside (the title job).
    """


# The message every raise carries. One string, because both raising branches
# describe the same situation from the user's side - the difference between
# "the server listed nothing" and "the server listed only embedders" changes
# nothing about the fix.
NO_CHAT_MODEL_MESSAGE = (
    "The model server holds no model this deployment can chat with. Pull a chat "
    "model - the Model Manager can - or point the deployment at a server that "
    "has one."
)


def _substitution_note(missing: str, chosen: str) -> str:
    """The substitution note for a configured tag the server does not hold."""
    return (
        f"Configured model '{missing}' is not installed on the model "
        f"server; answering with '{chosen}' instead. Install "
        f"'{missing}', or set the default model to one the server holds, "
        "to make this permanent."
    )


def _unconfigured_note(chosen: str) -> str:
    """The note for a deployment that has not chosen a chat model yet.

    Distinct from :func:`_substitution_note` because nothing was substituted:
    there is no missing tag to install and no mistake to correct, only a choice
    nobody has made. Naming a model that was never configured would send a
    reader looking for a setting they never wrote.
    """
    return (
        f"No default chat model is configured for this workspace; answering with "
        f"'{chosen}', which this model server holds. Choose a default to make it "
        "permanent."
    )


async def route_around_missing(
    provider: ModelProvider,
    provider_name: str,
    chat_pair: ChatModelPair,
    model: str,
    think: bool,
    *,
    listing: Sequence[ModelInfo] | None = None,
) -> tuple[str, str | None]:
    """Swap a config-routed model that is not installed for one that is.

    Only configured routing is rescued this way; an explicitly requested model
    is never substituted (callers do not route explicit picks through here).

    The decision table, in order:

    1. The model is installed - unchanged, no note.
    2. Its pair partner is installed - substitute, with the pair note. A
       deployment may route between a model PAIR (default + thinking) and have
       pulled only one of them, so the sibling is the closest match there is.
    3. Otherwise the first tag in the PROVIDER'S OWN LISTING ORDER whose
       runtime report says it can complete - substitute, with the general
       note. Listing order is deliberate and is NOT a quality ranking: this
       module has no way to rank weights and must not invent one, so it takes
       the server's order, which is at least deterministic and inspectable.
       Two cases reach here and they get different notes: a configured tag the
       server does not hold, and a deployment that has configured no chat model
       at all (the blank tag), where nothing was substituted and the note says
       so.
    4. Nothing is installed at all - :class:`NoChatModelConfiguredError`.
    5. Something is installed, at least one probe ANSWERED, and none of them
       can complete - the same error. The listing is known and the honest
       answer is that nothing here can chat.
    6. Something is installed but every probe FAILED - keep the model, no
       note. A probe failure is unknown, never evidence: refusing the turn on
       an unreachable metadata endpoint would replace the provider's own
       precise error with a guess about configuration. A BLANK model does not
       take this exit: keeping it would hand the runtime an empty tag, so an
       unreadable inventory raises instead.

    A probe failure inside the scan (rule 3) is skipped rather than read as
    "cannot complete", for the same reason as rule 6.

    Args:
        provider: The active model provider.
        provider_name: Its configured name (``settings.model_provider``), for
            the runtime-facts cache key.
        chat_pair: The turn's resolved (default, thinking) chat tags.
        model: The model the config routing resolved to, or empty when nothing
            configured one.
        think: Whether this turn resolved to thinking (decides the partner).
        listing: The provider's installed models, when the caller already has
            them. A turn lists once and hands the result to everything that
            needs it; None means "fetch it here", and a provider that cannot
            be reached then keeps the model with no note (its own typed error
            on generation tells the story).

    Returns:
        The model to use, plus a human-safe substitution note for the SSE
        status channel (None when nothing was substituted).

    Raises:
        NoChatModelConfiguredError: The server's inventory is known and holds
            no model that can chat.
    """
    if listing is None:
        try:
            listing = await provider.list_models()
        except ProviderUnavailableError:
            if not model:
                # An empty tag is not something to keep: the runtime would be
                # asked to generate on nothing. With the inventory unreadable
                # there is no candidate either, so the honest answer is the
                # typed first-run error rather than a request that cannot work.
                raise NoChatModelConfiguredError(NO_CHAT_MODEL_MESSAGE) from None
            return model, None
    installed = {info.name for info in listing}
    if model and installed_tag_matches(model, installed):
        return model, None

    partner = chat_pair.default if think else chat_pair.thinking
    if model and partner and partner != model and installed_tag_matches(partner, installed):
        # Gated on ``model`` as well as on the partner, because the two notes
        # say different things and only one of them is true of a blank tag. A
        # deployment that configured only a thinking sibling resolves an empty
        # default, and the substitution sentence would then name a model nobody
        # configured and tell the reader to install it.
        logger.warning("Configured chat model %r is not installed; substituting %r", model, partner)
        return partner, (
            f"Configured model '{model}' is not installed; answering with "
            f"'{partner}' instead. Download the configured model, or set the "
            "default to one the server holds, to make this permanent."
        )
    if not model and partner and installed_tag_matches(partner, installed):
        logger.info("No chat model is configured; answering on the paired model %r", partner)
        return partner, _unconfigured_note(partner)

    if not installed:
        raise NoChatModelConfiguredError(NO_CHAT_MODEL_MESSAGE)
    answered = False
    for info in listing:
        caps = await runtime_capabilities(provider, provider_name, info.name)
        if caps is None:
            continue
        answered = True
        if caps.completion:
            if model:
                logger.warning(
                    "Configured chat model %r is not installed; answering on %r instead",
                    model,
                    info.name,
                )
                return info.name, _substitution_note(model, info.name)
            logger.info("No chat model is configured; answering on %r", info.name)
            return info.name, _unconfigured_note(info.name)
    if not answered and model:
        # Every probe failed: the inventory is a list of names nothing is known
        # about, which is not grounds to refuse a turn that named a model.
        return model, None
    raise NoChatModelConfiguredError(NO_CHAT_MODEL_MESSAGE)


@dataclass(frozen=True)
class InstalledNameMatch:
    """What the installed inventory made of a name somebody typed.

    Three outcomes, told apart by which field is set rather than by a status
    code, because a caller has to branch on all three regardless and a pair of
    fields cannot be read in the wrong order:

    * ``model`` set - exactly one installed tag; use it.
    * ``candidates`` non-empty - the name fits several, and choosing here would
      load a model nobody asked for. The caller refuses and names them.
    * neither - nothing this server holds fits the name.

    Attributes:
        model: The installed tag the name resolved to, or None.
        candidates: The installed tags the name fits, when it fits more than
            one. Empty in the other two outcomes.
        by_fragment: True when the name matched as a FRAGMENT rather than as a
            tag. Carried because the two are different kinds of claim and a
            caller may reasonably guard only one: an exact name is what
            somebody asked for, while a fragment win is what this function
            INFERRED from it, and an inference is the half that can be wrong
            about which model was meant. False in the other two outcomes.
    """

    model: str | None = None
    candidates: tuple[str, ...] = ()
    by_fragment: bool = False


def _identity_keys(tag: str) -> set[str]:
    """Every spelling of ``tag`` that names the same model, for comparison.

    Alias expansion is the seam's (:func:`app.providers.base.tag_aliases`), so
    the ``:latest`` and namespace conventions stay a backend fact. Case is
    folded on top because the runtime lowercases a library name while resolving
    it, which makes ``QWEN3`` and ``qwen3`` one set of local weights. Folding is
    applied to Hugging Face names too, unlike the download guard's stricter key
    - the cost of over-folding differs by two orders of magnitude between the
    two. There it would refuse a legitimately different download as a
    duplicate; here it can only pick a tag the server already holds, and the
    caller reports which one it picked.
    """
    return {alias.casefold() for alias in tag_aliases(tag)}


def match_installed_name(name: str, installed: Sequence[str]) -> InstalledNameMatch:
    """Resolve a typed-in model name against the tags a server actually holds.

    Exists so naming a model by hand does not demand the exact tag: a person
    warming a model types what they remember, and the difference between
    ``qwen3`` and ``qwen3:4b-instruct`` is bookkeeping the machine can do. The
    forgiveness is deliberately narrow, in this order:

    1. An exact tag, allowing for the ways one model can be written down
       (``qwen3`` and ``qwen3:latest`` are the same weights, in both
       directions). This wins outright, so a name that is itself installed is
       never read as a fragment of a longer one.
    2. Otherwise a case-insensitive substring of exactly ONE installed tag.

    Anything else refuses. A fragment fitting several tags returns them rather
    than picking one: the shortest, the first alphabetically and the first the
    server listed are all arbitrary, and warming the wrong one on a card that
    fits a single model evicts whatever was already there. A fragment fitting
    none returns nothing, so the caller can say so.

    Pure - no IO, no provider, no settings - which is what makes the table of
    inputs above exhaustively testable.

    Args:
        name: The model name exactly as it was typed.
        installed: Tag names as reported by the provider's listing. Not
            filtered by role: narrowing it to chat-capable tags would cost a
            capability probe per tag, and a name that resolves to an embedder
            is refused by the model server itself with a better message than a
            guess made here.

    Returns:
        An :class:`InstalledNameMatch` in one of its three states.
    """
    wanted = name.strip()
    if not wanted:
        return InstalledNameMatch()
    keys = _identity_keys(wanted)
    exact = [tag for tag in installed if _identity_keys(tag) & keys]
    if len(exact) == 1:
        return InstalledNameMatch(model=exact[0])
    if exact:
        # Two installed tags claiming one identity (``qwen3`` beside
        # ``qwen3:latest``) is a server oddity, not a name to guess through.
        return InstalledNameMatch(candidates=tuple(exact))
    needle = wanted.casefold()
    partial = [tag for tag in installed if needle in tag.casefold()]
    if len(partial) == 1:
        return InstalledNameMatch(model=partial[0], by_fragment=True)
    # Several, or none: both are "cannot resolve this", and the empty tuple is
    # exactly what the caller needs to tell the two apart.
    return InstalledNameMatch(candidates=tuple(partial))
