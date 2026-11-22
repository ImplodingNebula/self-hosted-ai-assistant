"""Whether a turn may send raw images to the model it is about to run.

One question, one answer, one place - because the two halves of getting it
wrong are both silent. Sending pixels to weights that cannot take them wastes a
quarter of the window and produces an answer about nothing; refusing pixels a
model could have read leaves the user told their picture was not seen while a
capable model sat idle.

The runtime's own report decides it, read through the shared facts cache. A tag
is mutable - a re-pull can swap the weights under it - so nothing written down
about a name could ever have been enough on its own; the measurement always had
to agree, and it is now the whole answer.
"""

from __future__ import annotations

import logging

from app.providers.base import ModelProvider
from app.services.model_facts import (
    VISION_MAX_AGE_SECONDS,
    effective_vision,
    runtime_capabilities,
)

logger = logging.getLogger(__name__)


async def can_see(*, provider: ModelProvider, provider_name: str, model: str) -> bool:
    """Return whether ``model`` may be sent raw images this turn.

    Reports are read through the shared facts cache, so a turn, the model
    listing and the management view can never answer differently about the same
    weights. This caller accepts a much fresher answer than the others
    (:data:`~app.services.model_facts.VISION_MAX_AGE_SECONDS`), because the
    failure directions are opposite: a stale capability read is harmless on a
    card and not harmless here, where it keeps sending pixels to weights that
    may no longer accept them. It is still ONE cache - a stricter reader simply
    re-probes past its own limit and leaves the fresher answer behind for
    everyone else.

    A probe that fails - unreachable server, timeout, an unexpected payload -
    resolves to False with a warning, never to True. The honest reading of "the
    runtime would not tell us" is that the pixels do not go, and the caller
    keeps telling the user the picture was not shown.

    Args:
        provider: The model provider serving this turn.
        provider_name: Its configured name, for the cache key.
        model: The exact tag the turn will generate with. Must be resolved
            AFTER any model substitution, or the probe describes weights that
            are not the ones about to run.

    Returns:
        True only when the resolution admits the pixels.
    """
    if not model:
        return False
    report = await runtime_capabilities(
        provider, provider_name, model, max_age_seconds=VISION_MAX_AGE_SECONDS
    )
    if report is None:
        # The failure itself is already logged with its traceback where the
        # probe was taken; this line says what it COST, which is the half a
        # reader chasing a missing picture is looking for.
        logger.info(
            "No capability report for %s on %s; treating as no-vision this turn",
            model,
            provider_name,
        )
        return False
    return effective_vision(model, report)
