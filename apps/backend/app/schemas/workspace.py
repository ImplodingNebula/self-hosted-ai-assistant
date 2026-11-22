"""HTTP schemas for the workspace-defaults API - the frozen wire contract.

The read side of the settings page. The WRITE side is
:class:`app.schemas.settings.WorkspaceDefaultsPatch`, which lives beside the
stored typed views on purpose: the patch and the blob it merges into have to
agree field for field, and splitting them across two modules is how a stored
key and a wire key drift apart.

What this adds over the stored values is the DEPLOYMENT facts a page needs to
render them honestly - whether a reranker exists for the toggle to govern, and
what range the context ceiling accepts - so the page never has to guess, and
never shows a control for something this deployment cannot do.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.schemas.settings import (
    MAX_CONTEXT_CEILING,
    MAX_SYSTEM_PROMPT_CHARS,
    MIN_CONTEXT_CEILING,
    SettingMode,
    WorkspaceDefaults,
)
from app.services.workspace_defaults import RerankerAvailability


class RerankerStatusOut(BaseModel):
    """Whether this deployment has a reranker, and what to say when it does not.

    Attributes:
        configured: A reranker would run if the workspace permitted it. False
            means the toggle governs nothing, and the page shows a status
            instead of a switch.
        model: The sidecar's model tag, when one is configured.
        detail: One sentence for the page to render as visible text.
    """

    configured: bool
    model: str | None
    detail: str

    @classmethod
    def from_availability(cls, availability: RerankerAvailability) -> RerankerStatusOut:
        """Project the service value onto the wire.

        Args:
            availability: What the deployment reports about its reranker.

        Returns:
            The wire form.
        """
        return cls(
            configured=availability.configured,
            model=availability.model,
            detail=availability.detail,
        )


class WorkspaceDefaultsOut(BaseModel):
    """The tenant's stored defaults plus the facts needed to render them.

    Every stored field is nullable and null means UNSET - the tier below it in
    the resolution chain decides. That is deliberately distinct from a value:
    a null thinking default lets the per-turn heuristic run, where ``"auto"``
    stored pins the heuristic explicitly.

    Attributes:
        model: The workspace's default chat tag, or null.
        embedding_model: The tag retrieval embeds with, or null to let the
            deployment answer - which it does from the model its stored
            passages were written by, then from the one embedding model the
            server holds when it holds exactly one.
            Null therefore does NOT mean "no embedder": it means nobody has
            chosen one here.
        thinking: The workspace's default thinking mode, or null.
        web_search: The workspace's default web-search mode, or null.
        system_prompt: The persona every turn here opens with, or null when
            the deployment's own persona stands.
        context_ceiling: The largest window any turn may request, or null to
            leave the size to the deployment's configured default. It never
            exceeds the window the weights themselves admit; below that it is
            the deployment's affordance outright, so it can raise a window as
            readily as lower one.
        reranker: False when retrieval is forced dense-only; true or null when
            the deployment's reranker (if any) may run.
        reranker_status: What this deployment can actually do about that.
        min_context_ceiling: The smallest ceiling the API accepts.
        max_context_ceiling: The largest.
        max_system_prompt_chars: The longest persona the API accepts, so the
            page can say the limit rather than discover it in a 422.
    """

    model: str | None = None
    embedding_model: str | None = None
    thinking: SettingMode | None = None
    web_search: SettingMode | None = None
    system_prompt: str | None = None
    context_ceiling: int | None = None
    reranker: bool | None = None
    reranker_status: RerankerStatusOut
    min_context_ceiling: int = Field(default=MIN_CONTEXT_CEILING)
    max_context_ceiling: int = Field(default=MAX_CONTEXT_CEILING)
    max_system_prompt_chars: int = Field(default=MAX_SYSTEM_PROMPT_CHARS)

    @classmethod
    def build(
        cls, defaults: WorkspaceDefaults, availability: RerankerAvailability
    ) -> WorkspaceDefaultsOut:
        """Assemble the response from the stored defaults and the deployment facts.

        Args:
            defaults: The tenant's stored defaults.
            availability: What this deployment reports about its reranker.

        Returns:
            The response body.
        """
        return cls(
            model=defaults.model,
            embedding_model=defaults.embedding_model,
            thinking=defaults.thinking,
            web_search=defaults.web_search,
            system_prompt=defaults.system_prompt,
            context_ceiling=defaults.context_ceiling,
            reranker=defaults.reranker,
            reranker_status=RerankerStatusOut.from_availability(availability),
        )
