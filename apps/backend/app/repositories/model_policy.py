"""The model_policy repository - deliberately MACHINE-scoped, not tenant-scoped.

The second sanctioned exception to the tenant-injecting base repository, on
exactly the pattern ``model_probes`` established: a declaration describes the
shared model server, so scoping it to a tenant would let two workspaces hold
contradictory answers about one set of weights. The declaring tenant/user ride
along as audit-only columns; no method filters by them, and none accepts an
identity argument at all - there is nothing here an identity could scope.

Append-only by design, and here that is load-bearing rather than tidy. The
effective policy is FOLDED per axis from recent rows
(:mod:`app.services.model_policy`), which is what lets a declaration made
against one digest retire on its own when the weights change. An update method
- or a write that copied prior axes forward - would carry a grant across a
re-pull, so there is no such method to misuse.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entities import ModelPolicy


class ModelPolicyRepository:
    """Append and read operator declarations for the shared model server."""

    def __init__(self, session: AsyncSession) -> None:
        """Bind to a session; no tenant - see the module docstring."""
        self.session = session

    async def add(self, policy: ModelPolicy) -> ModelPolicy:
        """Append one declaration row (flushed, not committed - caller commits).

        One operator action writes ONE row carrying every axis that action set.
        Rows are ordered by ``created_at`` alone, so two rows written for the
        same model in the same instant have no defined order between them and
        the fold could read either as the newer - splitting one declaration
        across several rows is therefore a bug, not a style.

        Args:
            policy: The unsaved row.

        Returns:
            The same row, flushed so its id is usable.
        """
        self.session.add(policy)
        await self.session.flush()
        return policy

    async def recent(self, provider: str, model: str, *, limit: int = 50) -> list[ModelPolicy]:
        """Return the recent declarations for a (provider, model), newest first.

        Deliberately NOT digest-filtered, unlike the probe log's equivalent
        read. Half the axes here follow the TAG rather than the weights (a
        context window an operator chose for this model is not a claim about a
        checkpoint), so a digest-bound query would drop them on every re-pull.
        Which rows still bind is decided one axis at a time by the fold in
        :mod:`app.services.model_policy`, where both rules can be stated
        beside each other.

        Args:
            provider: The ``MODEL_PROVIDER`` selector value rows key on.
            model: The installed spelling of the tag.
            limit: How many rows back to read. The fold only needs the newest
                row per axis, so this bounds the read rather than the answer -
                but only while each axis is still represented inside the
                window. An axis pushed out of it reads as never declared, which
                for a ceiling means UNCLAMPING, so the caller's bound is set
                far past any plausible history rather than tuned for the query.

        Returns:
            The rows, newest first. Ties on ``created_at`` break on ``id``:
            arbitrary, but stable, so one fold cannot answer two ways.
        """
        result = await self.session.execute(
            select(ModelPolicy)
            .where(ModelPolicy.provider == provider, ModelPolicy.model == model)
            .order_by(
                ModelPolicy.created_at.desc(),  # type: ignore[attr-defined]
                ModelPolicy.id.desc(),  # type: ignore[attr-defined]
            )
            .limit(limit)
        )
        return list(result.scalars().all())
