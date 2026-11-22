"""Check whether the served model can actually drive the tool loop.

A model's agency ceiling and reasoning-with-tools grant are claims, and a
claim nobody measured is a guess. This script is the measurement: run it with
`make tool-check`, which pipes this file into the backend container where the
app package and the model provider's network are both available.
It drives the provider directly - a canned tool list, canned tool results,
never a real tool - through the cases that separate a model that can drive
the loop from one that cannot (must-call, must-NOT-call, selection, the
invented-tool guard, continuation after a result, a loop guard, and
reasoning-with-tools). Run it AFTER `make warm`: it is a diagnostic to run on
demand against the warm model at serve-time settings, never at boot and never
on a chat turn.

FAST runs by default: must-call plus must-not-call, one sample each, earning
at most an `assisted` verdict. TOOL_CHECK_DEEP=1 runs every case three times,
which is what can earn `agentic` and `thinking_with_tools` - and on a
pair-routed tier it checks BOTH tags of the model pair, since a verdict for
one says nothing about the other. TOOL_CHECK_JSON=1 prints the summary as
JSON instead of the table.

"Serve-time" reaches into the stored workspace settings, which is why this
reads them before it measures anything: they decide which tag serves by default
and how large a window a turn on it asks for. The window is one of the two
serve-time options a deployment can change without touching the weights, so a
verdict earned at a different one describes a configuration nothing here runs.
Each sample records the window it ran at, and the recorded verdict names it.

The think flag is the other, and it resolves through the whole chain a turn
uses - including the two rungs that live in the database, a stored operator
declaration and a previous check's measurement. Those reach the harness as
injected callables, built by the same factory the console's own check uses
(`model_verify.serve_time_think_fetchers`), because a row records the running
harness version and that version is a promise that any two rows under it
measured the same thing. A terminal run that resolved the flag more weakly
would drive a declared always-thinking tag with the flag cleared, leak a
`<think>` block into its answers, fail on it, and write a `deterministic` row -
and a failure clamps whatever produced it, so the terminal would quietly demote
a model the console had just certified.

Which PARTNER a deep run's reasoning case drives resolves the same way, through
the same shared ranking (`model_resolution.thinking_sibling_for`): environment
override, then a stored operator pairing. That verdict binds
to the partner's weights, so the two writers of the log - this script and the
console's verify job - have to name the same tag or neither row can be read
back.

TOOL_CHECK_NO_DB is the one arm that resolves without them, and it has to be:
the flag exists for a stack whose database is down, so it also skips the stored
settings, the stored pairing, the two think-flag declaration reads, and the
write. Such a run answers from the code-shipped quirk seed and the
runtime's template report, which is the whole chain wherever nothing has been
declared or measured for a tag. Each sample records the flag it ran with either
way, so a row always says which kind of run wrote it.

The script reports; it does not gate. A completed run exits 0 whatever the
verdict earned - the number is the point, and a failing model is a finding,
not an error. Exit 1 means only that the run itself could not happen: the
model provider was unreachable.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import sys

from app.core.config import Settings
from app.core.factories import build_model_provider
from app.core.tenancy import DEFAULT_TENANT_ID, DEFAULT_USER_ID, TenantContext
from app.db.engine import build_engine, build_sessionmaker
from app.providers.base import ProviderUnavailableError, installed_tag_matches
from app.repositories.model_probes import ModelProbeRepository, row_from_summary
from app.schemas.settings import WorkspaceDefaults
from app.services import tool_check
from app.services.model_policy import policy_for
from app.services.model_resolution import (
    resolve_chat_models,
    thinking_sibling_for,
)
from app.services.model_verify import serve_time_think_fetchers
from app.services.workspace_defaults import load_defaults

_TRUTHY = {"1", "true", "yes"}
DEEP = os.environ.get("TOOL_CHECK_DEEP", "").strip().lower() in _TRUTHY
AS_JSON = os.environ.get("TOOL_CHECK_JSON", "").strip().lower() in _TRUTHY
# Verdicts persist to the model_probes measurement log by default - that log
# is the whole reason the harness exists. NO_DB skips the write (hermetic
# runs, a stack whose database is down but whose model is worth probing) - and
# with it the stored-settings READ, so such a run also measures the
# deployment's own model at its own window rather than the workspace's.
NO_DB = os.environ.get("TOOL_CHECK_NO_DB", "").strip().lower() in _TRUTHY


def _fmt_rate(rate: float | None) -> str:
    return "-" if rate is None else f"{rate:.1f}"


async def _workspace_defaults(settings: Settings) -> WorkspaceDefaults:
    """Read the stored workspace defaults, or the unset view when there are none.

    A probe has to run what this deployment's turns run, and two halves of that
    are settings rather than configuration: WHICH tag serves by default, and how
    big a window a turn on it asks for. A diagnostic that read neither would
    measure some other tag at some other window - which on a
    deployment that has set either one is a model nobody runs, sized for nobody.
    Before identity is real there is exactly one workspace and it IS the
    deployment, which is why a terminal run may read the seeded principal's
    settings without being handed a principal.

    Best-effort: a database that will not answer costs the settings, never the
    run. TOOL_CHECK_NO_DB skips this read outright - the flag exists for a
    stack whose database is down, and reading one here would be the flag
    failing to mean what it says.

    Args:
        settings: The active settings (the database URL the engine needs).

    Returns:
        The typed defaults; every field unset when nothing is stored, the read
        failed, or the run was told to leave the database alone.
    """
    if NO_DB:
        return WorkspaceDefaults()
    engine = None
    try:
        engine = build_engine(settings)
        sessionmaker = build_sessionmaker(engine)
        async with sessionmaker() as session:
            return await load_defaults(
                session, TenantContext(tenant_id=DEFAULT_TENANT_ID, user_id=DEFAULT_USER_ID)
            )
    except Exception as exc:  # noqa: BLE001 - a settings read must not fail the run
        print(
            f"warning: could not read the workspace defaults ({exc}); "
            "checking the deployment's own model at its own window",
            file=sys.stderr,
        )
        return WorkspaceDefaults()
    finally:
        # Built inside the try, so a URL the engine refuses is a warning rather
        # than a traceback - and there is nothing to dispose of when it was.
        if engine is not None:
            await engine.dispose()


async def _listed_digests(settings: Settings) -> dict[str, str]:
    """List the backend, for the ONE thing the declaration read needs it for.

    Declarations are keyed on the name the PROVIDER lists, because that is the
    spelling the writer had in hand, and a read arriving with the CONFIGURED
    spelling is canonicalized against a listing before it queries
    (:func:`app.services.model_policy.policy_for`). Without one, a deployment
    that configures ``qwen3`` against an installed ``qwen3:latest`` reads no
    declaration at all - which here means silently driving the wrong
    partner while the console drives the declared one, the exact disagreement
    between the two writers this resolution exists to prevent.

    Its own throwaway client, closed in the loop that opened it: an async client
    binds to its loop, and the provider the RUN uses must reach its first sample
    with a client that has never been touched by another one.

    Best-effort: an unreachable backend costs the canonicalization, not the run.
    The read then falls back to the configured spelling, which is what it had
    before this existed and is correct on every deployment that spells its tags
    the way the runtime lists them.

    Args:
        settings: The active settings (the backend URL the provider needs).

    Returns:
        Installed name -> digest, or an empty map when the backend would not say.
    """
    provider = build_model_provider(settings)
    try:
        return {info.name: info.digest for info in await provider.list_models() if info.digest}
    except Exception as exc:  # noqa: BLE001 - a listing must not fail the run
        print(
            f"warning: could not list the backend to match stored declarations ({exc}); "
            "reading them under the configured spelling",
            file=sys.stderr,
        )
        return {}
    finally:
        aclose = getattr(provider, "aclose", None)
        if callable(aclose):
            await aclose()


async def _declared_pairing(settings: Settings, model: str) -> str | None:
    """Read the operator's stored thinking partner for ``model``, or None.

    The rung the environment cannot supply, and the one
    that decides which weights this run's reasoning case measures. The console's
    verify job reads it too, so that both writers of the measurement log drive
    the sibling a TURN routes to; a check that skipped it would drive the
    partner, record its digest, and hand the log two rows that
    disagree about which pair was measured.

    Best-effort on the same rule as the settings read above: a database that
    will not answer costs the rung, never the run.

    Args:
        settings: The active settings (the database URL the engine needs).
        model: The chat tag this run is about to check.

    Returns:
        The stored partner tag, or None when nothing is paired, the read failed,
        or the run was told to leave the database alone.
    """
    if NO_DB or not model:
        return None
    engine = None
    try:
        digests = await _listed_digests(settings)
        engine = build_engine(settings)
        view = await policy_for(
            provider_name=settings.model_provider,
            model=model,
            digests=digests,
            sessionmaker=build_sessionmaker(engine),
        )
        return None if view is None else view.thinking_model
    except Exception as exc:  # noqa: BLE001 - a declaration read must not fail the run
        print(
            f"warning: could not read the stored thinking pair ({exc}); "
            "resolving the partner from the environment alone",
            file=sys.stderr,
        )
        return None
    finally:
        if engine is not None:
            await engine.dispose()


def _partner_for(settings: Settings, chat: str, declared: str | None) -> str:
    """Name the tag this run's reasoning case drives, through the shared ranking.

    A one-line wrapper, and named rather than inlined for one reason: the
    parity between the log's two writers is only worth claiming if it is
    testable, and a test can ask this the same question it asks the console's
    verify job and compare the two answers. The ranking itself lives in
    :func:`app.services.model_resolution.thinking_sibling_for`, which is where the
    console's job and the earned binding read it too.

    Args:
        settings: The active settings (the environment rung).
        chat: The tag this run is checking.
        declared: The stored operator pairing for it, or None.

    Returns:
        The partner tag, or ``""`` for no pair - the empty string the harness
        reads as "stay on the single-model path".
    """
    return thinking_sibling_for(settings, chat, declared=declared) or ""


def _print_table(summary: tool_check.ToolCheckSummary) -> None:
    """Render the per-sample rows, the tallies, and the verdict."""
    print()
    header = f"{'case':<22} {'model':<20} {'pass':<5} {'time':>7} {'tok':>5}  detail"
    print(header)
    print("-" * len(header))
    for sample in summary.samples:
        print(
            f"{sample.case:<22} {sample.model:<20} "
            f"{'PASS' if sample.passed else 'FAIL':<5} "
            f"{sample.latency_s:>6.2f}s {sample.tokens:>5}  {sample.detail}"
        )
    print()
    tally_line = ", ".join(
        f"{name} {passed}/{run}" for name, (passed, run) in summary.tallies.items()
    )
    print(f"per-case: {tally_line}")
    print(f"tok/s: {_fmt_rate(summary.tokens_per_second)}")
    if summary.depth == "fast":
        print(
            "note: a fast check earns at most 'assisted'; set TOOL_CHECK_DEEP=1 "
            "to earn 'agentic' and 'thinking_with_tools'."
        )
    print(
        f"earned: {summary.earned}; thinking_with_tools: {str(summary.thinking_with_tools).lower()}"
    )
    # Reported on its own line rather than folded into the verdict above,
    # because it is not one: it says which CHANNEL this tag's reasoning comes
    # out on, which is a measurement about the weights and not a grade. It is
    # also the line worth reading against whatever mode was declared for the
    # tag - a disagreement there is what a re-pulled tag looks like.
    if summary.measured_think_mode is not None:
        print(f"think flag measured: {summary.measured_think_mode}")


def main() -> int:
    """Run the harness against the configured provider and report the verdicts."""
    settings = Settings()
    provider = build_model_provider(settings)
    # Read FIRST, and in its own loop: it decides both halves of what gets
    # checked, and it touches the database rather than the model server, so it
    # shares nothing with the provider's client below.
    defaults = asyncio.run(_workspace_defaults(settings))
    ceiling = defaults.context_ceiling
    # Check what actually SERVES: the same workspace > env-override >
    # env-override resolution the chat path uses, so a stored default is
    # probed rather than the tag it displaced. Existence is deliberately
    # not checked here, exactly as the title job does not check it - a stored
    # tag the backend no longer holds is a broken deployment whose turns are
    # already failing, and the samples say so plainly.
    chat = resolve_chat_models(settings, workspace_model=defaults.model).default
    thinking = _partner_for(settings, chat, asyncio.run(_declared_pairing(settings, chat)))

    if not chat:
        # The first-run state, said plainly rather than raised: nothing has
        # chosen a chat model, so there is nothing to measure and no failure to
        # report about a model. Every other refusal in this script prints a
        # sentence; a ValueError out of the harness would be the one that does
        # not.
        print(
            "No chat model is configured, so there is nothing to check. Set one in the "
            "console's workspace defaults (or DEFAULT_CHAT_MODEL), then run this again.",
            file=sys.stderr,
        )
        return 2
    if not AS_JSON:
        pair_note = f" (thinking pair: {thinking})" if thinking and DEEP else ""
        print(f"model: {chat}{pair_note}")
        print(f"depth: {'deep' if DEEP else 'fast'}  harness: {tool_check.harness_version()}")
        if ceiling is not None:
            print(f"context ceiling: {ceiling} (workspace setting)")

    def progress(sample: tool_check.CaseSample) -> None:
        # Progress goes to stderr so TOOL_CHECK_JSON output stays pipeable.
        stream = sys.stderr if AS_JSON else sys.stdout
        print(
            f"  ran {sample.case} on {sample.model}: "
            f"{'pass' if sample.passed else 'FAIL'} ({sample.latency_s:.2f}s)",
            file=stream,
            flush=True,
        )

    async def run() -> tuple[
        tool_check.ToolCheckSummary, str | None, str | None, str | None, int | None
    ]:
        # The two declaration reads, built HERE rather than reused from the
        # settings read above: an async engine binds to the loop that created
        # it, and that one ran in its own asyncio.run. A URL the engine refuses
        # is a warning and a weaker resolution, never a failed run - the same
        # trade the settings read makes, and the reads themselves already
        # degrade to "nothing declared" on their own.
        engine = None
        fetch_policy = None
        fetch_measured_think = None
        if not NO_DB:
            try:
                engine = build_engine(settings)
            except Exception as exc:  # noqa: BLE001 - a declaration read must not fail the run
                print(
                    f"warning: could not reach the database for the stored think mode ({exc}); "
                    "resolving the flag from the seed and the runtime alone",
                    file=sys.stderr,
                )
            else:
                fetch_policy, fetch_measured_think = serve_time_think_fetchers(
                    provider=provider,
                    provider_name=settings.model_provider,
                    sessionmaker=build_sessionmaker(engine),
                )
        # The provider owns a lazily-created HTTP client; close it inside the
        # loop that created it rather than letting asyncio.run tear it down.
        try:
            result = await tool_check.run_tool_check(
                provider,
                provider_name=settings.model_provider,
                deep=DEEP,
                chat_model=chat,
                thinking_model=thinking,
                context_ceiling=ceiling,
                # The other half of the window: what this deployment affords a
                # tag nothing sizes, for a workspace that named no ceiling.
                default_window=settings.default_context_window,
                # The SAME pair the console's check wires, from the same factory.
                # A terminal run that resolved the flag more weakly would drive a
                # declared always-thinking tag with the flag cleared and write the
                # resulting failure as a clamp - under the harness version the
                # console certifies under. See serve_time_think_fetchers.
                fetch_policy=fetch_policy,
                fetch_measured_think=fetch_measured_think,
                on_sample=progress,
            )
            # Measurement hooks, captured while the model is still resident:
            # the digest binds the verdict to the WEIGHTS the tag pointed at,
            # the runtime version to the serving stack, and the VRAM reading
            # feeds the Model Manager's measured-facts grid. All best-effort -
            # None is recorded as None ("unmeasured"), never estimated.
            digest = None
            thinking_digest = None
            try:
                for info in await provider.list_models():
                    if installed_tag_matches(chat, {info.name}):
                        digest = info.digest
                    if thinking and installed_tag_matches(thinking, {info.name}):
                        thinking_digest = info.digest
            except ProviderUnavailableError:
                pass
            runtime = await provider.runtime_version()
            vram = await provider.resident_vram_bytes(chat)
            return result, digest, thinking_digest, runtime, vram
        finally:
            aclose = getattr(provider, "aclose", None)
            if aclose is not None:
                await aclose()
            # Disposed in the loop that built it, for the same reason it was
            # built here; the write below opens its own.
            if engine is not None:
                await engine.dispose()

    try:
        summary, digest, thinking_digest, runtime, vram = asyncio.run(run())
    except ProviderUnavailableError as exc:
        print(f"error: the model provider is unreachable: {exc}", file=sys.stderr)
        return 1

    # The Ollama provider surfaces connection failures as typed error CHUNKS
    # rather than raising, so a run where every sample died that way is the
    # same infrastructure failure wearing a different shape.
    #
    # ALL here, where the verify job behind the endpoint uses ANY - a
    # difference that is deliberate, not drift. This run is ATTENDED: the table
    # below prints every sample's detail, so a partly-failed run is a finding a
    # person can read and act on, and discarding it would throw away the only
    # copy of what went wrong. The endpoint's run is UNATTENDED (it fires
    # automatically after a download), nobody is reading a table, and a single
    # blip there would silently record a verdict that clamps the running model.
    if summary.samples and all(
        sample.error_code == "provider_unavailable" for sample in summary.samples
    ):
        print(
            "error: the model provider was unreachable for every sample - "
            "is the stack up ('make up') and the model downloaded?",
            file=sys.stderr,
        )
        return 1

    if AS_JSON:
        # The summary travels with what the CLI captured alongside it, so a
        # JSON consumer sees exactly what persists (digest binds the verdict).
        summary.digest = digest
        payload = dataclasses.asdict(summary)
        payload["runtime_version"] = runtime
        payload["serving_endpoint"] = provider.serving_endpoint()
        payload["vram_bytes"] = vram
        print(json.dumps(payload, indent=2))
    else:
        _print_table(summary)

    if NO_DB:
        print("not recorded (TOOL_CHECK_NO_DB=1)", file=sys.stderr)
        return 0

    async def persist() -> str:
        engine = build_engine(settings)
        try:
            sessionmaker = build_sessionmaker(engine)
            async with sessionmaker() as session:
                row = await ModelProbeRepository(session).add(
                    row_from_summary(
                        summary,
                        provider_name=settings.model_provider,
                        digest=digest,
                        thinking_digest=thinking_digest,
                        runtime_version=runtime,
                        # WHERE this ran. The CLI and the management view write
                        # the same row, so a verdict earned from a terminal
                        # against a remote pod binds to that pod exactly as one
                        # earned from the page does.
                        serving_endpoint=provider.serving_endpoint(),
                        vram_bytes=vram,
                    )
                )
                await session.commit()
                return str(row.id)
        finally:
            await engine.dispose()

    # A failed write must be LOUD but not flip the exit code: the measurement
    # itself succeeded, and exit 1 is reserved for "the run could not happen".
    try:
        row_id = asyncio.run(persist())
    except Exception as exc:  # noqa: BLE001 - any persistence failure reports the same way
        print(f"warning: verdict NOT recorded in model_probes: {exc}", file=sys.stderr)
    else:
        short = (digest or "no-digest")[:19]
        # Stderr, like the progress lines: TOOL_CHECK_JSON stdout must stay
        # a single pipeable JSON document. The server is named because a
        # verdict binds to the host that earned it, and a run against a remote
        # backend is otherwise indistinguishable from a local one in the log.
        print(
            f"recorded: model_probes {row_id} ({short}, runtime "
            f"{runtime or 'unknown'}, server {provider.serving_endpoint() or 'unnamed'})",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
