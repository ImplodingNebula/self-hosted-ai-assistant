"""The tool-check harness: a measured verdict on a model's tool-calling.

It measures one more thing than its name says: how the model actually treats
the THINK FLAG. The two questions share a harness because they share the
expensive part - a loaded model at serve-time options - and because the flag
decides which channel reasoning comes out on, which is a fact about the same
weights the tool cases are measuring. See the THINKING case group below.

``make tool-check`` is the opt-in diagnostic that earns a model its agency
ceiling and its reasoning-with-tools grant (see
docs/ARCHITECTURE.md). This module is the PURE service behind it -
no database, no HTTP server, no queue: it drives a
:class:`~app.providers.base.ModelProvider` directly with a canned tool list
and canned tool results, so a run never executes a real tool and touches no
network beyond the model provider itself. The CLI (``scripts/tool_check.py``)
and, later, the Model Manager's verify job wrap this one service, so a
terminal run and a UI click cannot disagree about what was measured.

The cases are the ones that actually separate a model that can drive the tool
loop from one that cannot: must-call in two framings (a freshness ask, and a
workspace-record question phrased personally - the framing measured to drop a
small model into an "I don't have access" refusal), must-NOT-call (the
sharpest blade: a model that reaches for a tool on "what is 2 + 2" will reach
for one everywhere), selection among tools, a guard against inventing a tool
that was never offered, a multi-turn continuation where a result is fed back
(where the published function-calling scores collapse), a loop guard, and the
reasoning-with-tools case whose failure mode is silent EMPTY output
(ollama/ollama#10976) - which is exactly why ``thinking_with_tools`` defaults
to false and must be earned, never assumed.

Two depths, and the depth caps what a verdict can earn: FAST (must-call plus
must-not-call, one sample each) earns at most ``assisted``; DEEP (every case,
three samples each) can earn ``agentic`` and ``thinking_with_tools``. On a
pair-routed deployment, DEEP runs the reasoning case against the pair's
thinking tag and everything else against the chat tag - a verdict
for one tag says nothing about the other - while FAST touches only the chat
tag. Every TOOL sample runs at serve-time options: the window a turn on this
deployment would request, and the think flag the same chain a turn uses
resolves for that tag.

"Serve-time" is meant literally for both halves, and each of them used to be
read off a committed file alone.

The think flag now goes through
:func:`app.services.model_facts.effective_think_mode` - the resolver the turn,
the warm and the title job already share, which ranks a stored declaration and
an unconfirmed measurement ABOVE everything, and the quirk seed and the
runtime's template inference below those. That file answered NOTHING for a tag
it did not name, which is what a free-text pull is and what most of a remote
server's inventory is: that resolved to "unknown", so an always-thinking model
was driven with the flag CLEARED - the configuration that leaks a raw
reasoning block into the visible answer, which the harness's own emission
check could then score as a failure of the model. Said plainly, the tool cases
measured a think configuration the deployment does not run, the same defect the
window half was repaired for and on the same tags.

The two DATABASE-backed rungs of that chain arrive as INJECTED CALLABLES
(:data:`PolicyFetcher`, :data:`MeasuredThinkFetcher`), for exactly the reason
the context ceiling arrives as an argument: this service still touches no
database, and it must not import the verdict log that reads
:func:`harness_version` back out of it. The verify job holds a session and wires
both; the command-line diagnostic passes neither and resolves the SESSIONLESS
half - the quirk seed and the runtime report. A rung nobody wired reads as
"nothing declared, nothing measured", never as a different answer.

What that costs is worth stating exactly, because it is not "one rung on
undescribed tags". The measurement rung sits ABOVE a declaration (a measured
``always`` outranks every declaration - see
:func:`~app.services.model_facts.effective_think_mode`), so a sessionless run
can differ from a wired one on a DECLARED tag too. A run that reads neither
is answering from the seed and the template, which is the whole
chain only where nothing has been declared or measured for the tag.

Feeding a previous MEASUREMENT into a run that is about to measure again is
deliberate rather than circular. What a turn on these weights would send is
exactly what the tool cases have to be driven with, and the THINKING group - the
only thing that produces that measurement - is exempt from the whole resolution
(see below), so no run can confirm itself.

For the window that literalness is what
:func:`app.services.model_facts.resolve_num_ctx` is doing here rather than the
declared window a committed file used to be able to state. Such a file
answered for the tags it named and NOTHING for the rest, so a probe used to run
unsized - the runtime picking its own VRAM-scaled default, 4096 on most cards -
while every real turn on that tag asked for a resolved window. A verdict earned
that way describes a configuration nothing serves, and the disagreement costs a
model reload between the check and the re-warm that follows it. The workspace
context ceiling is part of the same answer: where an operator has clamped their
turns, the certificate has to be earned at the clamped window or it certifies
somebody else's deployment.

The ceiling arrives as an ARGUMENT for the same reason the two think-mode reads
arrive as callables - this service still touches no database. The caller that
has a tenant to read for (the verify job, the CLI) loads it and passes it down;
a caller with none passes nothing and probes unclamped, which is the same window
such a deployment's turns request. A LATER change to that ceiling is
display-only: it does not clamp, revert, or stale any verdict already earned.
Staleness is keyed on the digest, the harness version and the serving endpoint -
facts about the weights and the machine - and a settings edit is none of those.

The THINKING group is the exception to three of the rules above, on purpose:

- It runs at BOTH depths. Two short generations with no tools is cheap enough
  that withholding it from the fast check would only mean the answer is
  missing exactly where a fresh pull needs it - the automatic post-download
  check is a fast one.
- It drives the tag UNDER CHECK, never the pair's thinking sibling. The
  question is what THIS tag does when the flag is set and when it is cleared;
  routing the ``think=True`` half to a sibling that always reasons would
  measure the sibling and answer about the wrong weights.
- It sends the flag LITERALLY, opting out of the resolution above entirely.
  Resolving is what the chat path does to avoid a 400 on weights already known
  not to reason - here every source in that chain (the declaration, the seed,
  the template, and a previous run of this very group) is the thing under test,
  so pre-resolving would measure the claim instead of the weights. A model that
  rejects the flag gets the provider's own retry-without-think, which reads as
  "no reasoning came back" and classifies as ``never`` rather than erroring.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from app.providers.base import ChatChunk, ChatMessage, GenerationOptions, ModelProvider, ToolCall
from app.services.model_facts import (
    effective_think_mode,
    report_for,
    resolve_num_ctx,
)
from app.services.model_policy import PolicyView
from app.services.model_resolution import (
    AgencyMode,
    ThinkMode,
)
from app.services.prompts import (
    DEFAULT_SYSTEM_PROMPT,
    append_tool_guidance,
    build_tool_result_message,
)
from app.services.thinking import normalize_think

Depth = Literal["fast", "deep"]

# The stored operator declarations for one tag, folded and already digest-gated
# (None when nothing is declared), and what a check MEASURED about that tag's
# think flag (None when nothing has). Both are DATABASE reads, so they are
# injected rather than taken here - see the module docstring on why this service
# holds no session and, in particular, why it must not import
# :mod:`app.services.earned`, which reads :func:`harness_version` out of this
# module. Same shapes and same names as the chat planner's
# (:mod:`app.services.turn_plan`), so a reader meeting them twice meets one idea.
PolicyFetcher = Callable[[str], Awaitable[PolicyView | None]]
MeasuredThinkFetcher = Callable[[str], Awaitable[ThinkMode | None]]

WEB_SEARCH = "web_search"
SEARCH_KNOWLEDGE = "search_knowledge"

# The mini loop's round cap - the loop-guard bound: a model still requesting
# tools after three passes, with a result already in hand, is chasing rather
# than answering. Deliberately at or below the live loop's tool_max_iterations,
# because a probe should fail a chaser before it burns more rounds of GPU time.
_MAX_ROUNDS = 3

# Samples per case at deep depth. One sample proves nothing about a stochastic
# decoder; three lets a majority absorb one unlucky sample without letting a
# coin-flip model pass.
_DEEP_SAMPLES = 3

# Per-result cap handed to the fence, mirroring the shipped
# tool_result_max_tokens default. Pinned here rather than read from Settings:
# the service is pure, and the canned results sit far below any sane cap, so
# the value never changes what a sample measures.
_RESULT_MAX_TOKENS = 1000

# Canned tool specs in the exact wire shape the chat service sends across the
# provider seam (tools.offer.tools_to_wire): the OpenAI-function schema, one
# `query` argument. The names, descriptions, and schema MIRROR the live tools
# in app/services/tools/local.py - the description wording is measured to
# matter for whether a small model calls at all - but they are deliberately a
# COPY, not the live Tool classes: the harness must never construct (let alone
# run) a tool that can reach a search backend or the vector store.
_CANNED_TOOL_SPECS: dict[str, dict] = {
    WEB_SEARCH: {
        "type": "function",
        "function": {
            "name": WEB_SEARCH,
            "description": (
                "Search the web for current or external information and read the top "
                "results. Use it for facts that may have changed or that are not in the "
                "conversation or the knowledge base."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to look up, as a natural-language search query.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    SEARCH_KNOWLEDGE: {
        "type": "function",
        "function": {
            "name": SEARCH_KNOWLEDGE,
            "description": (
                "Search this workspace's own stored records and files: its policies, "
                "products, pricing, support and account details, orders, documentation, "
                "and anything uploaded earlier in this conversation. Questions about "
                '"the" company, product, subscription, or policy refer to THIS '
                "workspace's records - always search before saying such information is "
                "unavailable."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to look up, as a natural-language search query.",
                    }
                },
                "required": ["query"],
            },
        },
    },
}

# What a call gets fed back as its result - canned, so a probe is reproducible
# and free of network luck. Shaped like the live tools' numbered-block digests
# so the continuation measures reading a REALISTIC result, not a toy one.
_CANNED_RESULTS: dict[str, str] = {
    WEB_SEARCH: (
        "[1] Python 3.13.5 released\n"
        "https://www.python.org/downloads/\n"
        "Python 3.13.5 is the latest stable release of Python, published on "
        "11 June 2026. It includes security fixes and performance improvements "
        "over 3.13.4."
    ),
    SEARCH_KNOWLEDGE: (
        "[1] Refund policy\n"
        "Orders may be returned for a full refund within 30 days of delivery. "
        "Refunds are issued to the original payment method within 5 business "
        "days of the return being received."
    ),
}

Expectation = Literal[
    "must_call",
    "must_not_call",
    "select",
    "no_invented_call",
    "continuation",
    "loop_guard",
    "thinking_with_tools",
    "think_on",
    "think_off",
]

# Which question a case answers. The groups are scored by different rules and
# read by different consumers - the TOOLS group decides the agency verdict, the
# THINKING group classifies the measured think mode and decides nothing about
# agency - so a reader (and the console) can present them apart instead of as
# one undifferentiated run of samples.
CaseGroup = Literal["tools", "thinking"]

# The markup a serving stack leaks into the VISIBLE answer when reasoning is
# emitted but nothing is parsing it out (the shape ``think:false`` produces on
# weights that reason regardless). Matched case-insensitively against the
# answer, opening tag or closing tag - a stream cut short can carry only one.
_THINK_MARKERS = ("<think>", "</think>")


def _leaks_thinking(answer: str) -> bool:
    """True when reasoning markup reached the visible answer."""
    lowered = answer.lower()
    return any(marker in lowered for marker in _THINK_MARKERS)


@dataclass(frozen=True)
class CaseSpec:
    """One harness case, as data.

    Attributes:
        name: Stable case identifier (keys tallies and the verdict rules).
        message: The user turn the model is asked.
        tools: Names of the canned tools offered this case (subset of the two).
        expectation: Which evaluation rule scores the exchange.
        expected_tool: The tool a call-expecting rule looks for, when it names one.
        think: Whether the case runs with reasoning requested (case h and the
            thinking group).
        group: Which question the case answers - see :data:`CaseGroup`.
    """

    name: str
    message: str
    tools: tuple[str, ...]
    expectation: Expectation
    expected_tool: str | None = None
    think: bool = False
    group: CaseGroup = "tools"


# Case names, so the verdict rules and the fast subset reference cases by
# identity rather than by repeated string literals.
_FRESHNESS = "must-call-freshness"
_WORKSPACE = "must-call-workspace"
_MUST_NOT_CALL = "must-not-call"
_SELECTION = "selection"
_HALLUCINATION_GUARD = "hallucination-guard"
_CONTINUATION = "continuation"
_LOOP_GUARD = "loop-guard"
_THINKING_WITH_TOOLS = "thinking-with-tools"
_THINK_ON = "think-on"
_THINK_OFF = "think-off"

# The one question both thinking cases ask. Deliberately trivial: what is
# measured is which CHANNEL the model answers on, not whether it answers well,
# and a hard question would spend minutes of reasoning to measure the same
# thing. "Answer with just the city name" keeps the visible half short too, so
# a model that reasons at length still finishes.
_THINK_PROBE_MESSAGE = "What is the capital of France? Answer with just the city name."

# The generation cap on a thinking sample, in tokens. It exists because this
# group now runs in the FAST check, which fires unattended after every download:
# an always-thinking tag asked a question with no cap can reason for as long as
# it likes, and a post-pull check that never returns is worse than one that
# measures nothing. Generous enough that the honest case - a short reasoning
# block and one word of answer - never comes near it.
#
# The trade a cap makes is the reason for the rule beside it. Hitting the cap
# looks EXACTLY like the empty-answer failure the case exists to catch: no
# visible content. So a sample that reasoned and then ran out of budget is
# reported as UNMEASURED (an errored sample, classification None) rather than
# failed - the run genuinely did not find out what this tag does, and marking a
# healthy model red for being slow to think would be a worse answer than
# admitting the measurement was not taken.
_THINK_MAX_TOKENS = 512

# The sample-level code for "the cap stopped this, not the model". Distinct
# from any provider code on purpose: the verify job discards a run over
# `provider_unavailable`, and a slow thinker is not a reason to throw away a
# perfectly good agency verdict.
_CAPPED = "generation_capped"

# The full case set. Editing ANY field here changes harness_version() - the
# future model_probes measurement log keys on it, so an edited case correctly
# invalidates every verdict the old wording earned.
CASES: tuple[CaseSpec, ...] = (
    # The THINKING group: the same trivial question asked with the flag set and
    # with it cleared, on the tag under check. Together they say which of
    # never / hybrid / always these WEIGHTS actually are - the fact a stored
    # declaration, the quirk seed and the template inference all only guess at,
    # and the fact a re-pull silently changes.
    #
    # FIRST for three reasons, all of them about running order rather than
    # importance. They are the cheapest pair (no tools, one short generation
    # each), so a fast check shows a reader progress in seconds instead of
    # after a full tool exchange. They measure the flag before the harness has
    # put any tool schema in front of the model. And leaving them here keeps
    # case (h) LAST, which is what makes the verify job's mid-run residency
    # reading demonstrably necessary: (h) drives the pair's thinking tag and
    # evicts the tag being measured, so a reading taken only afterwards would
    # report the wrong model. The lettering below labels the TOOL cases and is
    # unaffected.
    CaseSpec(
        name=_THINK_ON,
        message=_THINK_PROBE_MESSAGE,
        tools=(),
        expectation="think_on",
        think=True,
        group="thinking",
    ),
    CaseSpec(
        name=_THINK_OFF,
        message=_THINK_PROBE_MESSAGE,
        tools=(),
        expectation="think_off",
        think=False,
        group="thinking",
    ),
    # (a) The explicit freshness ask, measured to draw a web_search call from
    # the served dev model (same wording as perf_smoke's tool scenario).
    CaseSpec(
        name=_FRESHNESS,
        message="What is the latest stable version of Python right now?",
        tools=(WEB_SEARCH,),
        expectation="must_call",
        expected_tool=WEB_SEARCH,
    ),
    # (b) The personally-framed workspace-record ask. Measured (dev tier,
    # 2026-07-16) to drop a small model into an "I don't have access" refusal
    # that no strength of prompt wording breaks - whether a model escapes it is
    # exactly the per-model fact this harness exists to record.
    CaseSpec(
        name=_WORKSPACE,
        message="How many days do I have to get a refund on my order?",
        tools=(SEARCH_KNOWLEDGE,),
        expectation="must_call",
        expected_tool=SEARCH_KNOWLEDGE,
    ),
    # (c) The sharpest case: a model that reaches for a tool to answer this
    # will reach for one everywhere.
    CaseSpec(
        name=_MUST_NOT_CALL,
        message="What is 2 + 2?",
        tools=(WEB_SEARCH, SEARCH_KNOWLEDGE),
        expectation="must_not_call",
    ),
    # (d) Both tools on the table, a stored-records-shaped question: the pass
    # is search_knowledge specifically, never web_search.
    CaseSpec(
        name=_SELECTION,
        message="What does our returns policy say about damaged items?",
        tools=(WEB_SEARCH, SEARCH_KNOWLEDGE),
        expectation="select",
        expected_tool=SEARCH_KNOWLEDGE,
    ),
    # (e) A web-shaped question with ONLY the knowledge tool offered. Declining
    # or answering passes; so does trying the offered tool. Inventing a tool
    # that was never offered is the hard fail the case exists to catch.
    CaseSpec(
        name=_HALLUCINATION_GUARD,
        message="What is the latest stable version of Node.js right now?",
        tools=(SEARCH_KNOWLEDGE,),
        expectation="no_invented_call",
    ),
    # (f) The multi-turn continuation - where the published function-calling
    # scores collapse: call, get the canned result fed back, then actually
    # answer from it within the round cap.
    CaseSpec(
        name=_CONTINUATION,
        message="What is the latest stable version of Python right now?",
        tools=(WEB_SEARCH,),
        expectation="continuation",
        expected_tool=WEB_SEARCH,
    ),
    # (g) Same exchange as (f), scored only on stopping: a model still
    # requesting tools after _MAX_ROUNDS passes fails.
    CaseSpec(
        name=_LOOP_GUARD,
        message="What is the latest stable version of Python right now?",
        tools=(WEB_SEARCH,),
        expectation="loop_guard",
        expected_tool=WEB_SEARCH,
    ),
    # (h) Reasoning on AND a tool offered. Passes only with BOTH a well-formed
    # tool call across the exchange AND non-empty answer output -
    # ollama/ollama#10976 fails this silently with EMPTY output (reasoning
    # streams, the answer never arrives), which is exactly what it catches.
    CaseSpec(
        name=_THINKING_WITH_TOOLS,
        message="What is the latest stable version of Python right now?",
        tools=(WEB_SEARCH,),
        expectation="thinking_with_tools",
        expected_tool=WEB_SEARCH,
        think=True,
    ),
)

# The fast subset of the TOOLS group: the two cheapest, most load-bearing
# cases, one sample each. The thinking group joins every depth on its own
# terms (see THINK_CASE_NAMES) and is deliberately absent here, because this
# tuple is also the set the fast agency verdict is scored over - a think case
# says nothing about whether a model can drive a tool loop, and letting one
# blame the agency verdict would clamp a healthy model for reasoning.
FAST_CASE_NAMES: tuple[str, ...] = (_FRESHNESS, _MUST_NOT_CALL)

# The thinking group, which runs at BOTH depths and once per depth. One sample
# each rather than the deep depth's three: a decoder's choice of channel is a
# property of the serving stack's parsing rather than of the sampler, so extra
# samples buy repetition rather than confidence - and the whole point of
# folding this into the fast check is that it stays cheap.
THINK_CASE_NAMES: tuple[str, ...] = (_THINK_ON, _THINK_OFF)

# The deep cases whose majorities decide assisted vs agentic (the guard cases
# clamp to deterministic instead, and case h decides only thinking_with_tools).
_DECIDING_CASE_NAMES: tuple[str, ...] = (
    _FRESHNESS,
    _WORKSPACE,
    _MUST_NOT_CALL,
    _SELECTION,
    _CONTINUATION,
)

# EDIT THESE STRINGS WHENEVER THE RULE THEY NAME CHANGES.
#
# The case specs above are DATA, so editing one moves :func:`harness_version` on
# its own. These two rules live in CODE, where nothing about rewriting them
# moves a hash - and a fingerprint that does not move is a promise that two rows
# measured the same thing. That promise is exactly what the think flag broke:
# the tool cases were once normalized against a committed per-model file alone,
# a tag it did not name was therefore driven with the flag cleared, and
# repairing it changed what every tool case measures without touching a spec.
#
# The strings are never parsed and never compared to behaviour. They are a
# hand-maintained declaration, and the only thing standing between a changed
# rule and a stale verdict that still looks current. Same device, same reason,
# as :data:`app.services.embedder_check._CODE_RULES`.
_CODE_RULES: dict[str, str] = {
    "tool_case_think": (
        "resolved through effective_think_mode, strongest first - measured always, "
        "stored declaration, quirk seed, runtime template - then normalized"
    ),
    "thinking_group_think": "sent literally, on the tag under check, resolved against nothing",
    "thinking_with_tools_case": (
        "passes only on reasoning AND a call AND an answer in one exchange, so a "
        "tag whose resolution omits the flag can never earn it"
    ),
}


def expected_sample_count(*, deep: bool) -> int:
    """Return how many samples a run at this depth will produce.

    Exists so a progress reporter can say "sample 7 of 24" without importing
    the case list or hard-coding a number that a new case would silently make
    wrong - the count is derived from the same specs the run itself reads.

    Args:
        deep: True for the deep depth, False for the fast subset.

    Returns:
        The total number of samples the run will produce.
    """
    think = len(THINK_CASE_NAMES)
    if deep:
        return (len(CASES) - think) * _DEEP_SAMPLES + think
    return len(FAST_CASE_NAMES) + think


def case_named(name: str) -> CaseSpec:
    """Return the case spec with ``name``.

    Args:
        name: A case identifier from :data:`CASES`.

    Returns:
        The matching :class:`CaseSpec`.

    Raises:
        KeyError: ``name`` is not a known case.
    """
    for case in CASES:
        if case.name == name:
            return case
    raise KeyError(name)


def harness_version() -> str:
    """Fingerprint the harness's case specs and the served prompt stack.

    The ``model_probes`` measurement log keys verdicts on this value: editing
    a case's wording, its offered tools, a canned result, the loop bound, the
    persona / tool-guidance layers the harness drives, or one of the
    :data:`_CODE_RULES` changes what a verdict MEASURED, so the version must
    change with it and correctly invalidate what the old cases earned. Computed
    from live module state (never frozen at import), so replacing a case
    demonstrably changes the result.

    Returns:
        A short stable hex digest of every behavior-bearing spec constant.
    """
    payload = json.dumps(
        {
            "cases": [
                {
                    "name": case.name,
                    "message": case.message,
                    "tools": list(case.tools),
                    "expectation": case.expectation,
                    "expected_tool": case.expected_tool,
                    "think": case.think,
                    # Behaviour-bearing, not a label: the group decides which
                    # tag a case drives, how many samples it runs, and whether
                    # its result can clamp the agency verdict.
                    "group": case.group,
                }
                for case in CASES
            ],
            # The spec wording is behavior-bearing too: the descriptions are
            # measured to decide whether a small model calls at all, so an
            # edited description must invalidate old verdicts as well.
            "tool_specs": _CANNED_TOOL_SPECS,
            "canned_results": _CANNED_RESULTS,
            "max_rounds": _MAX_ROUNDS,
            "deep_samples": _DEEP_SAMPLES,
            # The thinking group's generation cap, on the same footing as the
            # loop bound above: it is a limit the harness imposes on what a
            # sample may do, and lowering it turns measurements into
            # "unmeasured". A verdict earned under a different bound answered a
            # different question.
            "think_max_tokens": _THINK_MAX_TOKENS,
            # So is the SHIPPED prompt stack the harness drives: one guidance
            # sentence moved continuation 3/6 -> 6/6 on the same weights
            # (measured 2026-08-06), so a persona or guidance edit changes
            # what a verdict measured and must invalidate old ones the same
            # way a case edit does. Composed with both bullets so every layer
            # (persona, header, both tool bullets, footer, attribution) is in
            # the fingerprint.
            "system_prompt": append_tool_guidance(DEFAULT_SYSTEM_PROMPT, web=True, knowledge=True),
            # The rules that live in code rather than in the specs above - see
            # :data:`_CODE_RULES`. Without this entry the think-flag repair
            # would have changed what every tool case measures while leaving
            # every stored verdict looking current.
            "code_rules": _CODE_RULES,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


# Import-time snapshot for callers that just want the current version; tests
# that monkeypatch CASES must call harness_version() for a live value.
HARNESS_VERSION: str = harness_version()


@dataclass
class CaseSample:
    """One run of one case against one model tag.

    Attributes:
        case: The case name this sample ran.
        model: The exact model tag it ran against.
        passed: Whether the exchange met the case's expectation.
        detail: One human-readable line saying why (pass or fail).
        latency_s: Wall time of the whole exchange, seconds.
        tokens: Answer tokens streamed across the exchange (thinking excluded).
        gen_window_s: Summed per-round first-to-last token spans, for the
            tok/s aggregate. Summed per round rather than across the exchange,
            because the gap between rounds is a fresh prefill - counting it
            would deflate the decode rate.
        gen_intervals: Token intervals inside those spans (per-round tokens
            minus one, summed) - the numerator of the tok/s aggregate.
        error_code: Why this sample measured nothing, else None. Usually the
            provider's own typed code (the CLI uses ``provider_unavailable`` to
            tell infrastructure failure from a measured failure); the harness
            adds :data:`_CAPPED` for a generation that ran out of budget before
            it produced anything, which is likewise a measurement that did not
            happen rather than a model that behaved badly.
        thought: The model produced reasoning during this exchange - on the
            reasoning channel, or as ``<think>`` markup that reached the
            visible answer. Both count, because the question the thinking
            group asks is whether the weights REASONED, and which channel it
            came out on is the separate (and separately reported) failure.
            Observation only: it classifies the measured think mode and has no
            part in whether the sample passed.
        num_ctx: The context window this sample actually ran at, or None when
            the caller sized nothing and the runtime chose. Recorded so a
            verdict NAMES the configuration it measured: the window is the one
            serve-time option a deployment can move without touching the
            weights, and a stored verdict that does not say which window it
            was earned at cannot be read against the turns it certifies.
        think: The think flag the request ASKED for - True, False, or None for
            the flag OMITTED, which is a third request and not a missing field.
            What was requested, never what finally went over the wire: a
            provider that retries without the flag on a model that rejects it
            (the shipped Ollama one does) sends something this field does not
            see. Recorded for the same self-describing reason as
            ``num_ctx``, and for a sharper one: this value is resolved from
            sources that MOVE under a stored row (a declaration confirmed
            later, a re-pull retiring one), so a verdict that did not say which
            flag it was earned with could not be read back at all. Descriptive
            only - nothing reads it into a decision.
    """

    case: str
    model: str
    passed: bool
    detail: str
    latency_s: float
    tokens: int
    gen_window_s: float = 0.0
    gen_intervals: int = 0
    error_code: str | None = None
    thought: bool = False
    num_ctx: int | None = None
    think: bool | None = None


@dataclass
class ToolCheckSummary:
    """What one harness run measured, and the verdicts it earned.

    ``thinking_with_tools`` is False unless a DEEP run earned it - at fast
    depth False means "not earned at this depth", never "measured to fail",
    matching the rule that only an explicit, earned true counts.

    Attributes:
        model: The chat tag the run drove.
        thinking_model: The pair's thinking tag when one exists, else None.
        depth: Which depth ran ("fast" or "deep").
        earned: The agency verdict this run earned (the EARNED key of the
            two-key agency resolution; a human-declared ceiling still clamps
            beside it and a probe can never raise that ceiling).
        thinking_with_tools: True only when a deep run passed case (h) on
            every sample.
        samples: Every case sample, in run order.
        tallies: Per-case (passed, run) counts.
        tokens_per_second: Aggregate decode rate over every sample that
            generated at least two tokens, or None when nothing did.
        measured_think_mode: How this run watched the tag under check treat
            the think flag, or None when the thinking group did not produce a
            usable pair of samples (a provider error on either half). A
            MEASUREMENT of the chat tag's own weights - never of the pair's
            thinking sibling, which the group deliberately does not drive.
        harness_version: :func:`harness_version` at run time.
        digest: None here by design - the harness stays pure, so the CLI
            captures the digest (from the provider's installed-model listing)
            at persist time and binds the verdict to the weights a tag pointed
            at, not the mutable tag. The field exists for wrappers (the Model
            Manager verify job) that carry it alongside the summary.
    """

    model: str
    thinking_model: str | None
    depth: Depth
    earned: AgencyMode
    thinking_with_tools: bool
    samples: list[CaseSample]
    tallies: dict[str, tuple[int, int]]
    tokens_per_second: float | None
    harness_version: str
    measured_think_mode: ThinkMode | None = None
    digest: str | None = None


@dataclass
class _Exchange:
    """Everything one drive of the mini tool loop observed."""

    calls: list[ToolCall] = field(default_factory=list)
    invented: list[str] = field(default_factory=list)
    answer: str = ""
    # Everything that came back on the REASONING channel, across every round.
    # Collected rather than counted because emptiness is the whole signal: an
    # empty reasoning channel with the flag set is what says "never".
    reasoning: str = ""
    answered: bool = False
    # The generation stopped because it ran out of budget, not because the
    # model finished. Only ever set where a cap was actually requested.
    truncated: bool = False
    rounds: int = 0
    tokens: int = 0
    latency_s: float = 0.0
    window_s: float = 0.0
    gen_intervals: int = 0
    cold_load: bool = False
    error_code: str | None = None
    error_detail: str | None = None


async def _aclose_quietly(stream: AsyncIterator[ChatChunk]) -> None:
    """Close a finished provider stream so its connection is released promptly.

    Same shape as the chat service's helper (private there): the mini loop
    opens one stream per round, and a stream left suspended at its terminal
    holds its HTTP connection until the async-gen finalizer runs.
    """
    aclose = getattr(stream, "aclose", None)
    if aclose is not None:
        await aclose()


async def _drive(
    provider: ModelProvider,
    *,
    case: CaseSpec,
    model: str,
    options: GenerationOptions,
    max_rounds: int,
) -> _Exchange:
    """Run one sample's exchange: offer the canned tools, feed canned results.

    Mirrors the live loop's message shapes exactly (assistant message carrying
    ``tool_calls``, then one ``tool``-role message per call with the fenced
    result) so the probe exercises the wire path a real turn takes. Two
    deliberate differences: results are CANNED (never a real tool, never the
    network), and tools stay offered on every round - the live loop strips
    them on its last round to force an answer, but the measurement here is
    whether the MODEL stops chasing on its own, which is the loop-guard case.
    """
    system = append_tool_guidance(
        # The shipped persona plus the guidance layer for the offered set -
        # the prompt a tools-offered turn actually carries, and the layer is
        # measured to decide whether a small model calls at all (0/6 bare vs
        # 6/6 with it; see prompts.py). Probing without it would measure a
        # prompt nobody serves.
        DEFAULT_SYSTEM_PROMPT,
        web=WEB_SEARCH in case.tools,
        knowledge=SEARCH_KNOWLEDGE in case.tools,
    )
    messages = [
        ChatMessage(role="system", content=system),
        ChatMessage(role="user", content=case.message),
    ]
    tools_payload = [_CANNED_TOOL_SPECS[name] for name in case.tools]
    offered = set(case.tools)
    exchange = _Exchange()
    started = time.monotonic()
    truncating = options.max_tokens is not None
    for _ in range(max_rounds):
        exchange.rounds += 1
        requested: list[ToolCall] = []
        round_text: list[str] = []
        # The decode window is measured PER ROUND and summed: the gap between
        # rounds is a fresh prefill, and folding it into a first-to-last span
        # across the exchange would deflate the reported decode rate.
        round_first: float | None = None
        round_last: float | None = None
        stream = provider.chat_stream(messages, model=model, tools=tools_payload, options=options)
        # Thinking chunks are collected but deliberately NOT counted as answer
        # output: the reasoning channel is exactly what ollama#10976 leaves
        # intact while the answer comes back empty, so counting it would hide
        # the failure case (h) exists to catch. Keeping the TEXT is what lets
        # the thinking group tell "it reasoned" from "it did not".
        async for chunk in stream:
            if chunk.type == "token":
                now = time.monotonic()
                round_first = now if round_first is None else round_first
                round_last = now
                exchange.tokens += 1
                round_text.append(chunk.delta)
            elif chunk.type == "thinking":
                exchange.reasoning += chunk.delta
            elif chunk.type == "tool_call" and chunk.tool_call is not None:
                requested.append(chunk.tool_call)
            elif chunk.type == "status" and chunk.status_code == "loading_model":
                exchange.cold_load = True
            elif chunk.type == "error":
                exchange.error_code = chunk.error_code or "error"
                exchange.error_detail = chunk.error or ""
                break
            elif chunk.type == "done":
                # A run that hit its own generation cap did not finish saying
                # what it was going to say, so nothing after this round can be
                # read as the model's choice - see _THINK_MAX_TOKENS.
                exchange.truncated = truncating and chunk.finish_reason == "length"
                break
        await _aclose_quietly(stream)
        if round_first is not None and round_last is not None:
            exchange.window_s += round_last - round_first
            exchange.gen_intervals += len(round_text) - 1
        if exchange.error_code is not None:
            break
        if not requested:
            # No calls this round: its text is the exchange's answer.
            exchange.answer = "".join(round_text)
            exchange.answered = True
            break
        exchange.calls.extend(requested)
        exchange.invented.extend(call.name for call in requested if call.name not in offered)
        messages.append(
            ChatMessage(role="assistant", content="".join(round_text), tool_calls=requested)
        )
        for call in requested:
            # An invented tool gets the live loop's honest "no such tool" text
            # fed back, so the exchange keeps mirroring serve behavior even
            # while the evaluation is already going to fail the sample.
            canned = _CANNED_RESULTS.get(call.name, f"No tool named {call.name!r} is available.")
            fenced = build_tool_result_message(call.name, canned, max_tokens=_RESULT_MAX_TOKENS)
            messages.append(
                ChatMessage(role="tool", name=call.name, tool_call_id=call.id, content=fenced)
            )
    exchange.latency_s = time.monotonic() - started
    return exchange


def _evaluate_think(case: CaseSpec, exchange: _Exchange) -> tuple[bool, str]:
    """Score one thinking-group exchange on the HEALTH of its emission.

    The split that matters, and the reason it is a split: what the thinking
    group MEASURES (never / hybrid / always) is not the same question as
    whether the sample is a pass. Every one of the three modes is a legitimate
    thing for weights to be - the dev tier's own everyday chat model never
    reasons, and its sibling always does - so scoring "it did not reason" as a
    failure would put a permanent red mark on a healthy model and say nothing
    true about it. The mode is CLASSIFIED (see :func:`classify_think_mode`)
    from the same samples, separately.

    What a sample fails on is a broken emission, in the two shapes that are
    always wrong whatever mode the model is in:

    - Reasoning markup in the VISIBLE answer. This is the ``think:false``
      leak: the weights reason regardless, the serving stack was told not to
      parse a reasoning channel, and the raw ``<think>`` block lands in the
      text a user reads. It is the failure the routing clamp exists to prevent.
    - No visible answer at all. Silence is the whole answer here, and a model
      that reasons and then returns nothing is the same silent failure shape
      that case (h) catches on the tool path.
    """
    answer = exchange.answer.strip()
    requested = "think:true" if case.think else "think:false"
    if exchange.truncated and not answer:
        # The cap, not the model. This is indistinguishable from the
        # empty-answer failure by inspection, so it is reported as a
        # measurement that could not be TAKEN rather than one the model failed
        # - and the classifier drops the whole run's mode on the errored
        # sample. Marking a model red for reasoning at length would be a claim
        # about the weights that this run has no basis for.
        return False, (
            f"the generation cap was reached before any visible answer at {requested}, "
            "so what this model does with the flag was not measured"
        )
    if _leaks_thinking(exchange.answer):
        return False, (
            f"reasoning markup leaked into the visible answer at {requested} - "
            "a reader would see the raw <think> block"
        )
    if not answer:
        reasoned = " (it reasoned, then said nothing)" if exchange.reasoning.strip() else ""
        return False, f"produced no visible answer at {requested}{reasoned}"
    if exchange.reasoning.strip():
        return True, f"reasoned on the thinking channel and answered cleanly at {requested}"
    return True, f"answered directly with no reasoning at {requested}"


def _evaluate(case: CaseSpec, exchange: _Exchange) -> tuple[bool, str]:
    """Score one exchange against its case's expectation.

    Returns:
        ``(passed, detail)`` - the detail is one human-readable line.
    """
    if exchange.error_code is not None:
        detail = f"provider error ({exchange.error_code}): {exchange.error_detail or ''}".strip()
        return False, detail
    if case.expectation in ("think_on", "think_off"):
        return _evaluate_think(case, exchange)
    called = [call.name for call in exchange.calls]
    answer = exchange.answer.strip()
    if exchange.invented:
        # An invented tool fails EVERY case, not just the guard case - no
        # expectation tolerates a call to a tool that was never offered. The
        # guard case is merely where the verdict rule keys on it.
        return False, f"called tool(s) never offered: {sorted(set(exchange.invented))}"
    expected = case.expected_tool
    if case.expectation == "must_call":
        if expected in called:
            return True, f"called {expected} ({len(called)} call(s), {exchange.rounds} round(s))"
        if called:
            return False, f"called {called} instead of {expected}"
        return False, f"no tool call was made (expected {expected})"
    if case.expectation == "must_not_call":
        if called:
            return False, f"called {called} for a question that needs no tool"
        if not answer:
            return False, "made no call but also produced no answer"
        return True, f"answered directly with no tool call ({exchange.tokens} tokens)"
    if case.expectation == "select":
        if not called:
            return False, f"no tool call was made (expected {expected})"
        if set(called) == {expected}:
            return True, f"selected {expected} and only {expected}"
        return False, f"called {called}; a stored-records question must use {expected} only"
    if case.expectation == "no_invented_call":
        # Inventions already failed above; declining, answering, or trying the
        # offered tool all pass - but an exchange that produced nothing at all
        # is a broken sample, and the guard fails closed.
        if not answer:
            return False, f"produced no answer within {exchange.rounds} round(s)"
        if called:
            return True, f"used the offered {called[0]} and answered"
        return True, "declined or answered without inventing a tool"
    if case.expectation == "continuation":
        if expected not in called:
            return False, f"no {expected} call, so there was no result to continue from"
        if not exchange.answered or not answer:
            return False, f"result fed back but no final answer within {exchange.rounds} round(s)"
        return True, (
            f"called {expected}, read the result, answered "
            f"({exchange.tokens} tokens, {exchange.rounds} round(s))"
        )
    if case.expectation == "loop_guard":
        if not exchange.answered:
            return False, f"still requesting tools after {exchange.rounds} round(s)"
        if not answer:
            return False, "stopped calling but produced no answer"
        return True, f"stopped and answered within {exchange.rounds} round(s)"
    # thinking_with_tools: all THREE halves must hold in the SAME exchange -
    # reasoning came back, a tool was called, and an answer arrived.
    #
    # The reasoning check is not decoration, and it is checked FIRST because
    # without it the other two are being read out of an exchange that never
    # asked the question. This case is driven at the flag the tag's serve-time
    # resolution gives it, so a tag resolved ``never`` is sent no flag at all;
    # a well-behaved model then calls its tool and answers, and every remaining
    # condition here is satisfied by an exchange in which nothing reasoned. The
    # verdict that fell out said these weights can reason AND call tools, on
    # evidence containing no reasoning - and this is the one axis where the
    # optimistic answer fails silently (ollama/ollama#10976 returns EMPTY
    # output, never an error), so it must never be earned by default.
    #
    # Either channel counts, the same disjunction :attr:`CaseSample.thought`
    # uses: reasoning that LEAKED into the answer is a broken emission (which
    # the thinking group scores on its own terms) but it is still reasoning,
    # and failing it here would blame the wrong fault.
    if not (exchange.reasoning.strip() or _leaks_thinking(exchange.answer)):
        return False, (
            "no reasoning came back on either channel, so this exchange never "
            "exercised reasoning-with-tools - nothing here can earn it"
        )
    if not called:
        return False, "no well-formed tool call while reasoning was enabled"
    if not answer:
        return False, (
            "tool call made but the answer came back empty - the silent "
            "ollama/ollama#10976 failure this case exists to catch"
        )
    return True, (
        f"reasoned, called {called[0]}, and answered ({exchange.tokens} tokens, "
        f"{exchange.rounds} round(s))"
    )


async def run_case(
    provider: ModelProvider,
    case: CaseSpec,
    *,
    model: str,
    options: GenerationOptions | None = None,
    max_rounds: int = _MAX_ROUNDS,
) -> CaseSample:
    """Run one sample of one case and score it.

    Args:
        provider: The model provider to drive.
        case: The case spec to run.
        model: The exact model tag to run against.
        options: Serve-time generation options; defaults to provider defaults.
        max_rounds: The mini loop's round cap (the loop-guard bound).

    Returns:
        The scored :class:`CaseSample`.
    """
    resolved = options or GenerationOptions()
    exchange = await _drive(
        provider,
        case=case,
        model=model,
        options=resolved,
        max_rounds=max_rounds,
    )
    passed, detail = _evaluate(case, exchange)
    if exchange.cold_load:
        # Verdicts are unaffected, but the latency number measured a model
        # load - flag it the way perf_smoke does.
        detail += " [COLD LOAD - run 'make warm' first for a trustworthy latency]"
    return CaseSample(
        case=case.name,
        model=model,
        passed=passed,
        detail=detail,
        latency_s=round(exchange.latency_s, 3),
        tokens=exchange.tokens,
        gen_window_s=exchange.window_s,
        gen_intervals=exchange.gen_intervals,
        # A cap that stopped the generation before any visible content is
        # recorded as a sample that measured NOTHING, so the classifier drops
        # the run's mode rather than reading a truncation as an answer.
        error_code=(
            exchange.error_code
            if exchange.error_code is not None
            else (_CAPPED if exchange.truncated and not exchange.answer.strip() else None)
        ),
        # Either channel counts as having reasoned - see CaseSample.thought.
        thought=bool(exchange.reasoning.strip()) or _leaks_thinking(exchange.answer),
        # Both read off the options the sample was DRIVEN with rather than
        # re-derived from configuration: what was requested is the only thing
        # that says anything about what ran.
        num_ctx=resolved.num_ctx,
        think=resolved.think,
    )


def classify_think_mode(samples: Sequence[CaseSample]) -> ThinkMode | None:
    """Classify how the checked tag treats the think flag, from its samples.

    One boolean per direction, and the table is read in this order:

    ==========================  ==========================  ===========
    reasoned at ``think:false``  reasoned at ``think:true``  measured
    ==========================  ==========================  ===========
    yes                         (either)                    ``always``
    no                          yes                         ``hybrid``
    no                          no                          ``never``
    ==========================  ==========================  ===========

    The off sample is read FIRST and decides on its own, because "it reasoned
    when told not to" is the only observation that cannot be explained any
    other way: the flag was cleared and reasoning happened anyway, so the
    weights do not have an off switch this serving stack can reach. That is
    also the reading a hybrid model can never produce, which is what makes the
    routing clamp built on it safe in one direction only.

    ``never`` is the honest reading of the retry path too. A model whose
    serving stack rejects the flag gets the provider's retry-without-think, so
    the ``think:true`` sample comes back with an empty reasoning channel - no
    error, and exactly the observation "these weights do not reason".

    Args:
        samples: Every sample of a run; the thinking group's are picked out
            here rather than by the caller, so no caller has to know which
            case names belong to the group.

    Returns:
        The measured mode, or None when the run cannot support a
        classification: a missing half (a fast run of an older harness, a
        truncated run) or a provider error on either half. None means "not
        measured" and must never be read as any of the three modes.
    """
    by_name = {name: [s for s in samples if s.case == name] for name in THINK_CASE_NAMES}
    if not all(by_name[name] for name in THINK_CASE_NAMES):
        return None
    if any(sample.error_code is not None for runs in by_name.values() for sample in runs):
        # A sample that never reached the model observed the network. Reading
        # its empty reasoning channel as "never" would record a fabricated
        # measurement, and this one is a routing input.
        return None
    if any(sample.thought for sample in by_name[_THINK_OFF]):
        return "always"
    if any(sample.thought for sample in by_name[_THINK_ON]):
        return "hybrid"
    return "never"


def _target_for(case: CaseSpec, *, chat: str, thinking: str | None) -> str:
    """Return the tag ``case`` drives on a run over this pair.

    The routing rule in one place, because two things now need it: the loop
    that runs a case, and the window resolution that has to happen once per
    TARGET before any sample does. A second copy would be a probe sized for one
    tag and run against another.

    Case (h) routes to the pair's thinking sibling exactly as a reasoning turn
    does. The THINKING group never does, whichever way its flag is set: it
    measures the tag under check, and routing its ``think:true`` half to an
    always-thinking sibling would answer about the sibling's weights.

    Args:
        case: The case about to run.
        chat: The chat tag under check.
        thinking: The pair's thinking tag, or None on a single-model tier.

    Returns:
        The exact tag this case will be driven against.
    """
    if case.group == "thinking":
        return chat
    return (thinking or chat) if case.think else chat


async def _serve_time_think_mode(
    provider: ModelProvider,
    provider_name: str,
    model: str,
    *,
    fetch_policy: PolicyFetcher | None,
    fetch_measured_think: MeasuredThinkFetcher | None,
) -> ThinkMode:
    """Resolve how a TURN on ``model`` would treat the think flag.

    The one composition point, so a terminal run and a UI click cannot disagree
    about which flag a case was driven with - the same reason the verify job
    keeps the probe's ORDER in one module. It assembles the chain rather than
    restating it: :func:`app.services.model_facts.effective_think_mode` owns the
    ranking, and a second ranking written here is exactly the drift that would
    put the harness back where this repair found it.

    METADATA ONLY. The report is the shared short-TTL capability probe, taken
    through the shared short-TTL cache; the two injected reads hit the
    database and never the model server. Nothing here loads weights, which is
    what lets it run in the window between the card being freed and the first
    sample.

    Args:
        provider: The model provider serving this deployment.
        provider_name: Its configured name, for the facts cache key.
        model: The exact tag this run will drive.
        fetch_policy: The stored-declaration read, or None for a caller with no
            session - see :data:`PolicyFetcher`.
        fetch_measured_think: The measured-think-mode read, or None for the
            same reason - see :data:`MeasuredThinkFetcher`.

    Returns:
        The resolved :data:`~app.services.model_resolution.ThinkMode`.
    """
    report = await report_for(provider, provider_name, model)
    policy = await fetch_policy(model) if fetch_policy is not None else None
    measured = await fetch_measured_think(model) if fetch_measured_think is not None else None
    return effective_think_mode(model, report, policy=policy, measured=measured)


def clamping_cases(depth: str, tallies: Mapping[str, tuple[int, int]]) -> tuple[str, ...]:
    """Name the cases in ``tallies`` that held the agency verdict down.

    The scoring rule, expressed as "which cases are to blame" rather than as
    "what did they add up to" - and it is the SAME rule, because
    :func:`_earned_verdict` is written on top of this function. That matters:
    a reader-facing explanation built from a second copy of these thresholds
    would drift, and the drift would take the form of a card naming a case
    that actually passed.

    Two thresholds, and they are not the same:

    - The guard cases (hallucination-guard, loop-guard) blame on ANY lost
      sample. Inventing a tool and chasing tools forever are the two actively
      dangerous behaviors, so one observed instance is enough.
    - The deciding cases are scored by per-case MAJORITY at deep depth, so
      ``2/3`` is a PASS and blames nothing, while ``1/3`` blames. A fast run
      has one sample per case and no majority to take, so any loss blames.

    A case with no samples at all counts as blame (the run never demonstrated
    it), which is what makes a truncated run fail closed. A caller rendering
    tallies should intersect this with the cases it actually HAS, since a
    missing case has no numbers to show.

    The THINKING group is in neither scored set and can never blame. Which
    channel a model emits reasoning on says nothing about whether it can drive
    a tool loop, so letting a leaked ``<think>`` block clamp the agency verdict
    would hold a capable model at the safe floor for an unrelated fault - and
    the mismatch it does mean is reported on its own terms elsewhere.

    Args:
        depth: The depth the run was measured at - it changes the threshold.
            Typed ``str`` rather than :data:`Depth` because a reporting caller
            reads it off a stored row, where it is whatever was written; any
            value but ``"fast"`` is scored as a deep run.
        tallies: Per-case ``(passed, run)`` counts.

    Returns:
        The blamed case names, guards first. Empty when nothing held the
        verdict down, which is exactly when the run earned its depth's cap.
    """
    blamed: list[str] = []
    for name in (_HALLUCINATION_GUARD, _LOOP_GUARD):
        passed, run = tallies.get(name, (0, 0))
        if passed < run:
            blamed.append(name)
    scored = FAST_CASE_NAMES if depth == "fast" else _DECIDING_CASE_NAMES
    for name in scored:
        passed, run = tallies.get(name, (0, 0))
        # Any loss at fast (one sample, no majority to take); a lost majority
        # at deep. Never both - the two name sets are read per depth.
        lost = passed < run if depth == "fast" else passed * 2 <= run
        if (run == 0 or lost) and name not in blamed:
            blamed.append(name)
    return tuple(blamed)


def _earned_verdict(depth: Depth, tallies: dict[str, tuple[int, int]]) -> AgencyMode:
    """Resolve the agency verdict a run earned from its per-case tallies.

    The rules, in order: any hallucination-guard or loop-guard sample failure
    earns ``deterministic`` - inventing tools and chasing forever are the two
    actively dangerous behaviors, so one observed instance is enough. A fast
    run earns ``assisted`` only when every sample passed, else
    ``deterministic`` (conservative by design: a coarse two-sample screen that
    could not even pass itself has demonstrated nothing, and a human-declared
    declaration remains the day-one grace path). A deep run earns
    ``agentic`` when every deciding case - both must-call framings,
    must-not-call, selection, and continuation - passes by per-case majority,
    else ``assisted`` (safe for any model: one that never calls a tool in
    assisted behaves exactly as the pre-tool pipeline).

    Written on top of :func:`clamping_cases` so the verdict and the
    explanation a reader is shown can never disagree about which case was to
    blame.
    """
    blamed = clamping_cases(depth, tallies)
    if any(name in blamed for name in (_HALLUCINATION_GUARD, _LOOP_GUARD)):
        return "deterministic"
    if depth == "fast":
        return "deterministic" if blamed else "assisted"
    return "assisted" if blamed else "agentic"


def _thinking_with_tools_verdict(depth: Depth, tallies: dict[str, tuple[int, int]]) -> bool:
    """True only when a DEEP run passed case (h) on every sample.

    Guessing optimistically here fails silently (empty output, no error), so
    a fast run - which never exercises the case - always reports False.
    """
    if depth != "deep":
        return False
    passed, run = tallies.get(_THINKING_WITH_TOOLS, (0, 0))
    return run > 0 and passed == run


def _aggregate_rate(samples: list[CaseSample]) -> float | None:
    """Tokens/second over every sample that generated a measurable window.

    Same timing method as scripts/perf_smoke.py - token intervals over the
    first-to-last token span - applied per generation round (a between-rounds
    gap is a prefill, not decode) and summed across samples so long exchanges
    weigh proportionally. None when nothing generated a measurable window.
    """
    generated = [(s.gen_intervals, s.gen_window_s) for s in samples if s.gen_intervals > 0]
    total_window = sum(window for _, window in generated)
    if total_window <= 0:
        return None
    return sum(intervals for intervals, _ in generated) / total_window


async def run_tool_check(
    provider: ModelProvider,
    *,
    provider_name: str,
    default_window: int,
    deep: bool = False,
    chat_model: str | None = None,
    thinking_model: str | None = None,
    samples_per_case: int | None = None,
    context_ceiling: int | None = None,
    fetch_policy: PolicyFetcher | None = None,
    fetch_measured_think: MeasuredThinkFetcher | None = None,
    on_sample: Callable[[CaseSample], Awaitable[None] | None] | None = None,
) -> ToolCheckSummary:
    """Run the harness at one depth and return the measured summary.

    Args:
        provider: The model provider to drive (the real one, or a fake).
        provider_name: The ``MODEL_PROVIDER`` selector value, for the runtime
            facts cache both resolutions read. Passed rather than inferred so a
            probe shares the cache the turn path already fills instead of
            opening its own metadata round trip for the same tag.
        default_window: The deployment's ``DEFAULT_CONTEXT_WINDOW`` - what one
            turn on a tag NOTHING sizes may spend here. An argument for the
            same reason ``context_ceiling`` is: this service reads no settings
            of its own, and a probe run at a window the deployment's turns do
            not use certifies somebody else's configuration.
        deep: False runs the FAST subset (one sample each, earns at most
            ``assisted``); True runs every case, three samples each. The
            thinking group runs at either depth, once each way.
        chat_model: The tag to check. Required.
        thinking_model: The pair sibling the reasoning case drives, when
            there is one; pass "" or None to force single-model behavior (case
            h then runs on the chat tag with the think flag on).
        samples_per_case: Testing affordance overriding the depth's sample
            count for the TOOL cases; the depth's verdict cap is unaffected,
            and the thinking group keeps its one sample per direction.
        context_ceiling: The workspace's stored context ceiling, when the
            caller has read it. None probes unclamped - correct for a caller
            with no tenant to read for, and the same window such a deployment's
            turns request. See the module docstring for why this is an argument
            and why a later change to it disturbs nothing already measured.
        fetch_policy: Reads the stored operator declaration for one tag. None
            resolves the think flag without that rung, which is the answer for
            a caller holding no database session - see :data:`PolicyFetcher`
            and the module docstring on the sessionless half.
        fetch_measured_think: Reads what a previous check measured about one
            tag's think flag. None is the same degradation, and the same
            explicit one - see :data:`MeasuredThinkFetcher`.
        on_sample: Called after each sample completes (CLI progress). May be
            async - an awaitable return value is awaited before the next
            sample starts, so a caller can observe the machine at a moment it
            controls rather than only report.

    Returns:
        The :class:`ToolCheckSummary` with per-sample results and verdicts.

    Raises:
        ValueError: No chat model was given.
    """
    chat = chat_model
    thinking = thinking_model
    thinking = thinking or None
    if not chat:
        raise ValueError("No chat model to check: chat_model is required.")
    depth: Depth = "deep" if deep else "fast"
    count = samples_per_case if samples_per_case is not None else (_DEEP_SAMPLES if deep else 1)
    cases = [
        case for case in CASES if deep or case.name in FAST_CASE_NAMES or case.group == "thinking"
    ]
    # ONCE per target, before any sample runs, and BOTH serve-time options
    # together. Resolving a tag asks the runtime for its metadata
    # and (for the think flag) the database for two rows; doing either per
    # sample would put round trips in front of every generation of a run that
    # already costs GPU minutes - for answers that cannot change between two
    # samples of the same tag. A run touches two tags at most (the chat tag
    # and, on a pair, its thinking sibling), so this is two resolutions of each
    # at the very worst.
    #
    # Per TARGET rather than per run: an env override can put an undeclared
    # chat tag beside a declared thinking sibling, and one shared answer would
    # then size and flag case (h) for the wrong model.
    windows: dict[str, int] = {}
    think_modes: dict[str, ThinkMode] = {}
    for target in sorted({_target_for(case, chat=chat, thinking=thinking) for case in cases}):
        windows[target] = await resolve_num_ctx(
            provider,
            provider_name,
            target,
            default_window=default_window,
            ceiling=context_ceiling,
        )
        think_modes[target] = await _serve_time_think_mode(
            provider,
            provider_name,
            target,
            fetch_policy=fetch_policy,
            fetch_measured_think=fetch_measured_think,
        )
    samples: list[CaseSample] = []
    for case in cases:
        # Case (h) routes to the pair's thinking tag exactly as a reasoning
        # turn does; everything else runs on the chat tag (see _target_for).
        # The think flag is normalized per target against the mode the chat
        # path resolves for it (an always-thinking tag gets True, a
        # never-thinking instruct model gets None so the flag is omitted from
        # the request) - so a tool case is driven with the flag a real turn on
        # that tag would carry, whether or not anything describes it.
        #
        # The THINKING group opts out of both, and the module docstring says
        # why: it drives the tag under check (routing its think:true half to an
        # always-thinking sibling would answer about the sibling's weights),
        # and it sends the flag literally (resolving it would measure the
        # sources this group exists to test, itself among them).
        think_case = case.group == "thinking"
        target = _target_for(case, chat=chat, thinking=thinking)
        options = GenerationOptions(
            think=case.think if think_case else normalize_think(case.think, think_modes[target]),
            num_ctx=windows[target],
            # Only the thinking group is capped. A tool case has to be free to
            # run its exchange to the end, and its answer length is bounded by
            # the question anyway; a thinking case asks a trivial question of a
            # model that may reason without limit, unattended, after a pull.
            max_tokens=_THINK_MAX_TOKENS if think_case else None,
        )
        # One sample per depth for the thinking group; see THINK_CASE_NAMES.
        # The samples_per_case testing affordance is honoured for the tool
        # cases alone, so a test that thins a deep run does not accidentally
        # widen this one.
        for _ in range(1 if think_case else count):
            sample = await run_case(provider, case, model=target, options=options)
            samples.append(sample)
            if on_sample is not None:
                # Awaited when the hook returns an awaitable, so a caller can do
                # real work here rather than only print: the verify job takes a
                # live residency reading, which has to happen while THIS
                # sample's model is still the resident one. A plain sync
                # callback (the CLI's) returns None and is unaffected.
                progress = on_sample(sample)
                if inspect.isawaitable(progress):
                    await progress
    tallies: dict[str, tuple[int, int]] = {}
    for case in cases:
        runs = [sample for sample in samples if sample.case == case.name]
        tallies[case.name] = (sum(1 for sample in runs if sample.passed), len(runs))
    return ToolCheckSummary(
        model=chat,
        thinking_model=thinking,
        depth=depth,
        earned=_earned_verdict(depth, tallies),
        thinking_with_tools=_thinking_with_tools_verdict(depth, tallies),
        samples=samples,
        tallies=tallies,
        # The TOOL samples only. The thinking pair answers a trivial question
        # in a word or two, so each contributes an interval or two over a
        # sub-millisecond window - and this number is read as "how fast does
        # this deployment decode", which is a statement about the exchanges
        # that do real work. Window-weighting would keep the distortion small
        # either way; excluding them keeps the number measuring one thing.
        tokens_per_second=_aggregate_rate(
            [sample for sample in samples if sample.case not in THINK_CASE_NAMES]
        ),
        harness_version=harness_version(),
        measured_think_mode=classify_think_mode(samples),
        # Digest deliberately unset: the CLI captures the digest at persist time (the harness stays pure)
        # (a tag is mutable, so a verdict must bind to the weights it saw).
        digest=None,
    )
