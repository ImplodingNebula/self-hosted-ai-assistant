"""Measure chat-turn latency against a running stack.

The point is to show that a change did not quietly make the assistant slower.
Run it with `make perf`, which pipes this file into the backend container where
httpx and the service network are already available. It drives the real SSE
endpoint exactly as a browser would and reports, per scenario:

  ttft    time to the first answer token - what a person actually feels
  total   request start to the `done` event
  tok/s   tokens per second over the generation window (first token -> done)

Only the plain turn runs by default. PERF_THINK=1 adds a thinking turn,
PERF_WEB=1 adds a web-search turn, and PERF_TOOL=1 adds a tool-calling turn;
all are opt-in because none is left warm by `make warm`, so their first
reading measures a model load or a search engine rather than the change under
test.

The tool turn asks a question measured to make the served model call
`web_search`, so it times the whole agentic loop: a first generation that emits
the call, the live search and page fetches, and the second generation that
streams the answer. Its notes column reports how many tool calls ran - and
flags the reading when NONE did (the model declined, or an AGENCY_MODE clamp
stripped the tools), because that number measured a plain turn wearing the
tool scenario's name.

PERF_AGENCY=<auto|agentic|assisted|deterministic> clamps every scenario in the
run to one agency mode, which is what makes the three-way control affordable: a
reading on the base branch, one on the change, and one on the change clamped to
`deterministic`. That third leg reproduces the pre-tool path, so landing it
within noise of the first is the evidence that the policy layer itself costs
nothing. Sending the clamp per request keeps all three legs in one session -
without it the deterministic leg means editing AGENCY_MODE in `.env` and
restarting the container, because compose bakes a container's environment in at
start, and a restart throws away the warm model the reading depends on.

The clamp only ever clamps DOWN. `deterministic` therefore always lands, while
`agentic` cannot lift a turn above what AGENCY_MODE and the model's own ceiling
already allowed - so the mode this script reports is the one it REQUESTED, and
the turn ran at that mode or lower. A typo in PERF_AGENCY is fatal rather than
ignored: falling back to "auto" would hand back an unclamped reading that then
gets filed as the deterministic leg, which is the exact false control the flag
exists to prevent.

A cold model load is reported separately, because a `loading_model` status can
add minutes on a small GPU and would otherwise be mistaken for a regression.
Prime the stack with `make warm` first for numbers that reflect steady state.

Two honest caveats about the numbers. `total` (and therefore the tok/s window)
ends at `done`, which lands after citations are streamed and the turn is
persisted, so a grounded or web turn reports a slightly pessimistic rate. And a
turn that reasons emits its reasoning on a separate channel, so its `ttft` marks
the first *answer* token, well after the thinking finished.

The script reports; it does not gate. Absolute numbers are meaningless across
machines - what matters is the delta against the same machine before a change.
Setting PERF_MAX_TTFT / PERF_MAX_TOTAL (seconds) turns a scenario red and exits
non-zero. Those budgets apply to every scenario that ran, so a budget set for
the plain turn will flag the slower thinking and web turns; size it for the
slowest scenario enabled, or gate the plain turn on its own.

Each scenario starts a fresh conversation, and every conversation it created is
deleted before the script exits so repeated runs leave no clutter behind.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field, replace

import httpx


def _env_float(name: str, default: float | None = None) -> float | None:
    """Read a float from the environment, treating unset AND empty as absent.

    `make` forwards an unset variable as an empty string (`-e PERF_MAX_TTFT=""`),
    so an empty value must mean "no budget", not a parse error.
    """
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"warning: ignoring {name}={raw!r} (not a number)", file=sys.stderr)
        return default


def _env_agency(name: str) -> str:
    """Read the agency clamp from the environment, treating unset AND empty as "auto".

    `make` forwards an unset variable as an empty string, so empty must mean "no
    clamp" for the same reason `_env_float` reads it as "no budget".

    Unlike a bad budget, a bad mode is fatal. A budget that fails to parse only
    fails to gate, but a clamp that silently fell back to "auto" would return an
    unclamped reading that the operator files as the deterministic leg - the
    false control this flag exists to make cheap. Better to stop than to hand
    back a number that means something other than its label.
    """
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return "auto"
    if raw not in _AGENCY_MODES:
        valid = ", ".join(sorted(_AGENCY_MODES))
        print(
            f"error: {name}={raw!r} is not an agency mode (expected one of: {valid})",
            file=sys.stderr,
        )
        sys.exit(2)
    return raw


BASE_URL = os.environ.get("PERF_BASE_URL", "").strip() or "http://localhost:8000"
TIMEOUT = _env_float("PERF_TIMEOUT", 600.0)
_TRUTHY = {"1", "true", "yes"}
# Mirrors ChatRequest.agency's Literal in app/schemas/chat.py. Validating here
# turns a typo into a named error instead of a 422 from the API.
_AGENCY_MODES = {"auto", "agentic", "assisted", "deterministic"}
INCLUDE_WEB = os.environ.get("PERF_WEB", "").strip().lower() in _TRUTHY
INCLUDE_THINK = os.environ.get("PERF_THINK", "").strip().lower() in _TRUTHY
INCLUDE_TOOL = os.environ.get("PERF_TOOL", "").strip().lower() in _TRUTHY
AGENCY = _env_agency("PERF_AGENCY")
MAX_TTFT = _env_float("PERF_MAX_TTFT")
MAX_TOTAL = _env_float("PERF_MAX_TOTAL")


@dataclass
class Scenario:
    """One timed chat turn."""

    name: str
    message: str
    thinking: str = "off"
    web_search: str = "off"
    agency: str = "auto"


@dataclass
class Result:
    """What one scenario measured."""

    name: str
    agency: str = "auto"
    ttft: float | None = None
    total: float | None = None
    tokens: int = 0
    statuses: list[str] = field(default_factory=list)
    citations: int = 0
    error: str | None = None
    conversation_id: str | None = None

    @property
    def gen_rate(self) -> float | None:
        """Tokens per second over the generation window, or None if it never generated."""
        if self.ttft is None or self.total is None or self.tokens < 2:
            return None
        window = self.total - self.ttft
        return (self.tokens - 1) / window if window > 0 else None

    @property
    def cold_load(self) -> bool:
        """True when the turn paid for loading a model into memory."""
        return "loading_model" in self.statuses

    @property
    def tool_calls(self) -> int:
        """How many tool executions this turn ran (one status per call)."""
        return self.statuses.count("tool_running")


# The plain turn is the only scenario that runs by default, because it is the
# only one `make warm` leaves warm. On tiers that pair-route (a separate model
# answers thinking turns) a thinking scenario reloads a second model, and a web
# turn reaches the network - both would swamp the signal with a cold load or an
# engine timeout, and a number nobody trusts is a number nobody runs.
# The prompt asks for a long, mechanical answer on purpose. A short reply ("say
# hi") finishes in two or three token events, and tokens/second computed over
# that window measures request overhead rather than decode speed. Counting to 50
# generates enough tokens for the rate to mean something, and needs no knowledge
# the model might lack.
SCENARIOS = [
    Scenario("simple", "Count from 1 to 50, separated by spaces. Output only the numbers.")
]
if INCLUDE_THINK:
    SCENARIOS.append(Scenario("thinking", "What is 17 times 23? Think it through.", thinking="on"))
if INCLUDE_WEB:
    SCENARIOS.append(
        Scenario("web", "What are the latest developments in the news today?", web_search="on")
    )
if INCLUDE_TOOL:
    # A question measured to draw a web_search call from the served model (an
    # explicit "latest ... right now" freshness ask), so the reading times the
    # agentic loop end to end. web_search stays "auto": the tool offer needs
    # the switch not-off, and "auto" is what a real turn carries.
    SCENARIOS.append(
        Scenario(
            "tool",
            "What is the latest stable version of Python right now?",
            web_search="auto",
        )
    )

# One clamp covers the whole run: a control leg compares entire tables, and a
# table that mixed legs would be unreadable. Rewriting the scenarios (rather
# than reading the global inside run()) keeps Scenario the only place a mode
# lives, so run() stays a pure function of the scenario it is handed and each
# Result can carry the mode it was actually taken under.
if AGENCY != "auto":
    SCENARIOS = [replace(scenario, agency=AGENCY) for scenario in SCENARIOS]


def run(client: httpx.Client, scenario: Scenario) -> Result:
    """Drive one chat turn end to end, timing the events that matter.

    The backend emits exactly one ``data:`` line per event, so each ``data:``
    line completes the frame its preceding ``event:`` line opened. Parsing that
    way avoids depending on blank separator lines surviving the HTTP client's
    line iterator.
    """
    result = Result(scenario.name, agency=scenario.agency)
    body = {
        "message": scenario.message,
        "history": [],
        "thinking": scenario.thinking,
        "web_search": scenario.web_search,
    }
    # "auto" is ChatRequest's default and resolves to "no clamp", so sending it
    # would change nothing - but omitting it keeps an unclamped run's request
    # byte-identical to the readings taken before this flag existed, which is
    # what makes those older numbers still comparable.
    if scenario.agency != "auto":
        body["agency"] = scenario.agency
    started = time.monotonic()
    event: str | None = None
    try:
        with client.stream("POST", "/api/v1/chat/stream", json=body) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line or line.startswith(":"):  # blank separator or heartbeat
                    continue
                if line.startswith("event:"):
                    event = line[len("event:") :].strip()
                    continue
                if not line.startswith("data:"):
                    continue
                try:
                    data = json.loads(line[len("data:") :].strip())
                except json.JSONDecodeError:
                    data = {}
                if event == "token":
                    if result.ttft is None:
                        result.ttft = time.monotonic() - started
                    result.tokens += 1
                elif event == "status":
                    result.statuses.append(str(data.get("code", "")))
                elif event == "citation":
                    result.citations += 1
                elif event == "error":
                    result.error = f"{data.get('code')}: {data.get('message')}"
                elif event == "done":
                    result.total = time.monotonic() - started
                    result.conversation_id = data.get("conversation_id")
                    break
                event = None
    except (httpx.HTTPError, OSError) as exc:  # unreachable stack, timeout, transport
        result.error = f"{type(exc).__name__}: {exc}"
    if result.total is None and result.error is None:
        result.error = "stream ended without a done event"
    return result


def _fmt(value: float | None, suffix: str = "s") -> str:
    return "-" if value is None else f"{value:.2f}{suffix}"


def cleanup(client: httpx.Client, results: list[Result]) -> None:
    """Delete the throwaway conversations the run created.

    Measuring against the real endpoint necessarily persists a turn. Removing it
    afterwards keeps a benchmark from filling the conversation sidebar, and takes
    the session's ephemeral vectors with it. A failed cleanup is reported, never
    fatal - the measurement already succeeded.
    """
    for result in results:
        if not result.conversation_id:
            continue
        try:
            response = client.delete(f"/api/v1/conversations/{result.conversation_id}")
            response.raise_for_status()
        except httpx.HTTPError as exc:
            print(f"warning: could not delete conversation {result.conversation_id}: {exc}")


def main() -> int:
    """Time every scenario, print a table, and fail if a latency budget is exceeded."""
    print(f"target: {BASE_URL}")
    # Name the leg on EVERY run, not just a clamped one. The three-way control
    # is three pastes of this output compared by hand, and an unlabelled table
    # is one waiting to be read as the wrong leg - a deterministic reading
    # mistaken for an agentic one would "prove" the policy layer is free.
    # Report what was REQUESTED: the resolver only clamps down, and the mode it
    # settled on is internal, so the honest claim is "this mode or lower".
    agency_note = "" if AGENCY == "auto" else " (requested clamp; ran at this mode or lower)"
    print(f"agency: {AGENCY}{agency_note}")
    if MAX_TTFT or MAX_TOTAL:
        print(f"budget: ttft<={_fmt(MAX_TTFT)} total<={_fmt(MAX_TOTAL)}")
    print()

    results: list[Result] = []
    with httpx.Client(base_url=BASE_URL, timeout=TIMEOUT) as client:
        for scenario in SCENARIOS:
            print(f"running {scenario.name}...", flush=True)
            results.append(run(client, scenario))
        cleanup(client, results)

    print()
    header = f"{'scenario':<10} {'ttft':>8} {'total':>8} {'tokens':>7} {'tok/s':>7}  notes"
    print(header)
    print("-" * len(header))
    over_budget = False
    failed = False
    for result in results:
        notes: list[str] = []
        if result.cold_load:
            notes.append("COLD LOAD (run 'make warm' and retry)")
        if result.name == "tool":
            if result.tool_calls:
                notes.append(f"{result.tool_calls} tool calls")
            elif result.agency == "deterministic":
                # The control leg offers no tool at all, so no call is the leg
                # working - naming the clamp beats implying the model declined.
                # Only `deterministic` earns this: `assisted` still OFFERS
                # tools, so a silent turn there is a real declination.
                notes.append("no tool call (deterministic: none offered - this is the control leg)")
            elif result.error is None:
                # Without a call the number is a plain turn wearing the tool
                # scenario's name - flag it rather than let it pass as a
                # (false) improvement.
                notes.append("NO TOOL CALL (model declined or agency clamped; not a tool reading)")
        if result.citations:
            notes.append(f"{result.citations} citations")
        if result.error:
            notes.append(f"ERROR {result.error}")
            failed = True
        if not result.cold_load:
            if MAX_TTFT and result.ttft and result.ttft > MAX_TTFT:
                notes.append(f"ttft over budget ({_fmt(MAX_TTFT)})")
                over_budget = True
            if MAX_TOTAL and result.total and result.total > MAX_TOTAL:
                notes.append(f"total over budget ({_fmt(MAX_TOTAL)})")
                over_budget = True
        rate = result.gen_rate
        print(
            f"{result.name:<10} {_fmt(result.ttft):>8} {_fmt(result.total):>8} "
            f"{result.tokens:>7} {'-' if rate is None else f'{rate:.1f}':>7}  {'; '.join(notes)}"
        )

    print()
    print("Compare against the same machine before the change; absolute values are hardware-bound.")
    return 1 if (failed or over_budget) else 0


if __name__ == "__main__":
    sys.exit(main())
