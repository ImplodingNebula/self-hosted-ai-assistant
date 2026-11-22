"""DTOs for the liveness and readiness probes."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

DependencyState = Literal["ok", "unreachable", "unconfigured"]
"""How one backing service answered its probe.

``unconfigured`` is a first-class, non-failing answer: several dependencies are
optional by design (the reranker, the search sidecar), and a deployment that
runs without them is a normal deployment, not a broken one.
"""


class DependencyCheck(BaseModel):
    """The result of probing one backing service."""

    name: str
    # Whether a failure here makes the whole deployment unready. Reported so a
    # reader can see WHY a probe that found a dead dependency still says ready.
    required: bool
    state: DependencyState
    # How long the probe itself took, including a timeout that expired.
    latency_ms: float
    # A short, human-readable reason when the state is not "ok". Never carries
    # an exception's text: a connection error can quote a URL, and a URL can
    # carry a credential in its userinfo.
    detail: str | None = None


class ReadinessReport(BaseModel):
    """Whether this deployment can serve requests, and what each dependency said.

    ``ready`` is decided by the REQUIRED checks alone. The informational ones
    are published because they change what the app can do - a missing reranker
    silently degrades retrieval to dense-only, and a model server that has gone
    away answers no chat turns - but they never vote, because a probe that
    failed them would take a recoverable deployment out of a load balancer.
    """

    status: Literal["ready", "unready"]
    checks: list[DependencyCheck]
