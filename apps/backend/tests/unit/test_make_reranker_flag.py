"""Guard the one signal that decides whether a reranker exists at all.

``RERANKER_URL`` is read in two places that must never disagree: the backend
reads it to decide whether retrieval re-scores, and the Makefile reads it to
decide whether the sidecar container is started. Disagreement is silent in
both directions - a URL with no container behind it makes every turn reach for
a service that is not there, and a container nothing calls burns memory doing
nothing - which is exactly the state the old hardware-profile keying produced.

Make is asked the question rather than the file being pattern-matched: the
answer depends on conditional evaluation order and on command-line variables
beating ``-include .env``, and only make can be trusted about either. The
Makefile is not shipped inside the backend image, and make is not installed
there, so both absences skip rather than fail.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


def _find_makefile() -> Path | None:
    """Walk up for the repo's Makefile, rather than counting directories."""
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "Makefile"
        if candidate.is_file():
            return candidate
    return None


_MAKEFILE = _find_makefile()
_MAKE = shutil.which("make")

pytestmark = pytest.mark.skipif(
    _MAKEFILE is None or _MAKE is None,
    reason="no Makefile above this checkout, or no make on PATH",
)


def reranker_flag(**variables: str) -> str:
    """Ask make what ``RERANKER_FLAG`` evaluates to under these variables.

    Passed on the COMMAND LINE deliberately: the Makefile pulls the developer's
    own ``.env`` in with ``-include``, and only a command-line assignment
    outranks that - so the answer describes the arguments rather than the
    machine the test happens to run on. That is also why every case below
    states ``RERANKER_URL`` explicitly, empty where it means unset.

    Args:
        **variables: Make variables to assign on the command line.

    Returns:
        The expanded value of ``RERANKER_FLAG``, stripped.
    """
    assert _MAKEFILE is not None and _MAKE is not None  # guarded by the skip
    completed = subprocess.run(
        [
            _MAKE,
            "-f",
            str(_MAKEFILE),
            "--eval=__probe: ; @echo $(RERANKER_FLAG)",
            "__probe",
            *(f"{name}={value}" for name, value in variables.items()),
        ],
        capture_output=True,
        text=True,
        cwd=_MAKEFILE.parent,
        check=True,
    )
    return completed.stdout.strip()


def test_the_sidecar_starts_exactly_when_a_url_names_one() -> None:
    """The compose profile follows RERANKER_URL, and nothing else.

    Both halves matter, and both were live bugs while a hardware-profile name
    keyed this: a URL on a small profile started no container while the backend
    was told a reranker existed, and a large profile with no URL started one
    that nothing would ever call.
    """
    assert reranker_flag(RERANKER_URL="http://reranker:7997") == "--profile reranker"
    assert reranker_flag(RERANKER_URL="") == ""


def test_the_explicit_override_still_outranks_the_url() -> None:
    """RERANKER=0/1 remains the escape hatch, above the key it overrides."""
    assert reranker_flag(RERANKER="0", RERANKER_URL="http://reranker:7997") == ""
    assert reranker_flag(RERANKER="1", RERANKER_URL="") == "--profile reranker"


def test_nothing_else_in_the_environment_starts_the_sidecar() -> None:
    """Only the two keys above decide, so a stale MODEL_TIER line changes nothing.

    An upgraded checkout keeps whatever its ``.env`` already held, and
    ``MODEL_TIER`` is the value that used to key this rule outright. It is an
    inert make variable now; a rule that still read it would start a sidecar
    (or withhold one) on the strength of a line nothing else in the repo
    consults.
    """
    assert reranker_flag(RERANKER_URL="", MODEL_TIER="server-a10") == ""
    assert reranker_flag(RERANKER_URL="http://reranker:7997", MODEL_TIER="dev") == (
        "--profile reranker"
    )
