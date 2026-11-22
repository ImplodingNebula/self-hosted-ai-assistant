"""Guard the two steps that make ``make deploy``'s cheap path a real deploy.

The cheap path exists because the development layout bind-mounts the
application source: no image input moved, so nothing is rebuilt and the
containers already hold this working tree's files. Holding the files is not the
same as serving them, and two gaps sit exactly there.

Migrations run from the backend container's start command and nowhere else, so
a pull that adds one leaves the new code on the old schema until something
restarts that container - and where a reloader picks host edits up by itself,
the new code starts serving against the old schema straight away. Separately,
file-change notifications do not cross every bind mount (they do not cross the
WSL2 one at all), so the reloading backend and the web UI's dev watcher can be
serving the code they started with while the edited files sit beside them.

Both failures are silent: the deploy reports success, the stack is up, and the
wrong thing is running. Nothing else in the suite exercises this recipe -
deleting either line would break no test and no boot - so it is pinned here the
same way ``test_remote_guard.py`` pins the remote guard, by reading the repo's
own Makefile. The ORDER is pinned too: the upgrade has to precede the restart,
or it races the start command's own upgrade in a freshly recreated container.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

# The three services the cheap path restarts. Kept here rather than inline so
# the compose cross-check below and the recipe assertions cannot drift apart.
_APP_SERVICES = ("backend", "worker", "frontend")


def _repo_root() -> Path | None:
    """Walk up for the repo root, identified by its Makefile.

    Counting parents only holds when the tests run from the repo; inside the
    backend image the backend directory IS the filesystem root.
    """
    for parent in Path(__file__).resolve().parents:
        if (parent / "Makefile").is_file():
            return parent
    return None


_ROOT = _repo_root()

pytestmark = pytest.mark.skipif(_ROOT is None, reason="no Makefile above this checkout")


def _recipe(makefile: str, target: str) -> str:
    """Return the recipe body of one make target (its tab-indented lines)."""
    body: list[str] = []
    collecting = False
    for line in makefile.splitlines():
        if collecting:
            if line.startswith("\t"):
                body.append(line)
                continue
            if line.strip() == "" or line.lstrip().startswith("#"):
                continue
            break
        if re.match(rf"^{re.escape(target)}\s*:", line):
            collecting = True
    return "\n".join(body)


def _cheap_path() -> str:
    """Return the branch `make deploy` takes when no image input moved.

    Anchored on the line the branch announces itself with, so the rebuild
    branch above it (which only delegates to ``redeploy``) cannot satisfy any
    assertion below by accident.
    """
    assert _ROOT is not None  # guarded by the skip
    recipe = _recipe((_ROOT / "Makefile").read_text(encoding="utf-8"), "deploy")
    start = recipe.find("Skipping the build")
    assert start != -1, "make deploy no longer announces its no-rebuild branch"
    return recipe[start:]


def test_the_cheap_path_applies_migrations() -> None:
    """A deploy that skips the build must still move the schema.

    Without this, a pull carrying a migration deploys code that queries columns
    the database does not have - and reports success while doing it, because
    every step it did run worked.
    """
    assert "db-upgrade" in _cheap_path()


def test_the_cheap_path_restarts_the_application_services() -> None:
    """Delivering files is not deploying them; the processes have to restart.

    Named explicitly rather than restarting everything: the model server holds
    minutes of cold load and none of these three holds a model.
    """
    cheap = _cheap_path()
    assert re.search(r"restart\s+" + r"\s+".join(_APP_SERVICES), cheap), (
        "the cheap path no longer restarts backend, worker and frontend by name"
    )


def test_migrations_run_before_the_restart_and_both_before_the_stamp() -> None:
    """Order is the guard, not decoration.

    The upgrade runs in the container that is ALREADY up, before anything
    restarts, so it cannot race the start command's own upgrade in a
    just-recreated container. Both precede the stamp because the stamp is the
    claim that this commit is deployed, and it must not be written by a run
    that stopped short of deploying it.
    """
    cheap = _cheap_path()
    upgrade = cheap.find("db-upgrade")
    restart = cheap.find("restart backend")
    stamp = cheap.find("deploy-stamp")
    assert upgrade != -1 and restart != -1 and stamp != -1
    assert upgrade < restart, "migrations must be applied before the services restart"
    assert restart < stamp, "the deploy stamp must be written after the restart, not before"


def test_the_restarted_services_are_services_compose_defines() -> None:
    """A renamed service must break here, not at 2am on somebody's deploy.

    ``docker compose restart`` on a name nothing defines fails loudly, which
    would at least stop the chain - but it stops it after the migrations have
    run and before the stamp, leaving a deploy half-applied for a reason that
    has nothing to do with the change being deployed.
    """
    assert _ROOT is not None  # guarded by the skip
    compose = yaml.safe_load((_ROOT / "infra" / "docker-compose.yml").read_text(encoding="utf-8"))
    defined = set(compose["services"])
    assert set(_APP_SERVICES) <= defined
