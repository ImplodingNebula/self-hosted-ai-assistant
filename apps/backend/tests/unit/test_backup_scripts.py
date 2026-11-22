"""Guard the guards inside the backup and restore scripts.

``scripts/backup.sh`` and ``scripts/restore.sh`` are shell, so nothing in the
test suite imports them and nothing else exercises them - a broken refusal
would break no test and no boot, and would only be discovered during a restore,
which is the worst moment there is. Each script therefore exposes its rules as
subcommands that take their inputs as arguments (the same shape as
``scripts/remote_check.sh classify`` and ``scripts/gpu_check.sh classify``), and
these tests drive those subcommands directly.

The rules pinned here are the ones whose failure is silent:

- ``safe-name`` decides whether a name may be used as a path component. Both
  scripts carry it - backup validates what Qdrant reports before writing a file
  for it, restore validates what it reads back off that filename before placing
  the file inside a container - so both the rule and their agreement are pinned.
- ``collection-names`` is the parser deciding which collections get backed up.
  A parser that finds nothing produces a backup that reports success and holds
  no vectors at all.
- ``network-name`` is the Docker network every Qdrant request is made from.
  Both scripts reach Qdrant through a helper container joined to the stack's
  network rather than through a published port, so a wrong name means nothing
  resolves - and the conventional ``<project>_default`` IS wrong here.
- ``restore.sh collections`` is the guard at its call site: the list a restore
  would actually recover, which must come from the manifest (whose checksums
  were verified) and never from whatever ``*.snapshot`` files happen to be in
  the directory.
- ``restore.sh classify-running`` decides whether the application is down.
  Without it a restore drops a database that open connections are still using,
  and unpacks an archive underneath a worker that is writing into it.
- ``restore.sh classify-head`` decides whether the dump is ahead of the code.
  Without it, a restore that will make the stack fail to boot looks exactly
  like one that will not.
- ``restore.sh verify`` is the gate in front of every destructive step, so it
  is tested for what it accepts, what it rejects, and what it insists on: the
  restore replaces the database before it ever opens the archive directory, so
  a backup missing its dump has to be refused here or not at all.

Running under ``sh`` also means CI runs them under ``dash``, so a bashism that
happens to work on a development machine fails here.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest


def _repo_root() -> Path | None:
    """Walk up for the repo root, identified by the Makefile + scripts pair.

    Mirrors test_remote_guard.py: counting parents breaks inside the backend
    image, where the backend directory is the filesystem root.
    """
    for parent in Path(__file__).resolve().parents:
        if (parent / "Makefile").is_file() and (parent / "scripts" / "backup.sh").is_file():
            return parent
    return None


_ROOT = _repo_root()

pytestmark = pytest.mark.skipif(
    _ROOT is None or shutil.which("sh") is None,
    reason="needs the repo checkout and a POSIX sh",
)


def _run(script: str, *args: str, stdin: str = "") -> str:
    """Run one script subcommand and return its stdout, stripped.

    ``stdin`` carries anything quote-heavy. Windows rebuilds the command line
    the Python runtime produced and the shell re-parses it, consuming quote
    characters on the way - so a JSON document passed as an argument arrives
    mangled, and the subcommands that take one read it here instead.
    """
    assert _ROOT is not None
    result = subprocess.run(  # noqa: S603  - fixed argv, no shell
        ["sh", str(_ROOT / "scripts" / script), *args],
        capture_output=True,
        text=True,
        cwd=_ROOT,
        input=stdin,
        check=False,
    )
    assert result.returncode == 0, f"{script} {args} exited {result.returncode}: {result.stderr}"
    return result.stdout.strip()


# --- backup.sh safe-name -----------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "kb_chunks__qwen3-embedding_0_6b",
        "session_chunks__qwen3-embedding_0_6b",
        "a",
        "with.dots",
    ],
)
def test_safe_name_accepts_ordinary_collection_names(name: str) -> None:
    """The names this stack actually uses must pass, or a backup refuses itself."""
    assert _run("backup.sh", "safe-name", name) == "ok"


@pytest.mark.parametrize(
    "name",
    [
        "../etc/passwd",  # climbs out of the backup directory
        "a/b",  # writes into a subdirectory that was never created
        "..",  # the parent directory itself
        r"a\b",  # the same climb spelled for Windows
        "-rf",  # read as an option by whatever command receives it
        "",  # empty: composes to a bare ".snapshot"
        "with space",  # splits into two names further down
        "semi;colon",  # a command separator
        "dollar$var",  # expands rather than being copied
        "tick`cmd",  # substitutes a command
        "pipe|char",  # redirects
        "star*glob",  # expands to whatever happens to be nearby
    ],
)
def test_safe_name_rejects_anything_that_is_not_a_plain_name(name: str) -> None:
    """Every one of these is interpolated into a path or a command line.

    The check is an allowlist rather than a list of forbidden characters, so
    this parametrisation is a sample of the shapes that matter and not the
    definition - a new metacharacter is refused without anyone adding a case.

    Quote characters are deliberately absent from the sample even though the
    script rejects them: on Windows the shell re-parses the command line the
    Python runtime built, and a quote is consumed before the script ever sees
    it - so the case would test the argument-passing layer rather than the
    guard, and would fail on one host for a reason that is not a bug.
    """
    assert _run("backup.sh", "safe-name", name) == "unsafe"


_SAFE_NAME_SAMPLES = [
    "kb_chunks__qwen3-embedding_0_6b",
    "a",
    "with.dots",
    "../etc/passwd",
    "a/b",
    "..",
    r"a\b",
    "-rf",
    "",
    "with space",
    "semi;colon",
    "dollar$var",
    "tick`cmd",
    "pipe|char",
    "star*glob",
]


@pytest.mark.parametrize(
    "body,expected",
    [
        ('{"result":{"collections":[]},"status":"ok"}', []),
        ('{"result":{"collections":[{"name":"kb_chunks__x"}]},"status":"ok"}', ["kb_chunks__x"]),
        (
            '{"result":{"collections":[{"name":"kb__a"},{"name":"session__b"}]},"status":"ok"}',
            ["kb__a", "session__b"],
        ),
    ],
)
def test_collection_names_parses_a_qdrant_listing(body: str, expected: list[str]) -> None:
    """The parser that decides which collections get backed up.

    Finding nothing where there is something is the expensive direction: the
    backup then completes, reports success, and contains no vectors. Both
    scripts carry this parser, and the pairing test below keeps them identical.
    """
    assert _run("backup.sh", "collection-names", stdin=body).split() == expected


@pytest.mark.parametrize(
    "body",
    [
        '{"result":{"collections":[]},"status":"ok"}',
        '{"result":{"collections":[{"name":"kb__a"},{"name":"session__b"}]},"status":"ok"}',
    ],
)
def test_both_scripts_parse_a_listing_identically(body: str) -> None:
    """restore.sh uses its own copy to name the collections a restore left alone.

    A disagreement here would report live collections as missing from the
    backup, or fail to report ones that really are.
    """
    assert _run("backup.sh", "collection-names", stdin=body) == _run(
        "restore.sh", "collection-names", stdin=body
    )


def test_network_name_is_read_from_the_compose_file() -> None:
    """The stack's network is NOT named by Compose's default convention.

    Every Qdrant request in both scripts runs in a helper container joined to
    this network, so the name has to be right or nothing resolves. Compose
    calls a project's default network ``<project>_default`` unless the file
    renames it - and infra/docker-compose.yml does rename it. Code that
    assembles the conventional form finds no such network and reports a running
    stack as unreachable.
    """
    assert _ROOT is not None
    compose = (_ROOT / "infra" / "docker-compose.yml").read_text(encoding="utf-8")
    match = re.search(r"^networks:\s*\n\s+default:\s*\n\s+name:\s*(\S+)", compose, re.MULTILINE)
    assert match is not None, "compose no longer declares a default network name"
    declared = match.group(1)

    derived = _run("backup.sh", "network-name")
    assert derived == declared
    assert derived != "ai-chatbot-platform_default"


def test_both_scripts_derive_the_same_network() -> None:
    """A disagreement would send one of them to a network Qdrant is not on."""
    assert _run("backup.sh", "network-name") == _run("restore.sh", "network-name")


@pytest.mark.parametrize("name", _SAFE_NAME_SAMPLES)
def test_both_scripts_agree_on_what_is_a_safe_name(name: str) -> None:
    """backup.sh and restore.sh carry the same allowlist and must not drift.

    They check it at opposite ends of the same journey: backup validates the
    name Qdrant reports before writing a file for it, restore validates the
    name it reads back off that filename before placing the file inside the
    container. A backup directory gets copied between machines and renamed by
    hand in between, so the second check is not redundant - but if the two
    lists diverge, one script writes names the other refuses to restore, and
    that is only discovered during a recovery.
    """
    assert _run("backup.sh", "safe-name", name) == _run("restore.sh", "safe-name", name)


# --- restore.sh classify-running ---------------------------------------------


def test_running_app_services_are_reported() -> None:
    """backend, worker and frontend all block a restore, and each is named.

    Naming them matters as much as refusing: the operator has to stop exactly
    these and leave the stores alone, and a refusal that does not say which
    containers it means invites a `make down` that stops the stores too.
    """
    offenders = _run(
        "restore.sh",
        "classify-running",
        "ai-chatbot-platform",
        "ai-chatbot-platform-backend-1",
        "ai-chatbot-platform-worker-1",
        "ai-chatbot-platform-frontend-1",
    ).splitlines()
    assert offenders == [
        "ai-chatbot-platform-backend-1",
        "ai-chatbot-platform-worker-1",
        "ai-chatbot-platform-frontend-1",
    ]


def test_stores_do_not_block_a_restore() -> None:
    """postgres and qdrant must stay up - the restore writes THROUGH them.

    If they were treated as application services the refusal would be
    unsatisfiable: stopping them makes the restore impossible.
    """
    assert (
        _run(
            "restore.sh",
            "classify-running",
            "ai-chatbot-platform",
            "ai-chatbot-platform-postgres-1",
            "ai-chatbot-platform-qdrant-1",
            "ai-chatbot-platform-redis-1",
            "ai-chatbot-platform-searxng-1",
            "ai-chatbot-platform-mcp-chassis-1",
        )
        == ""
    )


def test_another_projects_backend_does_not_block_a_restore() -> None:
    """Matching is on Compose's <project>-<service>-<replica> shape, not a substring.

    A substring test says yes to any container with "backend" in its name, and
    an unrelated project on the same host would then make this stack's restore
    permanently impossible.
    """
    assert (
        _run(
            "restore.sh",
            "classify-running",
            "ai-chatbot-platform",
            "otherapp-backend-1",
            "ai-chatbot-platform-backend-sidecar-1",
        )
        == ""
    )


def test_no_containers_at_all_is_a_pass() -> None:
    """A fresh host has nothing running, and that is the ideal restore target."""
    assert _run("restore.sh", "classify-running", "ai-chatbot-platform") == ""


# --- restore.sh classify-head ------------------------------------------------


def _current_head_candidates() -> list[str]:
    """Every revision id declared by a migration in this checkout."""
    assert _ROOT is not None
    ids: list[str] = []
    for path in (_ROOT / "apps" / "backend" / "alembic" / "versions").glob("*.py"):
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("revision:") or stripped.startswith("revision ="):
                ids.append(stripped.split("=", 1)[1].strip().strip("\"'"))
                break
    return ids


def test_a_revision_this_checkout_has_is_known() -> None:
    """A dump at a revision we hold heals forward on boot - no warning is due."""
    candidates = _current_head_candidates()
    assert candidates, "no alembic revisions found; the fixture for this test is gone"
    assert _run("restore.sh", "classify-head", candidates[0]) == "known"


def test_a_revision_this_checkout_lacks_is_unknown() -> None:
    """This is the failure the runbook exists for.

    A dump from newer code leaves the database at a revision alembic cannot
    step forward from, and the backend exits at boot with "Can't locate
    revision". Reporting it as `known` would let a restore proceed silently
    into a stack that cannot start.
    """
    assert _run("restore.sh", "classify-head", "deadbeefface") == "unknown"


def test_an_unreadable_head_is_unchecked_rather_than_unknown(tmp_path: Path) -> None:
    """No verdict is its own answer, and deliberately not the alarming one.

    "unknown" means "the dump is ahead of this code", which is a real problem
    with a real fix. A missing migrations directory or a manifest with no head
    recorded is neither evidence of that nor evidence against it, and calling
    it "unknown" would send an operator chasing a version mismatch that is not
    there.
    """
    assert _run("restore.sh", "classify-head", "") == "unchecked"
    assert _run("restore.sh", "classify-head", "abc123", str(tmp_path)) == "unchecked"


# --- restore.sh verify -------------------------------------------------------


def _fake_backup(
    root: Path,
    newline: str = "\n",
    corrupt: bool = False,
    omit: str | None = None,
    stray: str | None = None,
    captured: str | None = None,
) -> Path:
    """Write a minimal backup directory that ``restore.sh verify`` can read.

    Real artifacts are a database dump, a tar and a Qdrant snapshot; verify
    only hashes them, so a few bytes each stands in exactly.

    ``omit`` drops one artifact from both the directory and the manifest, which
    is what a copy that lost a file looks like. ``stray`` adds a snapshot FILE
    that the manifest never mentions, which is what a hand-merged directory
    looks like. ``captured`` writes the ``attachments_captured`` field; left
    unset the field is absent entirely, which is what a manifest written before
    that field existed looks like.
    """
    import hashlib

    out = root / "20260101-000000"
    (out / "qdrant").mkdir(parents=True)
    artifacts = {
        "postgres.dump": b"stand-in for a dump",
        "attachments.tar": b"stand-in for a tar",
        "qdrant/kb_chunks__x.snapshot": b"stand-in for a snapshot",
    }
    if omit is not None:
        del artifacts[omit]
    if captured == "no":
        # A backup taken under STORAGE_BACKEND=s3 holds no archive at all.
        artifacts.pop("attachments.tar", None)
    lines = [
        "# manifest",
        "created_utc: 2026-01-01T00:00:00Z",
        "git_rev: abc123",
        "alembic_head: deadbeefface",
    ]
    if captured is not None:
        lines.append(f"storage_backend: {'local' if captured == 'yes' else 's3'}")
        lines.append(f"attachments_captured: {captured}")
    lines += [
        "",
        "# artifact <path> <bytes> <sha256>",
    ]
    for name, payload in artifacts.items():
        (out / name).write_bytes(payload)
        lines.append(f"artifact {name} {len(payload)} {hashlib.sha256(payload).hexdigest()}")
    if corrupt:
        # Same length, different bytes: only the checksum can catch this.
        (out / "postgres.dump").write_bytes(b"STAND-IN FOR A DUMP")
    if stray is not None:
        (out / "qdrant" / stray).write_bytes(b"unlisted, unverified")
    (out / "manifest.txt").write_bytes(newline.join(lines).encode() + newline.encode())
    return out


def _restore(*args: str) -> subprocess.CompletedProcess[str]:
    assert _ROOT is not None
    return subprocess.run(  # noqa: S603  - fixed argv, no shell
        ["sh", str(_ROOT / "scripts" / "restore.sh"), *args],
        capture_output=True,
        text=True,
        cwd=_ROOT,
        check=False,
    )


def _verify(directory: Path) -> subprocess.CompletedProcess[str]:
    return _restore("verify", str(directory))


def test_verify_accepts_an_intact_backup(tmp_path: Path) -> None:
    """The happy path, and the reason `verify` exists without Docker.

    Re-checking a backup on a shelf should not need a running stack, so this
    subcommand reads files and nothing else.
    """
    result = _verify(_fake_backup(tmp_path))
    assert result.returncode == 0, result.stderr
    assert "3 artifact(s) verified" in result.stdout


def test_verify_rejects_a_corrupted_artifact(tmp_path: Path) -> None:
    """A same-length change is invisible to every check except the checksum.

    Bit rot and a truncated transfer both produce a file that still looks
    plausible; without this the damage is discovered by pg_restore, mid-restore,
    with the live database already dropped.
    """
    result = _verify(_fake_backup(tmp_path, corrupt=True))
    assert result.returncode != 0
    assert "does not match its manifest checksum" in result.stderr


def test_verify_reads_a_manifest_with_windows_line_endings(tmp_path: Path) -> None:
    """A CRLF manifest must verify, not fail with two identical hashes.

    A backup directory gets copied between machines and sometimes opened in an
    editor on the way. Without the carriage return stripped off the last field,
    every checksum comparison fails while PRINTING the expected and actual
    hashes as the same string - the least debuggable error the script has.
    """
    result = _verify(_fake_backup(tmp_path, newline="\r\n"))
    assert result.returncode == 0, result.stderr
    assert "3 artifact(s) verified" in result.stdout


def test_verify_refuses_a_directory_that_is_not_a_backup(tmp_path: Path) -> None:
    """A path with no manifest is a mistyped argument, not an empty backup."""
    result = _verify(tmp_path)
    assert result.returncode != 0
    assert "manifest" in result.stderr


@pytest.mark.parametrize("missing", ["postgres.dump", "attachments.tar"])
def test_verify_refuses_a_backup_missing_a_required_artifact(tmp_path: Path, missing: str) -> None:
    """The artifact whose ABSENCE is destructive must be refused up front.

    Verify otherwise checks only what the manifest happens to list, so a
    directory that lost its dump in a copy verifies perfectly - and the restore
    then drops the live database, replaces the vectors, and discovers there is
    nothing to write back. There is no rollback from that point, which is why
    the check has to be here rather than at the step that needs the file.
    """
    result = _verify(_fake_backup(tmp_path, omit=missing))
    assert result.returncode != 0
    assert missing in result.stderr


# --- object storage: what a backup can and cannot cover ----------------------


@pytest.mark.parametrize(
    "backend,expected",
    [
        ("local", "yes"),
        ("s3", "no"),
        # Not a valid backend since the S3 adapter landed - the app fails at
        # boot on it - but an .env can still carry it, and the safe reading of
        # a misconfiguration is "the bytes are not in that volume".
        ("minio", "no"),
        ("", "no"),
        ("LOCAL", "no"),
    ],
)
def test_storage_backend_decides_whether_attachments_are_capturable(
    backend: str, expected: str
) -> None:
    """The attachment tar is a copy of a volume only the local adapter writes.

    Under `s3` the uploaded bytes are objects in a bucket and that volume is
    empty or stale, so tarring it yields an archive that looks like the
    uploaded files and contains none of them - the worst possible artifact,
    because it restores cleanly.
    """
    assert _run("backup.sh", "storage-captures-attachments", backend) == expected


@pytest.mark.parametrize("backend", ["local", "s3", "minio", ""])
def test_both_scripts_agree_on_storage_capture(backend: str) -> None:
    """Backup decides what to write; restore decides what to unpack.

    If they disagreed, restore would empty the attachment volume for an archive
    backup never took.
    """
    assert _run("backup.sh", "storage-captures-attachments", backend) == _run(
        "restore.sh", "storage-captures-attachments", backend
    )


def test_restore_skips_attachments_when_the_backup_captured_none(tmp_path: Path) -> None:
    """The step that runs here EMPTIES the volume before unpacking.

    Against an s3-era backup there is nothing to unpack, so running it would
    delete whatever is in that volume and write nothing back - destroying files
    on a deployment that has since been switched to local storage. Skipping is
    the guard; there is no safe no-op version of "empty it first".
    """
    out = _fake_backup(tmp_path, captured="no")
    assert _restore("attachments-plan", str(out)).stdout.strip() == "skip"


def test_restore_restores_attachments_when_the_backup_captured_them(tmp_path: Path) -> None:
    """The ordinary local-storage case still restores."""
    out = _fake_backup(tmp_path, captured="yes")
    assert _restore("attachments-plan", str(out)).stdout.strip() == "restore"


def test_a_manifest_without_the_field_is_treated_as_captured(tmp_path: Path) -> None:
    """Backups written before the field existed always tarred the volume.

    Defaulting a missing field to "no" would silently stop restoring files that
    are sitting right there in the archive.
    """
    out = _fake_backup(tmp_path)
    assert (out / "attachments.tar").exists()
    assert _restore("attachments-plan", str(out)).stdout.strip() == "restore"


def test_verify_accepts_a_backup_that_captured_no_attachments(tmp_path: Path) -> None:
    """An s3-era backup has no attachments.tar, and that is not a fault.

    The required-artifact check has to read the manifest rather than demand the
    file unconditionally, or every backup taken under object storage would be
    unrestorable.
    """
    result = _verify(_fake_backup(tmp_path, captured="no"))
    assert result.returncode == 0, result.stderr


# --- the compose layout a printed command names -------------------------------


def test_compose_files_default_to_the_development_layout() -> None:
    """With no DEPLOY_ENV set, the printed commands name the dev overlay.

    A fresh checkout has no .env at all, and dev is what `make` assumes there
    too - so the two agree about what an unconfigured machine is.
    """
    assert _run("restore.sh", "compose-files") == (
        "-f infra/docker-compose.yml -f infra/docker-compose.override.yml"
    )


def test_the_makefile_declares_the_same_two_layouts() -> None:
    """restore.sh only PRINTS compose commands, so it has to print the right ones.

    The development overlay bind-mounts source and publishes ports. Pasting it
    on a deployment recreates those containers in the wrong layout - the exact
    accident DEPLOY_ENV exists to prevent - so a restore that tells an operator
    to run it would undo that protection at the worst moment. This pins the two
    file pairs against the Makefile's own, which is the definition.
    """
    assert _ROOT is not None
    makefile = (_ROOT / "Makefile").read_text(encoding="utf-8")
    for layout in ("override", "prod"):
        expected = f"-f infra/docker-compose.yml -f infra/docker-compose.{layout}.yml"
        assert expected in makefile, f"the Makefile no longer declares the {layout} pair"
        assert expected in (_ROOT / "scripts" / "restore.sh").read_text(encoding="utf-8")


# --- restore.sh collections (the guard at its call site) ---------------------


def test_collections_come_from_the_manifest_not_the_directory(tmp_path: Path) -> None:
    """A snapshot the manifest never listed must not be recovered.

    Its checksum was never verified, so recovering it overwrites a live
    collection on the strength of a filename. Two backup directories merged by
    hand, or an aborted sync, produce exactly this.
    """
    out = _fake_backup(tmp_path, stray="kb_chunks__stale.snapshot")
    result = _restore("collections", str(out))
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["kb_chunks__x"]


def test_collections_refuses_a_hostile_name_from_the_manifest(tmp_path: Path) -> None:
    """The name guard is pinned where it is USED, not only as a pure function.

    A manifest is a text file that travels with the backup. This drives the
    same path a restore takes, so deleting the gate inside the loop - rather
    than the function it calls - is caught.

    The name carries no space on purpose: manifest fields are whitespace
    separated, so a name with one in it lands in a different column and never
    reaches the guard at all. That is safe, and it is a different code path -
    testing it here would pass while proving nothing about this one.
    """
    out = _fake_backup(tmp_path)
    manifest = out / "manifest.txt"
    manifest.write_text(
        manifest.read_text(encoding="utf-8")
        + "artifact qdrant/..;rm-rf.snapshot 5 "
        + "0" * 64
        + "\n",
        encoding="utf-8",
    )
    result = _restore("collections", str(out))
    assert result.returncode != 0
    assert "not a name this can use" in result.stderr


# --- Makefile wiring ---------------------------------------------------------


def test_make_targets_call_the_scripts() -> None:
    """The make targets are the only callers, so the wiring is part of the guard.

    A target that stopped invoking its script would break no test above - both
    scripts would still classify perfectly, for nobody.
    """
    assert _ROOT is not None
    makefile = (_ROOT / "Makefile").read_text(encoding="utf-8")
    assert "scripts/backup.sh" in makefile
    assert "scripts/restore.sh" in makefile
    assert "scripts/host_disk.sh" in makefile


def test_restore_target_requires_an_explicit_directory() -> None:
    """`make restore` must not guess which backup to use.

    A restore replaces the database wholesale; defaulting to "the newest one"
    turns a typo into a silent data swap.
    """
    assert _ROOT is not None
    if shutil.which("make") is None:
        pytest.skip("make is not available on this host")
    result = subprocess.run(  # noqa: S603  - fixed argv, no shell
        ["make", "restore"],
        capture_output=True,
        text=True,
        cwd=_ROOT,
        check=False,
    )
    assert result.returncode != 0
    assert "DIR" in result.stdout + result.stderr
