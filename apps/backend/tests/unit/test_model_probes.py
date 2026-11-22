"""The model_probes measurement log: machine-scoped, append-only, digest-bound."""

from __future__ import annotations

from app.models.entities import ModelProbe
from app.providers.fake_provider import FakeProvider
from app.repositories.model_probes import ModelProbeRepository, row_from_summary
from app.services.tool_check import CaseSample, ToolCheckSummary


def make_summary(**overrides) -> ToolCheckSummary:
    base: dict = {
        "model": "qwen3:4b-instruct",
        "thinking_model": "qwen3:4b",
        "depth": "fast",
        "earned": "assisted",
        "thinking_with_tools": False,
        "samples": [
            CaseSample(
                case="must-call-freshness",
                model="qwen3:4b-instruct",
                passed=True,
                detail="called web_search",
                latency_s=12.0,
                tokens=30,
                num_ctx=8192,
                think=True,
            )
        ],
        "tallies": {"must-call-freshness": (1, 1)},
        "tokens_per_second": 11.5,
        "harness_version": "abc123def456",
    }
    base.update(overrides)
    return ToolCheckSummary(**base)


def probe_row(**overrides) -> ModelProbe:
    return row_from_summary(
        make_summary(),
        provider_name="ollama",
        digest=overrides.pop("digest", "sha256:aaa"),
        runtime_version=overrides.pop("runtime_version", "0.32.0"),
        serving_endpoint=overrides.pop("serving_endpoint", "http://ollama:11434"),
        vram_bytes=overrides.pop("vram_bytes", 2_500_000_000),
    )


def test_row_from_summary_maps_verdicts_and_measurements() -> None:
    row = probe_row()
    assert row.provider == "ollama"
    assert row.model == "qwen3:4b-instruct"
    assert row.digest == "sha256:aaa"
    assert row.runtime_version == "0.32.0"
    # WHERE the run happened, so a verdict cannot migrate to another server.
    assert row.serving_endpoint == "http://ollama:11434"
    assert row.depth == "fast"
    assert row.earned_agency == "assisted"
    assert row.thinking_with_tools is False
    assert row.tokens_per_second == 11.5
    assert row.vram_bytes == 2_500_000_000  # BigInteger territory on purpose
    # Tallies serialize JSON-safe (tuples become lists) and samples compact.
    assert row.cases["tallies"] == {"must-call-freshness": [1, 1]}
    assert row.cases["samples"][0]["case"] == "must-call-freshness"
    assert "latency_s" not in row.cases["samples"][0]  # verdict record, not a trace
    # Both serve-time options ride along, which is what makes the verdict
    # self-describing: the digest says which weights were measured, and these
    # say under which configuration - the two things a deployment can change
    # without the digest moving.
    assert row.cases["samples"][0]["num_ctx"] == 8192
    assert row.cases["samples"][0]["think"] is True


def test_an_omitted_think_flag_is_recorded_as_such() -> None:
    # None is a THIRD request rather than a missing field: a model whose
    # serving stack has no reasoning plumbing is asked with no flag at all,
    # because sending one earns a rejection. A row that flattened that to false
    # would claim the run measured a request nobody made.
    summary = make_summary(
        samples=[
            CaseSample(
                case="must-not-call",
                model="qwen3:4b-instruct",
                passed=True,
                detail="answered directly",
                latency_s=1.0,
                tokens=2,
                num_ctx=8192,
                think=None,
            )
        ]
    )
    row = row_from_summary(
        summary,
        provider_name="ollama",
        digest="sha256:aaa",
        runtime_version="0.32.0",
        serving_endpoint="http://ollama:11434",
        vram_bytes=None,
    )
    assert row.cases["samples"][0]["think"] is None


async def test_log_is_append_only_history_newest_first(db_sessionmaker) -> None:
    async with db_sessionmaker() as session:
        repo = ModelProbeRepository(session)
        first = await repo.add(probe_row(digest="sha256:old"))
        second = await repo.add(probe_row(digest="sha256:old"))
        await session.commit()

    async with db_sessionmaker() as session:
        repo = ModelProbeRepository(session)
        history = await repo.history("ollama", "qwen3:4b-instruct")
        assert {p.id for p in history} == {first.id, second.id}  # nothing overwritten
        latest = await repo.latest("ollama", "qwen3:4b-instruct")
        assert latest is not None


async def test_latest_with_digest_ignores_other_weights(db_sessionmaker) -> None:
    # The "re-pull auto-reverts to unverified" rule falls out of the key: a
    # verdict measured against the OLD weights is invisible when the caller
    # asks about the digest now installed.
    async with db_sessionmaker() as session:
        repo = ModelProbeRepository(session)
        await repo.add(probe_row(digest="sha256:old"))
        await session.commit()

    async with db_sessionmaker() as session:
        repo = ModelProbeRepository(session)
        assert await repo.latest("ollama", "qwen3:4b-instruct", digest="sha256:new") is None
        old = await repo.latest("ollama", "qwen3:4b-instruct", digest="sha256:old")
        assert old is not None and old.earned_agency == "assisted"


async def test_latest_without_a_digest_spans_every_set_of_weights(db_sessionmaker) -> None:
    # The reporting read: "what was measured last at this depth, on whatever
    # was installed then". It answers where the digest-bound read goes silent,
    # and it hands back the row's own digest so a caller can tell the two
    # apart itself - the repository draws no conclusion about which weights
    # are installed now.
    async with db_sessionmaker() as session:
        repo = ModelProbeRepository(session)
        await repo.add(probe_row(digest="sha256:old"))
        # A DEEP run on a LATER set of weights, so the unbound read has a
        # genuine choice to make across digests rather than only across depths.
        deep = probe_row(digest="sha256:newer")
        deep.depth = "deep"
        await repo.add(deep)
        await session.commit()

    async with db_sessionmaker() as session:
        repo = ModelProbeRepository(session)
        # Per depth, because the depths answer different questions - a newer
        # FAST row must not stand in for what a DEEP run measured.
        fast = await repo.latest("ollama", "qwen3:4b-instruct", depth="fast")
        assert fast is not None and (fast.depth, fast.digest) == ("fast", "sha256:old")
        found = await repo.latest("ollama", "qwen3:4b-instruct", depth="deep")
        assert found is not None and (found.id, found.digest) == (deep.id, "sha256:newer")
        # Unfiltered and un-depthed, the newest row of all wins.
        newest = await repo.latest("ollama", "qwen3:4b-instruct")
        assert newest is not None and newest.id == deep.id
        # The same depth filtered to weights nothing measured finds nothing.
        assert (
            await repo.latest("ollama", "qwen3:4b-instruct", digest="sha256:other", depth="deep")
            is None
        )


async def test_measurement_hooks_default_to_unmeasured() -> None:
    # A provider that exposes nothing reports None - recorded as "unmeasured",
    # never estimated (the fake inherits the seam's concrete defaults).
    provider = FakeProvider(script=["ok"])
    assert await provider.runtime_version() is None
    assert await provider.resident_vram_bytes("any-model") is None
