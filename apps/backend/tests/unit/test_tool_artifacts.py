"""Files a tool hands back: extraction from MCP, and storing them host-side.

Two halves, matching the two layers the path crosses.

The PROVIDER half asks what ``_to_tool_result`` makes of a raw MCP result: a
blob-bearing ``EmbeddedResource`` has to become an artifact rather than being
swallowed by the "[non-text content omitted]" marker that non-text items have
always got, and one malformed blob must cost that file and nothing else.

The HOST half asks what the service does with those bytes. The properties worth
pinning are not "it saves a file" but the four ways saving a file goes wrong: a
filename chosen by a remote server, bytes written without a row (or a row without
bytes), a cap that fails open, and a generated row leaking into the upload path's
deduplication.
"""

from __future__ import annotations

import base64
import hashlib
from uuid import UUID

import pytest
from mcp import types as mcp_types
from pydantic import AnyUrl
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.core.tenancy import DEFAULT_TENANT_ID, DEFAULT_USER_ID, TenantContext
from app.models import (
    ATTACHMENT_ORIGIN_GENERATED,
    ATTACHMENT_ORIGIN_UPLOAD,
    ATTACHMENT_READY,
    Attachment,
)
from app.providers.mcp import source as source_module
from app.providers.tools.base import ToolArtifact
from app.repositories.attachments import AttachmentRepository
from app.repositories.conversations import ConversationRepository
from app.services.tool_artifacts import (
    _MEDIA_TYPE,
    safe_artifact_content_type,
    safe_artifact_filename,
    store_tool_artifacts,
)
from app.storage.base import StorageUnavailableError
from app.storage.local import FakeStorageBackend

TENANT = TenantContext(tenant_id=DEFAULT_TENANT_ID, user_id=DEFAULT_USER_ID)

PDF = b"%PDF-1.7 pretend this is a rendered report"


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


def _embedded(data: bytes, *, uri: str = "artifact://report.pdf", mime: str = "application/pdf"):
    """One EmbeddedResource carrying a base64 blob - the locked wire shape."""
    return mcp_types.EmbeddedResource(
        type="resource",
        resource=mcp_types.BlobResourceContents(
            uri=AnyUrl(uri), blob=base64.b64encode(data).decode(), mimeType=mime
        ),
    )


def _text(text: str) -> mcp_types.TextContent:
    return mcp_types.TextContent(type="text", text=text)


def _extract(raw: mcp_types.CallToolResult):
    """Adapt a raw result the way a real call does - naming the server it came from."""
    return source_module._to_tool_result(raw, server_name="example-server")


# --------------------------------------------------------------------------- #
# Extraction: what the provider makes of a raw MCP result                      #
# --------------------------------------------------------------------------- #


def test_a_blob_resource_becomes_an_artifact_and_leaves_the_text_alone() -> None:
    # The whole wire contract in one result: a summary the model reads, and a
    # file it does not. The bytes must not appear in `content` in any form -
    # that path is token-budgeted, and a base64 payload in it evicts the prompt
    # head to feed the model characters it cannot use.
    raw = mcp_types.CallToolResult(content=[_text("Wrote report.pdf (2 pages)."), _embedded(PDF)])

    result = _extract(raw)

    assert result.content == "Wrote report.pdf (2 pages)."
    assert [(a.filename, a.content_type, a.data) for a in result.artifacts] == [
        ("report.pdf", "application/pdf", PDF)
    ]


def test_a_kept_blob_does_not_trigger_the_omitted_marker() -> None:
    # The marker means "something came back that you are not being shown". A
    # file that WAS kept is the opposite of that, so claiming it was omitted
    # would make the model apologise for a document sitting in the user's hands.
    raw = mcp_types.CallToolResult(content=[_text("done"), _embedded(PDF)])

    assert "[non-text content omitted]" not in _extract(raw).content


def test_a_non_blob_embedded_resource_still_gets_the_omitted_marker() -> None:
    # The gate is the BLOB, not the EmbeddedResource wrapper. A text-valued
    # embedded resource is not a file, is not shown to the model, and therefore
    # keeps exactly the behaviour it had before artifacts existed.
    raw = mcp_types.CallToolResult(
        content=[
            _text("here"),
            mcp_types.EmbeddedResource(
                type="resource",
                resource=mcp_types.TextResourceContents(
                    uri=AnyUrl("file:///notes.txt"), text="some text", mimeType="text/plain"
                ),
            ),
        ]
    )

    result = _extract(raw)

    assert result.artifacts == []
    assert "[non-text content omitted]" in result.content


def test_an_image_item_still_gets_the_omitted_marker() -> None:
    raw = mcp_types.CallToolResult(
        content=[
            _text("here"),
            mcp_types.ImageContent(
                type="image", data=base64.b64encode(b"png").decode(), mimeType="image/png"
            ),
        ]
    )

    result = _extract(raw)

    assert result.artifacts == []
    assert "[non-text content omitted]" in result.content


def test_a_malformed_blob_costs_that_file_and_nothing_else() -> None:
    # A server that mangles one attachment must not cost the user the answer.
    # The bad file is dropped and DISCLOSED (the marker); the good one, the text
    # and the rest of the call all survive.
    bad = mcp_types.EmbeddedResource(
        type="resource",
        resource=mcp_types.BlobResourceContents(
            uri=AnyUrl("artifact://broken.pdf"), blob="not!base64!!", mimeType="application/pdf"
        ),
    )
    raw = mcp_types.CallToolResult(content=[_text("two files"), bad, _embedded(PDF)])

    result = _extract(raw)

    assert [a.filename for a in result.artifacts] == ["report.pdf"]
    assert "[non-text content omitted]" in result.content
    assert result.is_error is False


def test_the_uri_is_a_filename_hint_not_a_gate() -> None:
    # A document-generating server is expected to send artifact:// uris, but the
    # host must keep a file from ANY server. Filtering on the scheme would
    # silently discard real documents from every server that spells its uris
    # differently - and silently is the point: nothing would report the loss.
    raw = mcp_types.CallToolResult(
        content=[_text("ok"), _embedded(PDF, uri="https://example.test/exports/q3.pdf")]
    )

    assert [a.filename for a in _extract(raw).artifacts] == ["q3.pdf"]


def test_the_per_call_artifact_count_is_capped() -> None:
    # A structural bound BELOW any service policy: the host should not decode a
    # thousand blobs into memory before a setting gets a look at them.
    raw = mcp_types.CallToolResult(
        content=[_text("many")] + [_embedded(PDF) for _ in range(source_module._MAX_ARTIFACTS + 3)]
    )

    result = _extract(raw)

    assert len(result.artifacts) == source_module._MAX_ARTIFACTS
    assert "[non-text content omitted]" in result.content


def test_an_empty_blob_is_not_a_file() -> None:
    raw = mcp_types.CallToolResult(content=[_text("nothing"), _embedded(b"")])

    assert _extract(raw).artifacts == []


# --------------------------------------------------------------------------- #
# Filename validation: the name came from a stranger                           #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "hint",
    [
        "../../etc/passwd",
        "..",
        ".",
        "..\\..\\windows\\system32",
        "/etc/passwd",
        ".hidden",
        "report.",
        "CON.txt",
        "nul",
        "LPT1.pdf",
        'report";rm -rf /.pdf',
        "",
        "   ",
        "x" * 400,
    ],
)
def test_an_unusable_filename_is_replaced_not_repaired(hint: str) -> None:
    # Replaced, never patched up: editing a hostile name until it passes turns a
    # name that meant something into a name that means something else, and the
    # reader has no way to tell that happened.
    assert safe_artifact_filename(hint, content_type="application/pdf", index=1) == "artifact-1.pdf"


def test_an_honest_filename_survives_including_its_spaces() -> None:
    # The one normalization, and the reason for it: real documents are called
    # "Q3 report.xlsx", and failing that into a generated name would lose
    # information for no safety gained.
    assert (
        safe_artifact_filename(
            "Q3 report.xlsx",
            content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            index=1,
        )
        == "Q3-report.xlsx"
    )
    assert safe_artifact_filename("report.pdf", content_type="application/pdf", index=1) == (
        "report.pdf"
    )


def test_a_newline_in_a_filename_is_folded_not_carried() -> None:
    # Whitespace normalization doing security work as a side effect: a CR/LF in a
    # name is what would terminate a Content-Disposition header, and it comes out
    # of here as an ordinary dash rather than as a rejected name.
    assert safe_artifact_filename("re\r\nport.pdf", content_type="application/pdf", index=1) == (
        "re-port.pdf"
    )


@pytest.mark.parametrize(
    "declared",
    [
        "application/pdf\r\nX-Injected: yes",
        "application/pdf; charset=</script>",
        "not a media type",
        "",
        "/",
        "application/" + "x" * 200,
    ],
)
def test_a_media_type_that_is_not_one_is_replaced(declared: str) -> None:
    # Same reasoning as the filename, one field over. This value is what a
    # download response would echo as Content-Type, so a CR/LF in it is header
    # injection - and the endpoint that would suffer for it does not exist yet,
    # which is exactly why the guard belongs at the point of storage.
    assert safe_artifact_content_type(declared) == "application/octet-stream"


def test_the_media_type_pattern_itself_refuses_a_trailing_newline() -> None:
    # The pattern, not safe_artifact_content_type: that function strips first, so
    # a trailing newline never reaches the match and the endpoint behaves the
    # same either way. The guard being pinned here is the PATTERN's anchor. In
    # Python '$' also matches just before a trailing newline (the trap
    # app/storage/base.py documents for storage keys), so with '$' this value
    # would be accepted as a media type - and the only thing standing between
    # that and a Content-Type header carrying a line break would be one caller
    # remembering to strip. \Z removes the dependency on that memory.
    assert _MEDIA_TYPE.match("application/pdf\n") is None
    assert _MEDIA_TYPE.match("application/pdf") is not None


def test_an_honest_media_type_is_kept() -> None:
    assert safe_artifact_content_type("application/pdf") == "application/pdf"
    assert safe_artifact_content_type(
        " application/vnd.openxmlformats-officedocument.spreadsheetml.sheet "
    ) == ("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


async def test_a_hostile_media_type_never_reaches_the_row(
    db_sessionmaker, fake_storage, conversation_id
) -> None:
    stored, _ = await _store(
        [
            ToolArtifact(
                filename="report.pdf",
                content_type="application/pdf\r\nX-Injected: yes",
                data=PDF,
            )
        ],
        sessionmaker=db_sessionmaker,
        storage=fake_storage,
        conversation_id=conversation_id,
    )

    rows = await _rows(db_sessionmaker)
    assert stored[0].content_type == "application/octet-stream"
    assert rows[0].content_type == "application/octet-stream"


def test_an_unknown_media_type_falls_back_to_a_neutral_suffix() -> None:
    assert safe_artifact_filename("", content_type="application/x-nonsense", index=3) == (
        "artifact-3.bin"
    )


def test_a_percent_encoded_name_is_decoded_before_it_is_judged() -> None:
    # The hint comes off a URI, where a space is spelled %20. Judging the encoded
    # form fails a perfectly honest name on the "%" and hands back
    # "artifact-1.pdf" - so the decode has to happen before the check, not after.
    assert safe_artifact_filename("Q3%20report.pdf", content_type="application/pdf", index=1) == (
        "Q3-report.pdf"
    )
    # And decoding cannot smuggle anything past the check that follows: %2F is a
    # separator once decoded, and is refused exactly as a literal slash is.
    assert safe_artifact_filename(
        "..%2F..%2Fetc%2Fpasswd", content_type="application/pdf", index=1
    ) == ("artifact-1.pdf")


async def test_two_calls_in_one_turn_cannot_mint_the_same_fallback_name(
    db_sessionmaker, fake_storage, conversation_id
) -> None:
    # The numbering has to count files KEPT, not files seen. This call drops one
    # artifact (over the size cap) and keeps one, so a seen-based counter would
    # advance by two while the caller's kept-based counter advances by one - and
    # the next call's first file would reuse a number already handed out.
    settings = _settings(tool_artifact_max_bytes=len(PDF))
    first, _ = await _store(
        [_artifact(data=PDF * 2, name=".."), _artifact(name="..")],
        sessionmaker=db_sessionmaker,
        storage=fake_storage,
        conversation_id=conversation_id,
        settings=settings,
    )
    # The caller numbers the next call from what the turn has actually kept.
    second, _ = await store_tool_artifacts(
        [_artifact(name="..")],
        tenant=TENANT,
        conversation_id=conversation_id,
        sessionmaker=db_sessionmaker,
        storage=fake_storage,
        settings=settings,
        remaining=10 - len(first),
        first_index=len(first) + 1,
    )

    names = [file.filename for file in [*first, *second]]
    assert names == ["artifact-1.pdf", "artifact-2.pdf"]
    assert len(set(names)) == len(names)


# --------------------------------------------------------------------------- #
# Materialization: bytes, row, caps, and the dedupe boundary                   #
# --------------------------------------------------------------------------- #


class _DeadStorage(FakeStorageBackend):
    """A store that refuses every write, the way an unreachable backend does."""

    async def put(self, key, data, *, content_type=None):
        raise StorageUnavailableError("storage is down")


class _FailingSession:
    """A session whose commit fails, to drive the bytes-written-no-row window."""

    def __init__(self) -> None:
        self.added: list[object] = []

    def add(self, entity: object) -> None:
        self.added.append(entity)

    async def commit(self) -> None:
        raise RuntimeError("database is down")

    async def __aenter__(self) -> _FailingSession:
        return self

    async def __aexit__(self, *_exc: object) -> bool:
        return False


async def _store(
    artifacts: list[ToolArtifact],
    *,
    sessionmaker,
    storage,
    conversation_id: UUID,
    settings: Settings | None = None,
    remaining: int = 10,
):
    return await store_tool_artifacts(
        artifacts,
        tenant=TENANT,
        conversation_id=conversation_id,
        sessionmaker=sessionmaker,
        storage=storage,
        settings=settings or _settings(),
        remaining=remaining,
        first_index=1,
    )


def _artifact(data: bytes = PDF, name: str = "report.pdf") -> ToolArtifact:
    return ToolArtifact(filename=name, content_type="application/pdf", data=data)


async def _rows(db_sessionmaker: async_sessionmaker[AsyncSession]) -> list[Attachment]:
    async with db_sessionmaker() as session:
        result = await session.execute(select(Attachment))
        return list(result.scalars().all())


@pytest.fixture
async def conversation_id(db_sessionmaker: async_sessionmaker[AsyncSession]) -> UUID:
    """A real conversation row: an attachment carries a foreign key to one."""
    async with db_sessionmaker() as session:
        conversation = await ConversationRepository(session, TENANT).create()
        await session.commit()
        return conversation.id


async def test_the_bytes_land_in_storage_and_the_row_describes_them(
    db_sessionmaker, fake_storage, conversation_id
) -> None:
    stored, notices = await _store(
        [_artifact()],
        sessionmaker=db_sessionmaker,
        storage=fake_storage,
        conversation_id=conversation_id,
    )

    assert notices == []
    assert len(stored) == 1
    announced = stored[0]
    assert announced.filename == "report.pdf"
    assert announced.size_bytes == len(PDF)
    assert announced.origin == ATTACHMENT_ORIGIN_GENERATED
    # The key is the minted triple, which is the ONLY shape the orphan-blob
    # sweep will consider - a generated file that lost its row has to be
    # collectable by machinery that already exists.
    key = f"{TENANT.tenant_id}/{conversation_id}/{announced.attachment_id}"
    assert fake_storage.objects[key] == PDF
    rows = await _rows(db_sessionmaker)
    assert len(rows) == 1
    row = rows[0]
    assert str(row.id) == announced.attachment_id
    assert row.storage_key == key
    # Born ready with no hash: nothing parses it, and nothing may dedupe onto it.
    assert row.status == ATTACHMENT_READY
    assert row.origin == ATTACHMENT_ORIGIN_GENERATED
    assert row.content_hash is None
    assert row.tenant_id == TENANT.tenant_id
    assert row.user_id == TENANT.user_id
    assert row.conversation_id == conversation_id


async def test_a_storage_failure_writes_no_row_and_says_so(
    db_sessionmaker, conversation_id
) -> None:
    # Bytes before row means a failed write must leave NOTHING behind - a
    # committed row pointing at a blob that does not exist is a download that
    # 500s forever - and the turn must still answer.
    stored, notices = await _store(
        [_artifact()],
        sessionmaker=db_sessionmaker,
        storage=_DeadStorage(),
        conversation_id=conversation_id,
    )

    assert stored == []
    assert await _rows(db_sessionmaker) == []
    assert len(notices) == 1
    assert "could not be stored" in notices[0]


async def test_a_row_that_cannot_commit_leaves_the_blob_for_the_sweep(
    fake_storage, conversation_id
) -> None:
    # The other side of the same window. The blob is already written, so it is
    # deliberately NOT deleted here: it sits under a minted key, which is what
    # the orphan sweep collects. What must not happen is a silent success.
    stored, notices = await _store(
        [_artifact()],
        sessionmaker=lambda: _FailingSession(),
        storage=fake_storage,
        conversation_id=conversation_id,
    )

    assert stored == []
    assert len(notices) == 1
    assert "could not be saved" in notices[0]
    assert len(fake_storage.objects) == 1
    assert next(iter(fake_storage.objects)).startswith(f"{TENANT.tenant_id}/{conversation_id}/")


async def test_a_file_at_the_size_cap_is_kept(
    db_sessionmaker, fake_storage, conversation_id
) -> None:
    settings = _settings(tool_artifact_max_bytes=len(PDF))

    stored, notices = await _store(
        [_artifact()],
        sessionmaker=db_sessionmaker,
        storage=fake_storage,
        conversation_id=conversation_id,
        settings=settings,
    )

    assert len(stored) == 1
    assert notices == []


async def test_a_file_over_the_size_cap_is_dropped_and_disclosed(
    db_sessionmaker, fake_storage, conversation_id
) -> None:
    # The disclosure is the point. A file that silently fails to appear is
    # indistinguishable from one the tool never made, and the model would go on
    # describing a document nobody has.
    settings = _settings(tool_artifact_max_bytes=len(PDF) - 1)

    stored, notices = await _store(
        [_artifact()],
        sessionmaker=db_sessionmaker,
        storage=fake_storage,
        conversation_id=conversation_id,
        settings=settings,
    )

    assert stored == []
    assert await _rows(db_sessionmaker) == []
    assert fake_storage.objects == {}
    assert len(notices) == 1
    assert "was not kept" in notices[0]
    assert "report.pdf" in notices[0]


async def test_a_dropped_file_never_borrows_the_name_of_a_kept_one(
    db_sessionmaker, fake_storage, conversation_id
) -> None:
    # The exact collision shape. Two artifacts, NEITHER with a usable name: the
    # first is over the size cap and dropped, the second is stored. If the
    # dropped one had taken a fallback number, the stored one would take the same
    # number - and the notice would apologise for "artifact-1.pdf" while a file
    # the user can see and open is called "artifact-1.pdf". A fallback number is
    # a property of a KEPT file.
    settings = _settings(tool_artifact_max_bytes=len(PDF))
    stored, notices = await _store(
        [_artifact(data=PDF * 2, name=".."), _artifact(name="..")],
        sessionmaker=db_sessionmaker,
        storage=fake_storage,
        conversation_id=conversation_id,
        settings=settings,
    )

    assert [file.filename for file in stored] == ["artifact-1.pdf"]
    assert len(notices) == 1
    # The notice describes the dropped file by type and size, and names no file.
    assert "artifact-1.pdf" not in notices[0]
    assert "an unnamed pdf file" in notices[0]
    assert "was not kept" in notices[0]


async def test_a_dropped_file_is_still_named_when_the_server_gave_a_usable_name(
    db_sessionmaker, fake_storage, conversation_id
) -> None:
    # The other half: describing rather than naming is a fallback, not a policy.
    # A server that gave an honest name gets it quoted back, because "your Q3
    # report was too large" is far more use than "a file was too large".
    settings = _settings(tool_artifact_max_bytes=len(PDF) - 1)
    stored, notices = await _store(
        [_artifact(name="q3-report.pdf")],
        sessionmaker=db_sessionmaker,
        storage=fake_storage,
        conversation_id=conversation_id,
        settings=settings,
    )

    assert stored == []
    assert "'q3-report.pdf'" in notices[0]
    # And the sizes read as sizes. The unit scales rather than being fixed at MB:
    # a deployment that lowered the cap would otherwise be told its file was
    # "0.0 MB, over the 0.0 MB limit", which explains nothing at all.
    assert "0.0 MB" not in notices[0]
    assert " B, over the " in notices[0]


async def test_the_turn_count_cap_stops_at_the_budget_and_discloses(
    db_sessionmaker, fake_storage, conversation_id
) -> None:
    stored, notices = await _store(
        [_artifact(name=f"r{i}.pdf") for i in range(4)],
        sessionmaker=db_sessionmaker,
        storage=fake_storage,
        conversation_id=conversation_id,
        remaining=2,
    )

    assert [file.filename for file in stored] == ["r0.pdf", "r1.pdf"]
    assert len(await _rows(db_sessionmaker)) == 2
    assert len(notices) == 1
    assert "not kept" in notices[0]


async def test_no_room_left_keeps_nothing_at_all(
    db_sessionmaker, fake_storage, conversation_id
) -> None:
    stored, notices = await _store(
        [_artifact()],
        sessionmaker=db_sessionmaker,
        storage=fake_storage,
        conversation_id=conversation_id,
        remaining=0,
    )

    assert stored == []
    assert fake_storage.objects == {}
    assert len(notices) == 1


async def test_a_deployment_without_storage_keeps_nothing_and_says_so(
    db_sessionmaker, conversation_id
) -> None:
    stored, notices = await _store(
        [_artifact()],
        sessionmaker=db_sessionmaker,
        storage=None,
        conversation_id=conversation_id,
    )

    assert stored == []
    assert len(notices) == 1
    assert "no file storage" in notices[0]


async def test_a_hostile_filename_never_reaches_the_row(
    db_sessionmaker, fake_storage, conversation_id
) -> None:
    stored, _ = await _store(
        [ToolArtifact(filename="../../etc/passwd", content_type="application/pdf", data=PDF)],
        sessionmaker=db_sessionmaker,
        storage=fake_storage,
        conversation_id=conversation_id,
    )

    rows = await _rows(db_sessionmaker)
    assert stored[0].filename == "artifact-1.pdf"
    assert rows[0].filename == "artifact-1.pdf"


async def test_an_upload_is_never_deduplicated_onto_a_generated_row(
    db_sessionmaker, fake_storage, conversation_id
) -> None:
    # THE ONE THAT WOULD BE A REAL BUG. get_by_content_hash is how an upload
    # reuses an identical file already in the thread. If a generated row carried
    # a hash, someone uploading the same bytes would be handed the assistant's
    # file - a row with no parse job and no indexed text - and their upload would
    # never be readable. A null hash makes that unrepresentable.
    await _store(
        [_artifact()],
        sessionmaker=db_sessionmaker,
        storage=fake_storage,
        conversation_id=conversation_id,
    )

    async with db_sessionmaker() as session:
        match = await AttachmentRepository(session, TENANT).get_by_content_hash(
            hashlib.sha256(PDF).hexdigest(), conversation_id=conversation_id
        )

    assert match is None


async def test_the_repository_still_rejects_an_origin_nobody_defined(
    db_sessionmaker, conversation_id
) -> None:
    # `origin` is a plain string column (portable, migration-free), so this check
    # is the ONLY thing standing between a typo and a value nothing understands.
    async with db_sessionmaker() as session:
        repo = AttachmentRepository(session, TENANT)
        with pytest.raises(ValueError, match="origin"):
            repo.add(
                Attachment(
                    tenant_id=TENANT.tenant_id,
                    user_id=TENANT.user_id,
                    conversation_id=conversation_id,
                    filename="x.pdf",
                    content_type="application/pdf",
                    size_bytes=1,
                    storage_key="a/b/c",
                    origin="invented",
                )
            )


async def test_a_client_cannot_upload_a_file_claiming_the_assistant_made_it(
    client, db_sessionmaker
) -> None:
    # `generated` is now a valid column value, which is exactly why the upload
    # endpoint has to stop accepting whatever origin a form field names: the UI
    # reads that field to decide whether a file is a user's draft, so a client
    # able to claim it could hide an upload from its own composer.
    conversation = await client.post("/api/v1/conversations", json={})
    conversation_id = conversation.json()["id"]

    response = await client.post(
        "/api/v1/attachments",
        data={"conversation_id": conversation_id, "origin": ATTACHMENT_ORIGIN_GENERATED},
        files={"file": ("notes.txt", b"hello there", "text/plain")},
    )

    assert response.status_code == 400
    assert ATTACHMENT_ORIGIN_UPLOAD in response.json()["detail"]
