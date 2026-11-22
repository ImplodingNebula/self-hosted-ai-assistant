"""Attachment ingestion: lifecycle, terminal vs transient failure, and splitting."""

import time
from uuid import UUID, uuid4

import pytest

from app.core.config import get_settings
from app.core.tenancy import DEFAULT_TENANT_ID, DEFAULT_USER_ID, TenantContext
from app.models import (
    ATTACHMENT_EXPIRED,
    ATTACHMENT_FAILED,
    ATTACHMENT_ORIGIN_PASTE,
    ATTACHMENT_PARSING,
    ATTACHMENT_PENDING,
    ATTACHMENT_READY,
    Attachment,
    Conversation,
)
from app.parsing.base import ParsedDocument
from app.parsing.registry import default_registry
from app.providers.rag.embedder import FakeEmbedder
from app.repositories.attachments import AttachmentRepository
from app.schemas.rag import RagIngestItem
from app.services.attachment_ingest import (
    MAX_ITEM_CHARS,
    PASTED_DISPLAY_NAME,
    build_ingest_items,
    ingest_attachment,
    split_text,
)
from app.services.rag_service import ingest_structured
from app.storage.base import StorageUnavailableError
from app.storage.local import FakeStorageBackend
from tests.conftest import make_memory_store

TENANT = TenantContext(tenant_id=DEFAULT_TENANT_ID, user_id=DEFAULT_USER_ID, role="admin")

TEXT = (
    "The deployment failed because the storage volume was mounted read-only. "
    "Remounting it read-write and restarting the worker cleared the error."
)


async def _new_conversation(db_sessionmaker) -> UUID:
    async with db_sessionmaker() as session:
        conversation = Conversation(tenant_id=TENANT.tenant_id, user_id=TENANT.user_id)
        session.add(conversation)
        await session.commit()
        return conversation.id


def _attachment(conversation_id: UUID | None, **over) -> Attachment:
    defaults = {
        "tenant_id": TENANT.tenant_id,
        "user_id": TENANT.user_id,
        "conversation_id": conversation_id,
        "filename": "incident.txt",
        "content_type": "text/plain",
        "size_bytes": len(TEXT),
        "storage_key": f"{TENANT.tenant_id}/{uuid4()}",
    }
    return Attachment(**{**defaults, **over})


async def _ingest(session, attachment: Attachment, storage: FakeStorageBackend, **over):
    """Run the service with in-memory everything."""
    kwargs = {
        "attachment": attachment,
        "repo": AttachmentRepository(session, TENANT),
        "storage": storage,
        "registry": default_registry(),
        "embedder": FakeEmbedder(dim=32),
        "store": make_memory_store(),
        "settings": get_settings(),
    }
    return await ingest_attachment(**{**kwargs, **over})


async def test_a_text_file_becomes_retrievable_chunks(db_sessionmaker) -> None:
    conversation_id = await _new_conversation(db_sessionmaker)
    storage = FakeStorageBackend()
    async with db_sessionmaker() as session:
        repo = AttachmentRepository(session, TENANT)
        attachment = repo.add(_attachment(conversation_id))
        await repo.commit()
        await storage.put(attachment.storage_key, TEXT.encode())

        result = await _ingest(session, attachment, storage, repo=repo)

    assert result.status == ATTACHMENT_READY
    assert result.documents == 1
    assert result.chunks >= 1
    assert attachment.status == ATTACHMENT_READY
    assert attachment.error is None
    # The job hashes the bytes it read, so a later upload of the same file into
    # this thread can be recognised without re-reading storage.
    assert attachment.content_hash is not None


async def test_a_pdf_upload_becomes_retrievable_per_page(db_sessionmaker) -> None:
    # End-to-end through the real pipeline: a multi-page PDF is parsed (in the
    # parser's subprocess), one document per page, and each page lands under the
    # file's id so a chat turn can address the whole file.
    from tests.unit.test_pdf_parser import _make_pdf

    conversation_id = await _new_conversation(db_sessionmaker)
    storage = FakeStorageBackend()
    store = make_memory_store()
    pdf = _make_pdf(["Turbine maintenance schedule for Q3.", "Magnet supplier moved to Ostrava."])
    async with db_sessionmaker() as session:
        repo = AttachmentRepository(session, TENANT)
        attachment = repo.add(
            _attachment(conversation_id, filename="report.pdf", content_type="application/pdf")
        )
        await repo.commit()
        await storage.put(attachment.storage_key, pdf)

        result = await _ingest(session, attachment, storage, repo=repo, store=store)

    assert result.status == ATTACHMENT_READY
    assert result.documents == 2  # one per page
    assert result.chunks >= 2
    hits = await store.query(
        "session",
        "fake-embed",
        str(TENANT.tenant_id),
        await FakeEmbedder(dim=32).embed_query("Ostrava magnet supplier"),
        top_k=10,
        session_id=str(conversation_id),
    )
    assert hits
    # Both pages are addressable under the one file id, as distinct documents.
    docids = {hit.doc_id for hit in hits}
    assert docids == {
        f"attachment:{attachment.id}:0:0",
        f"attachment:{attachment.id}:1:0",
    }
    assert all(hit.doc_id.startswith(f"attachment:{attachment.id}:") for hit in hits)


async def test_a_spreadsheet_upload_becomes_retrievable_per_sheet(db_sessionmaker) -> None:
    # End-to-end for the office formats: a workbook is parsed into one document
    # per SHEET, and the sheet's own name rides through to the citation - which is
    # the whole reason a spreadsheet is split this way. "Q3 Forecast" tells the
    # reader where a number came from; "book.xlsx" does not.
    from tests.unit.test_office_parsers import make_xlsx

    conversation_id = await _new_conversation(db_sessionmaker)
    storage = FakeStorageBackend()
    store = make_memory_store()
    workbook = make_xlsx(
        {
            "Q3 Forecast": [["Region", "Revenue"], ["EMEA", 41207]],
            "Assumptions": [["Churn rate", "2%"]],
        }
    )
    async with db_sessionmaker() as session:
        repo = AttachmentRepository(session, TENANT)
        attachment = repo.add(
            _attachment(
                conversation_id,
                filename="book.xlsx",
                content_type=("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
            )
        )
        await repo.commit()
        await storage.put(attachment.storage_key, workbook)

        result = await _ingest(session, attachment, storage, repo=repo, store=store)

    assert result.status == ATTACHMENT_READY
    assert result.documents == 2  # one per sheet
    hits = await store.query(
        "session",
        "fake-embed",
        str(TENANT.tenant_id),
        await FakeEmbedder(dim=32).embed_query("EMEA revenue forecast"),
        top_k=10,
        session_id=str(conversation_id),
    )
    assert hits
    assert all(hit.record_id == str(attachment.id) for hit in hits)
    # The sheet name reaches the citation, qualified by the file it came from.
    assert {"book.xlsx - Q3 Forecast", "book.xlsx - Assumptions"} == {hit.title for hit in hits}


async def test_a_book_upload_becomes_retrievable_per_chapter(db_sessionmaker) -> None:
    # End-to-end for EPUB, and it pins the two things that make a book citable: it is
    # split per CHAPTER, and the chapter's own heading rides through to the citation.
    # The spine here reverses the archive, so a reader that walked the zip instead
    # would still produce two chapters - with the labels swapped, and every citation
    # from them pointing at the wrong one.
    from tests.unit.test_epub_parser import make_book

    conversation_id = await _new_conversation(db_sessionmaker)
    storage = FakeStorageBackend()
    store = make_memory_store()
    book = make_book(
        [
            ("c1", "The Return", "He came back to Ostrava in the spring."),
            ("c2", "The Departure", "She left on a Tuesday, with no forwarding address."),
        ],
        spine=["c2", "c1"],
    )
    async with db_sessionmaker() as session:
        repo = AttachmentRepository(session, TENANT)
        attachment = repo.add(
            _attachment(conversation_id, filename="novel.epub", content_type="application/epub+zip")
        )
        await repo.commit()
        await storage.put(attachment.storage_key, book)

        result = await _ingest(session, attachment, storage, repo=repo, store=store)

    assert result.status == ATTACHMENT_READY
    assert result.documents == 2  # one per chapter
    hits = await store.query(
        "session",
        "fake-embed",
        str(TENANT.tenant_id),
        await FakeEmbedder(dim=32).embed_query("who left on a Tuesday"),
        top_k=10,
        session_id=str(conversation_id),
    )
    assert hits
    assert {"novel.epub - The Departure", "novel.epub - The Return"} == {hit.title for hit in hits}


async def test_an_rtf_upload_is_not_indexed_as_its_own_markup(db_sessionmaker) -> None:
    """The RTF bug, pinned where it actually bit: in the index.

    Before this parser existed an .rtf reached the plain-text fallback, which read it
    successfully - as its own control words. The row went READY, the chunks were
    embedded, and the file then RETRIEVED and got CITED, quoting a font table at
    somebody who had asked what their document said.

    So the assertion that matters is not "the parse worked". It is that nothing
    retrievable from this conversation contains markup.
    """
    from tests.unit.test_rtf_parser import make_rtf

    conversation_id = await _new_conversation(db_sessionmaker)
    storage = FakeStorageBackend()
    store = make_memory_store()
    document = make_rtf(r"\pard\f0 The rollout begins on Tuesday.\par Rollback takes ten minutes.")
    async with db_sessionmaker() as session:
        repo = AttachmentRepository(session, TENANT)
        attachment = repo.add(
            _attachment(conversation_id, filename="plan.rtf", content_type="application/rtf")
        )
        await repo.commit()
        await storage.put(attachment.storage_key, document)

        result = await _ingest(session, attachment, storage, repo=repo, store=store)

    assert result.status == ATTACHMENT_READY
    hits = await store.query(
        "session",
        "fake-embed",
        str(TENANT.tenant_id),
        await FakeEmbedder(dim=32).embed_query("when does the rollout begin"),
        top_k=10,
        session_id=str(conversation_id),
    )
    assert hits
    retrievable = " ".join(hit.text for hit in hits)
    assert "The rollout begins on Tuesday." in retrievable
    for control_word in ("rtf1", "fonttbl", "Calibri", "pard"):
        assert control_word not in retrievable


async def test_an_image_with_no_readable_text_is_accepted_not_failed(db_sessionmaker) -> None:
    # An image's value is not only its text: a vision model can still see it. So
    # an image OCR finds no text in is READY (stored, usable by vision later), not
    # FAILED the way a text document with no text would be. A 1x1 white PNG has
    # nothing to read.
    from io import BytesIO

    from PIL import Image

    conversation_id = await _new_conversation(db_sessionmaker)
    storage = FakeStorageBackend()
    buffer = BytesIO()
    Image.new("RGB", (4, 4), "white").save(buffer, format="PNG")
    async with db_sessionmaker() as session:
        repo = AttachmentRepository(session, TENANT)
        attachment = repo.add(
            _attachment(conversation_id, filename="blank.png", content_type="image/png")
        )
        await repo.commit()
        await storage.put(attachment.storage_key, buffer.getvalue())

        result = await _ingest(session, attachment, storage, repo=repo)

    assert result.status == ATTACHMENT_READY
    assert result.documents == 0  # no text, so nothing retrievable
    assert attachment.status == ATTACHMENT_READY
    assert attachment.error is None


async def test_an_already_ready_attachment_is_not_reingested(db_sessionmaker) -> None:
    conversation_id = await _new_conversation(db_sessionmaker)
    # Deliberately empty: reading the blob would raise, so a pass here proves
    # the guard short-circuited before touching storage or the embedder.
    storage = FakeStorageBackend()
    async with db_sessionmaker() as session:
        repo = AttachmentRepository(session, TENANT)
        attachment = repo.add(_attachment(conversation_id, status=ATTACHMENT_READY))
        await repo.commit()

        result = await _ingest(session, attachment, storage, repo=repo)

    assert result.status == ATTACHMENT_READY
    assert result.chunks == 0


async def test_an_expired_attachment_keeps_its_tombstone(db_sessionmaker) -> None:
    # The retention sweep emptied this row and kept it to say so. Without the
    # guard the job would stamp it `parsing`, find nothing in the store (the same
    # empty storage the test above relies on), and settle it `failed` with "no
    # longer available" - replacing an honest "this aged out" with a fault that
    # never happened, on the one row that exists precisely to say the file was
    # here.
    conversation_id = await _new_conversation(db_sessionmaker)
    storage = FakeStorageBackend()
    async with db_sessionmaker() as session:
        repo = AttachmentRepository(session, TENANT)
        attachment = repo.add(_attachment(conversation_id, status=ATTACHMENT_EXPIRED))
        await repo.commit()

        result = await _ingest(session, attachment, storage, repo=repo)

    assert result.status == ATTACHMENT_EXPIRED
    assert result.chunks == 0
    async with db_sessionmaker() as session:
        row = await session.get(Attachment, attachment.id)
        assert row is not None
        assert row.status == ATTACHMENT_EXPIRED
        assert row.error is None


async def test_a_deleted_conversation_fails_the_row(db_sessionmaker) -> None:
    storage = FakeStorageBackend()
    async with db_sessionmaker() as session:
        repo = AttachmentRepository(session, TENANT)
        attachment = repo.add(_attachment(None))
        await repo.commit()

        result = await _ingest(session, attachment, storage, repo=repo)

    assert result.status == ATTACHMENT_FAILED
    assert "conversation" in result.error.lower()


async def test_a_missing_blob_fails_the_row(db_sessionmaker) -> None:
    conversation_id = await _new_conversation(db_sessionmaker)
    storage = FakeStorageBackend()
    async with db_sessionmaker() as session:
        repo = AttachmentRepository(session, TENANT)
        attachment = repo.add(_attachment(conversation_id))
        await repo.commit()
        # Bytes never stored, or already collected.

        result = await _ingest(session, attachment, storage, repo=repo)

    assert result.status == ATTACHMENT_FAILED
    assert attachment.error == result.error
    assert "no longer available" in result.error


async def test_an_unsupported_file_fails_the_row(db_sessionmaker) -> None:
    conversation_id = await _new_conversation(db_sessionmaker)
    storage = FakeStorageBackend()
    async with db_sessionmaker() as session:
        repo = AttachmentRepository(session, TENANT)
        attachment = repo.add(
            _attachment(conversation_id, filename="photo.heic", content_type="image/heic")
        )
        await repo.commit()
        # NUL bytes: not text, and no parser claims image/heic in this build.
        await storage.put(attachment.storage_key, b"\x00\x01\x02binary\x00")

        result = await _ingest(session, attachment, storage, repo=repo)

    assert result.status == ATTACHMENT_FAILED
    assert "photo.heic" in result.error


async def test_a_file_with_no_text_fails_rather_than_looking_answerable(db_sessionmaker) -> None:
    # `ready` would be a quiet lie: the file would look answerable and ground
    # nothing. A person deserves to be told their file held no readable text.
    conversation_id = await _new_conversation(db_sessionmaker)
    storage = FakeStorageBackend()
    async with db_sessionmaker() as session:
        repo = AttachmentRepository(session, TENANT)
        attachment = repo.add(_attachment(conversation_id))
        await repo.commit()
        await storage.put(attachment.storage_key, b"   \n\n \t  \n")

        result = await _ingest(session, attachment, storage, repo=repo)

    assert result.status == ATTACHMENT_FAILED
    assert "No readable text" in result.error


class _DeadStorage(FakeStorageBackend):
    """Storage that is up enough to accept writes and down when read."""

    async def get(self, key: str) -> bytes:
        raise StorageUnavailableError("Object storage is unavailable.")


async def test_transient_failure_propagates_and_leaves_the_row_parsing(db_sessionmaker) -> None:
    # The file is fine; a service it depends on is down. Retrying is the right
    # answer, so the error must reach arq rather than be recorded as a verdict
    # on the file. The row stays in `parsing` - a stale `updated_at` is how a
    # died-mid-parse job stays visible.
    conversation_id = await _new_conversation(db_sessionmaker)
    storage = _DeadStorage()
    async with db_sessionmaker() as session:
        repo = AttachmentRepository(session, TENANT)
        attachment = repo.add(_attachment(conversation_id))
        await repo.commit()

        with pytest.raises(StorageUnavailableError):
            await _ingest(session, attachment, storage, repo=repo)

    async with db_sessionmaker() as session:
        reloaded = await AttachmentRepository(session, TENANT).get(attachment.id)
        assert reloaded.status == ATTACHMENT_PARSING
        assert reloaded.error is None


async def test_reingesting_the_same_file_neither_duplicates_nor_re_embeds(db_sessionmaker) -> None:
    # Deterministic document ids make a retry idempotent twice over. The chunks
    # already in the store are recognised by their content fingerprint, so the
    # second run re-embeds nothing (chunks written = 0) while still reporting the
    # document as present - and the store holds one copy, not two.
    conversation_id = await _new_conversation(db_sessionmaker)
    storage = FakeStorageBackend()
    store = make_memory_store()
    async with db_sessionmaker() as session:
        repo = AttachmentRepository(session, TENANT)
        attachment = repo.add(_attachment(conversation_id))
        await repo.commit()
        await storage.put(attachment.storage_key, TEXT.encode())

        first = await _ingest(session, attachment, storage, repo=repo, store=store)
        # Force a second real run: a `ready` row short-circuits by design.
        await repo.set_status(attachment, ATTACHMENT_PENDING)
        await repo.commit()
        second = await _ingest(session, attachment, storage, repo=repo, store=store)

    assert first.chunks >= 1
    assert second.documents == first.documents
    # Embedding is the expensive half of ingestion; an unchanged file skips it.
    assert second.chunks == 0
    count = await store.count(
        "session", "fake-embed", str(TENANT.tenant_id), session_id=str(conversation_id)
    )
    assert count == first.chunks


async def test_an_attachments_text_outlives_the_session_ttl(db_sessionmaker) -> None:
    # A file's bytes live for the whole conversation; its searchable text must
    # too. Attachment chunks carry NO ttl, so the sweep that reclaims expired
    # session vectors (throwaway web results) walks past them. The bug this
    # pins: before, they expired after rag_session_ttl_minutes and the file
    # silently stopped grounding answers while the file itself remained.
    conversation_id = await _new_conversation(db_sessionmaker)
    storage = FakeStorageBackend()
    store = make_memory_store()
    space = "fake-embed"
    tenant_id = str(TENANT.tenant_id)
    async with db_sessionmaker() as session:
        repo = AttachmentRepository(session, TENANT)
        attachment = repo.add(_attachment(conversation_id))
        await repo.commit()
        await storage.put(attachment.storage_key, TEXT.encode())
        result = await _ingest(session, attachment, storage, repo=repo, store=store)

    # A throwaway web result in the SAME session, ingested the ordinary way (a
    # TTL), so the sweep has something to prove it still reclaims.
    await ingest_structured(
        source_type="web",
        tenant=TENANT,
        items=[RagIngestItem(id="web-1", title="A page", text="Ephemeral web text.")],
        embedder=FakeEmbedder(dim=32),
        store=store,
        settings=get_settings(),
        session_id=str(conversation_id),
    )

    before = await store.count("session", space, tenant_id, session_id=str(conversation_id))
    assert before == result.chunks + 1 >= 2

    # Sweep everything expired well into the future: the web chunk goes, the
    # file's chunks stay.
    await store.sweep_expired(time.time() + 10 * 365 * 24 * 3600)

    after = await store.count("session", space, tenant_id, session_id=str(conversation_id))
    assert after == result.chunks
    survivors = await store.query(
        "session",
        space,
        tenant_id,
        await FakeEmbedder(dim=32).embed_query(TEXT),
        top_k=50,
        session_id=str(conversation_id),
    )
    assert survivors  # the file is still retrievable
    assert all(hit.doc_id.startswith(f"attachment:{attachment.id}:") for hit in survivors)


async def test_a_pasted_block_is_labelled_pasted_content() -> None:
    attachment = _attachment(uuid4(), filename="pasted-text.txt", origin=ATTACHMENT_ORIGIN_PASTE)
    items = build_ingest_items(attachment, [ParsedDocument(ordinal=0, text=TEXT)])

    assert len(items) == 1
    # The citation names where the text came from, not a filename nobody chose.
    assert items[0].title == PASTED_DISPLAY_NAME


async def test_units_are_labelled_and_blank_ones_dropped() -> None:
    attachment = _attachment(uuid4(), filename="report.pdf")
    items = build_ingest_items(
        attachment,
        [
            ParsedDocument(ordinal=0, text="First page.", title="page 1"),
            ParsedDocument(ordinal=1, text="   \n ", title="page 2"),
            ParsedDocument(ordinal=2, text="Third page.", title="page 3"),
        ],
    )

    assert [item.title for item in items] == ["report.pdf - page 1", "report.pdf - page 3"]
    # Ids stay distinct per unit, so re-ingesting rewrites rather than duplicates.
    assert len({item.id for item in items}) == 2


def test_split_cap_matches_the_ingest_schema() -> None:
    # Drift here would truncate documents silently rather than split them.
    max_length = RagIngestItem.model_fields["text"].metadata[-1].max_length
    assert max_length == MAX_ITEM_CHARS


def test_short_text_is_not_split() -> None:
    assert split_text(TEXT) == [TEXT]


def test_long_text_is_split_not_truncated() -> None:
    text = " ".join(f"word{i:05d}" for i in range(30_000))
    pieces = split_text(text)

    assert len(pieces) > 1
    assert all(len(piece) <= MAX_ITEM_CHARS for piece in pieces)
    # Nothing is dropped: the tail of a document is exactly what a question
    # about the end of a long file needs.
    assert "".join(pieces) == text


def test_a_split_lands_between_words() -> None:
    pieces = split_text("alpha beta gamma delta", limit=12)

    assert pieces == ["alpha beta ", "gamma delta"]


def test_text_without_whitespace_is_cut_at_the_cap() -> None:
    # A minified bundle offers nowhere clean to break; the hard limit still holds.
    pieces = split_text("x" * 25, limit=10)

    assert pieces == ["x" * 10, "x" * 10, "x" * 5]


async def test_a_unit_longer_than_the_cap_becomes_several_items() -> None:
    attachment = _attachment(uuid4())
    long_text = " ".join(f"word{i:05d}" for i in range(20_000))
    items = build_ingest_items(attachment, [ParsedDocument(ordinal=0, text=long_text)])

    assert len(items) > 1
    assert [item.title for item in items] == [
        f"incident.txt (part {n + 1})" for n in range(len(items))
    ]
    assert len({item.id for item in items}) == len(items)


class _DeletesTheRowMidEmbed(FakeEmbedder):
    """Embeds normally, but deletes the attachment row on its first call.

    Stands in for the owner clicking remove while the job is still working: the
    row goes away AFTER the job claimed it (status `parsing`) and BEFORE the
    chunks it is about to write land in the store.
    """

    def __init__(self, db_sessionmaker, attachment_id: UUID, **kwargs) -> None:
        super().__init__(**kwargs)
        self._sessionmaker = db_sessionmaker
        self._attachment_id = attachment_id
        self._deleted = False

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not self._deleted:
            self._deleted = True
            async with self._sessionmaker() as session:
                doomed = await AttachmentRepository(session, TENANT).get(self._attachment_id)
                if doomed is not None:
                    await session.delete(doomed)
                    await session.commit()
        return await super().embed_texts(texts)


async def test_a_file_deleted_while_it_was_ingesting_leaves_no_retrievable_text(
    db_sessionmaker,
) -> None:
    """The worst failure this pipeline can produce, pinned.

    Parsing and embedding take seconds, and the owner can delete the file during
    them. The delete purges whatever chunks exist at that moment - none yet - and
    the job's write then lands behind it. Those chunks would be unreachable
    garbage that still answers questions: the row was the only thing that could
    address them, attachment chunks carry no TTL, and nothing short of deleting
    the whole conversation would ever sweep them. So the job that wrote them has
    to take them back.
    """
    conversation_id = await _new_conversation(db_sessionmaker)
    storage = FakeStorageBackend()
    store = make_memory_store()

    async with db_sessionmaker() as session:
        repo = AttachmentRepository(session, TENANT)
        attachment = repo.add(_attachment(conversation_id))
        await repo.commit()
        await storage.put(attachment.storage_key, TEXT.encode())

        result = await _ingest(
            session,
            attachment,
            storage,
            repo=repo,
            store=store,
            embedder=_DeletesTheRowMidEmbed(db_sessionmaker, attachment.id, dim=32),
        )

    # The job noticed its row had gone and took its own write back.
    assert result.status == "deleted"
    hits = await store.query(
        "session",
        "fake-embed",
        str(TENANT.tenant_id),
        await FakeEmbedder(dim=32).embed_query("incident"),
        top_k=50,
        session_id=str(conversation_id),
    )
    assert hits == []
