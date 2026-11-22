"""Tests for how an image's OCR'd text is presented to the model.

An image is stored as the text OCR read out of it, and that text is retrieved
like any other passage. But a text-only chat model asked "what is in this image"
refuses reflexively - it looks for image DATA, finds a passage labelled like
every other upload, and answers that it cannot see images, even while holding
the image's own text. Measured live on qwen3:4b-instruct against a real
screenshot: labelled ``(upload)`` it refused 3 times out of 3, once while
quoting the very text it was refusing to use; labelled as text read from the
image, with the matching note on the header, it answered correctly 3 times out
of 3.

So the label is load-bearing, not cosmetic, and these tests pin it.

Note the block shape they assert against: only the generated index and the
source label sit outside the fence, and the passage TITLE is the fence's first
line - untrusted titles must not land in the trusted scaffolding.
"""

from app.parsing.image import ImageParser
from app.providers.rag.base import ScoredChunk
from app.services.prompts import build_grounded_message


def _chunk(title: str, *, text: str = "passage text", source_type: str = "upload") -> ScoredChunk:
    return ScoredChunk(
        id=f"id-{title}",
        score=0.5,
        text=text,
        doc_id=f"doc-{title}",
        title=title,
        source_type=source_type,
    )


def test_an_image_passage_says_its_text_was_read_from_the_image() -> None:
    prompt = build_grounded_message("What is in this image?", [_chunk("screenshot.png")])

    assert "[1] (text read from the image)" in prompt
    # The bare source type is what made the model treat it as any other upload.
    assert "[1] (upload)" not in prompt
    # The filename is still shown, inside the fence where untrusted text belongs.
    assert "<<<\nscreenshot.png\npassage text\n>>>" in prompt


def test_an_image_passage_adds_the_ocr_note_to_the_header() -> None:
    prompt = build_grounded_message("What is in this image?", [_chunk("photo.jpg")])

    assert "extracted from that image by OCR" in prompt
    assert "rather than saying the image cannot be seen" in prompt


def test_a_non_image_passage_is_unchanged_and_carries_no_ocr_note() -> None:
    # The note costs header tokens the context budget accounts for, so an
    # ordinary turn must not pay for it.
    prompt = build_grounded_message("Explain this", [_chunk("main.py")])

    assert "[1] (upload)" in prompt
    assert "<<<\nmain.py\n" in prompt
    assert "OCR" not in prompt


def test_the_note_appears_once_even_with_several_images() -> None:
    prompt = build_grounded_message(
        "Compare these",
        [_chunk("a.png"), _chunk("b.jpeg"), _chunk("notes.md")],
    )

    assert prompt.count("extracted from that image by OCR") == 1
    assert "[1] (text read from the image)\n<<<\na.png\n" in prompt
    assert "[2] (text read from the image)\n<<<\nb.jpeg\n" in prompt
    assert "[3] (upload)\n<<<\nnotes.md\n" in prompt


def test_every_image_extension_the_parser_claims_is_labelled() -> None:
    # Driven off ImageParser itself, so a format added there is covered here
    # without a second edit - that shared set is the whole point of the wiring.
    for extension in ImageParser.extensions:
        prompt = build_grounded_message("q", [_chunk(f"shot{extension}")])
        assert "(text read from the image)" in prompt, extension


def test_the_label_is_case_insensitive_about_the_extension() -> None:
    prompt = build_grounded_message("q", [_chunk("SCREENSHOT.PNG")])

    assert "(text read from the image)" in prompt


def test_a_split_image_still_reads_as_an_image_in_every_part() -> None:
    # An image whose OCR text exceeds the item cap is ingested in pieces, and
    # every piece's title gains a " (part N)" suffix - which lands exactly where
    # the extension check looks. Unstripped, a big scan silently loses the label
    # and the model goes back to refusing to read it.
    prompt = build_grounded_message(
        "What is in this scan?",
        [_chunk("dense-scan.tiff (part 1)"), _chunk("dense-scan.tiff (part 2)")],
    )

    assert "[1] (text read from the image)\n<<<\ndense-scan.tiff (part 1)\n" in prompt
    assert "[2] (text read from the image)\n<<<\ndense-scan.tiff (part 2)\n" in prompt
    assert "extracted from that image by OCR" in prompt


def test_a_stored_record_that_merely_looks_like_an_image_is_not_dressed_up_as_one() -> None:
    # Records and tool results carry client-chosen titles. One that happens to
    # end in .png must not be presented as an image, nor drag the OCR note into
    # the header of a turn that holds no image at all.
    prompt = build_grounded_message("q", [_chunk("logo.png", source_type="records")])

    assert "[1] (records)\n<<<\nlogo.png\n" in prompt
    assert "OCR" not in prompt


def test_a_web_passage_keeps_its_own_source_type() -> None:
    prompt = build_grounded_message("q", [_chunk("An article", source_type="web")])

    assert "[1] (web)\n<<<\nAn article\n" in prompt
