"""Unit tests for the tool-guidance layer's composition across tool sources.

The layer's per-tool bullets are hand-written for the two LOCAL tools, but the
offered set also carries tools this module knows nothing about (MCP). These
tests pin the split: bullets stay local-only, while the header and the safety
footer key on "any tool at all". A turn that ships tools with no guidance is
the measured 0/6-tool-call prompt, so the MCP-only case is a real regression
guard, not a hypothetical.

The (web, knowledge) cases that predate ``other`` live in test_config.py
alongside the other system-prompt semantics; those pin backward compatibility
by passing no ``other`` at all.
"""

from app.services.prompts import (
    _SEARCH_KNOWLEDGE_GUIDANCE,
    _TOOL_GUIDANCE_ATTRIBUTION,
    _TOOL_GUIDANCE_FOOTER,
    _TOOL_GUIDANCE_HEADER,
    _WEB_ATTRIBUTION,
    _WEB_SEARCH_GUIDANCE,
    DEFAULT_SYSTEM_PROMPT,
    append_tool_guidance,
    tool_guidance,
)

# The locked ASCII-safe-glyph rule: no U+2026 ellipsis and no Unicode dash of
# any width (U+2010-U+2015 plus the U+2212 minus). They render as broken glyphs
# in some terminals, and prompt copy is also fed to a model, where a stray
# glyph is one more token of noise. Held as CODEPOINTS, never as literals: the
# rule binds this file too, and a literal here would be the very byte the test
# forbids.
_FORBIDDEN_CODEPOINTS = frozenset([*range(0x2010, 0x2016), 0x2212, 0x2026])

_ALL_LOCAL_COMBINATIONS = [(False, False), (False, True), (True, False), (True, True)]


def _expected(web: bool, knowledge: bool) -> str:
    """Compose the guidance the pre-``other`` implementation produced.

    Built from the module's own constants, so the pin is on the COMPOSITION
    (which bullets, in what order, with which footer) rather than on the exact
    prompt copy, which is tuned against live models and expected to move.
    """
    bullets = []
    if web:
        bullets.append(_WEB_SEARCH_GUIDANCE)
    if knowledge:
        bullets.append(_SEARCH_KNOWLEDGE_GUIDANCE)
    if not bullets:
        return ""
    footer = _TOOL_GUIDANCE_FOOTER + (_WEB_ATTRIBUTION if web else "")
    return "\n".join([_TOOL_GUIDANCE_HEADER, *bullets, footer])


def test_other_zero_is_byte_identical_to_the_pre_mcp_layer() -> None:
    # Adding `other` must not move an existing caller by a single byte: every
    # caller today passes only web/knowledge and must keep its exact prompt.
    for web, knowledge in _ALL_LOCAL_COMBINATIONS:
        explicit = tool_guidance(web=web, knowledge=knowledge, other=0)
        assert explicit == _expected(web, knowledge)
        # The default must BE zero, not merely behave like it nearby.
        assert tool_guidance(web=web, knowledge=knowledge) == explicit


def test_no_tools_at_all_returns_empty() -> None:
    # The no-tools contract: a turn offering nothing carries no tool text, or
    # the persona would describe capabilities the model was not handed.
    assert tool_guidance(web=False, knowledge=False) == ""
    assert tool_guidance(web=False, knowledge=False, other=0) == ""


def test_mcp_only_turn_gets_header_and_footer_and_names_no_tool() -> None:
    # The regression this parameter exists for: web off, retrieval unwired, two
    # MCP tools offered. The old two-boolean layer returned "" here and shipped
    # real tools with zero guidance.
    guidance = tool_guidance(web=False, knowledge=False, other=2)
    assert guidance
    assert _TOOL_GUIDANCE_HEADER in guidance
    assert "Never invent a tool" in guidance
    # Exactly header + footer: no bullet, and no invented account of what the
    # MCP tools do. Their wire descriptions are the only grounded description
    # of them, and a 4B taught an ungroundable specific fabricates.
    assert guidance == "\n".join([_TOOL_GUIDANCE_HEADER, _TOOL_GUIDANCE_FOOTER])
    assert "web_search" not in guidance
    assert "search_knowledge" not in guidance
    # No web tool this turn, so the URL-attribution sentence would dangle.
    assert _WEB_ATTRIBUTION.strip() not in guidance


def test_one_other_tool_is_enough_to_emit_the_layer() -> None:
    # The gate is "any tool at all", so the boundary sits at 1, not at 2.
    assert tool_guidance(web=False, knowledge=False, other=1)


def test_other_adds_no_text_alongside_local_tools() -> None:
    # With local tools present the layer is already emitted, so `other` must be
    # a no-op on the text: same bullets, same footer, nothing appended.
    for web, knowledge in _ALL_LOCAL_COMBINATIONS:
        if not (web or knowledge):
            continue
        with_other = tool_guidance(web=web, knowledge=knowledge, other=3)
        assert with_other == _expected(web, knowledge)


def test_web_attribution_stays_gated_on_web_when_other_tools_are_offered() -> None:
    # An MCP tool must not drag in a sentence about web results.
    web_on = tool_guidance(web=True, knowledge=False, other=2)
    assert "Never write a URL" in web_on
    kb_on = tool_guidance(web=False, knowledge=True, other=2)
    assert "Never write a URL" not in kb_on


def test_append_tool_guidance_layers_an_mcp_only_turn_onto_the_persona() -> None:
    layered = append_tool_guidance(DEFAULT_SYSTEM_PROMPT, web=False, knowledge=False, other=2)
    assert layered.startswith(DEFAULT_SYSTEM_PROMPT)
    assert "Never invent a tool" in layered
    # A rewritten persona must not lose the safety half either.
    custom = append_tool_guidance("Be terse.", web=False, knowledge=False, other=2)
    assert custom.startswith("Be terse.")
    assert "Never invent a tool" in custom


def test_append_tool_guidance_other_zero_is_byte_identical() -> None:
    for web, knowledge in _ALL_LOCAL_COMBINATIONS:
        expected = append_tool_guidance(DEFAULT_SYSTEM_PROMPT, web=web, knowledge=knowledge)
        assert (
            append_tool_guidance(DEFAULT_SYSTEM_PROMPT, web=web, knowledge=knowledge, other=0)
            == expected
        )


def test_append_tool_guidance_respects_disabled_prompt_for_other_tools() -> None:
    # "" disables the system prompt entirely, guidance included - MCP tools do
    # not resurrect it.
    assert append_tool_guidance("", web=False, knowledge=False, other=2) == ""
    assert append_tool_guidance("", web=True, knowledge=True, other=2) == ""


def test_append_tool_guidance_with_no_tools_returns_the_persona_unchanged() -> None:
    assert (
        append_tool_guidance(DEFAULT_SYSTEM_PROMPT, web=False, knowledge=False)
        == DEFAULT_SYSTEM_PROMPT
    )
    assert append_tool_guidance("Be terse.", web=False, knowledge=False, other=0) == "Be terse."


def test_attribution_sentence_rides_every_tools_offered_turn() -> None:
    # Usage-gating means the whole citation chain hangs off the model's own [n]:
    # a tool-grounded answer written without markers emits no citations at all.
    # So the sentence has to reach every turn that offers ANY source-returning
    # tool, MCP-only turns included - not just the two local ones.
    for other in (0, 2):
        for web, knowledge in _ALL_LOCAL_COMBINATIONS:
            guidance = tool_guidance(web=web, knowledge=knowledge, other=other)
            if not guidance:
                continue
            assert _TOOL_GUIDANCE_ATTRIBUTION in guidance


def test_attribution_sentence_keeps_its_conditional_and_its_prohibition() -> None:
    # Both halves are the mitigation, not decoration. An UNCONDITIONAL "cite by
    # [n]" in this layer is measured in this repo to make a DECLINING model
    # fabricate a marker it has nothing to ground in, which is why the teaching
    # otherwise lives inside the result fence. This sentence earns its place by
    # being conditional on results existing and by forbidding the invented
    # marker in the same breath; an edit that drops either half reintroduces the
    # measured failure.
    assert _TOOL_GUIDANCE_ATTRIBUTION.startswith("When a tool result lists numbered sources")
    assert "never write a bracketed number the results did not give you" in (
        _TOOL_GUIDANCE_ATTRIBUTION
    )


def test_guidance_text_is_ascii_safe() -> None:
    texts = [
        tool_guidance(web=web, knowledge=knowledge, other=other)
        for web, knowledge in _ALL_LOCAL_COMBINATIONS
        for other in (0, 2)
    ]
    texts.append(append_tool_guidance(DEFAULT_SYSTEM_PROMPT, web=True, knowledge=True, other=2))
    for text in texts:
        offenders = sorted({hex(ord(ch)) for ch in text if ord(ch) in _FORBIDDEN_CODEPOINTS})
        assert not offenders, f"forbidden glyphs in guidance: {offenders}"
        # Broader than the codepoint set above, and the cheaper guard: the
        # whole layer is ASCII, so any smart quote or NBSP fails here too.
        assert text.isascii()
