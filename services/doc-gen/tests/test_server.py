"""The wire contract, the tool annotations, and the transport gates.

The contract tests are the load-bearing ones: the platform host reads a file
result POSITIONALLY - summary first, artifact second - so the order, the block
count, the ``artifact://`` URI and the media type are pinned here rather than
left to whichever renderer happened to build them.
"""

from __future__ import annotations

import httpx
import pytest
from mcp.types import BlobResourceContents, EmbeddedResource, TextContent

from app.main import build_app
from tests.helpers import call_content, docgen_http_client, mcp_session

pytestmark = pytest.mark.anyio

API_KEY = "test-docgen-key"

_MINIMAL_DOCUMENT = {"format": "md", "spec": {"text": "hello"}}

# Every field the two tools accept, its type, its required-ness and its enum
# values - the whole of what a caller can express, with all prose removed.
#
# Pinned as a LITERAL, and deliberately blind to `title`, `description` and a
# null `default`. Those three are annotations no validator reads, so they can be
# trimmed out of the schema to save prompt tokens; a field, a type, a required
# entry or an enum value cannot, because each one is something the model would
# lose the ability to say. This snapshot is what makes that distinction testable
# instead of asserted: it must survive any amount of prose editing and must fail
# the moment the capability surface actually narrows.
EXPECTED_SURFACE: dict[str, dict] = {
    "create_document": {
        "required": ["format", "spec"],
        "properties": {"format": "enum[xlsx|docx|pdf|csv|md]", "spec": "DocumentSpec"},
        "additionalProperties": None,
        "$defs": {
            "Block": {
                "required": [],
                "properties": {
                    "items": "array[string]|null",
                    "level": "integer|null",
                    "rows": "array[array[string]]|null",
                    "text": "string|null",
                    "type": "string ='paragraph'",
                },
                "additionalProperties": None,
            },
            "DocumentSpec": {
                "required": [],
                "properties": {
                    "blocks": "array[Block]|null",
                    "filename": "string|null",
                    "rows": "array[array[string|integer|number|boolean|null]]|null",
                    "sheets": "array[Sheet]|null",
                    "text": "string|null",
                    "title": "string|null",
                },
                "additionalProperties": None,
            },
            "Sheet": {
                "required": [],
                "properties": {
                    "header_row": "boolean =False",
                    "name": "string|null",
                    "rows": "array[array[string|integer|number|boolean|null]]",
                },
                "additionalProperties": None,
            },
        },
    },
    "render_chart": {
        "required": ["spec"],
        "properties": {"spec": "ChartSpec"},
        "additionalProperties": None,
        "$defs": {
            "ChartSpec": {
                "required": [],
                "properties": {
                    "filename": "string|null",
                    "format": "enum[png|svg] ='png'",
                    "labels": "array[string]|null",
                    "series": "array[Series]",
                    "title": "string|null",
                    "type": "string ='bar'",
                    "x_label": "string|null",
                    "y_label": "string|null",
                },
                "additionalProperties": None,
            },
            "Series": {
                "required": [],
                "properties": {"name": "string|null", "values": "array[number]"},
                "additionalProperties": None,
            },
        },
    },
}


# The keywords _type_of renders itself, plus the annotations it is meant to
# ignore. Anything else is a BOUND, and gets appended verbatim rather than
# dropped: a constraint added later (max_length, minimum, pattern) narrows what
# the model may send just as surely as deleting a field does, and it would
# otherwise hide behind an unchanged type name.
_RENDERED_KEYS = frozenset(
    {
        "type",
        "items",
        "anyOf",
        "enum",
        "$ref",
        "default",
        # Prose. Listed so that editing it stays free - the point of the
        # surface is that it moves only when CAPABILITY moves.
        "title",
        "description",
        "examples",
        "deprecated",
    }
)


def _type_of(node: dict) -> str:
    """One schema node as a compact type expression: refs, enums, unions, arrays."""
    if "$ref" in node:
        base = node["$ref"].rsplit("/", 1)[-1]
    elif "enum" in node:
        base = "enum[" + "|".join(str(value) for value in node["enum"]) + "]"
    elif "anyOf" in node:
        base = "|".join(_type_of(branch) for branch in node["anyOf"])
    elif node.get("type") == "array":
        base = f"array[{_type_of(node.get('items', {}))}]"
    else:
        base = str(node.get("type", "?"))
    bounds = sorted(set(node) - _RENDERED_KEYS)
    return base + "".join(f"+{key}={node[key]!r}" for key in bounds)


def _properties_of(node: dict) -> dict[str, str]:
    """Each property as its type expression, plus any default that INFORMS.

    A null default is dropped and a ``false`` one kept: "omit it and you get
    null" is what being absent from ``required`` already says, where "omit it
    and the first row is not bold" is a fact only the default carries.
    """
    surface = {}
    for name, prop in node.get("properties", {}).items():
        entry = _type_of(prop)
        if prop.get("default") is not None:
            entry += f" ={prop['default']!r}"
        surface[name] = entry
    return surface


def _object_surface(node: dict) -> dict:
    """One object schema as its required set, its fields, and its extras policy.

    ``additionalProperties`` is read because it decides what happens to a key
    the model invented: ignored today, a hard validation error under
    ``extra="forbid"``. That turns a near-miss field name from a dropped value
    into a failed tool call, so it belongs to the capability surface even
    though it names no field.
    """
    return {
        "required": sorted(node.get("required", [])),
        "properties": _properties_of(node),
        "additionalProperties": node.get("additionalProperties"),
    }


def _capability_surface(schema: dict) -> dict:
    """Reduce a tool's input schema to the parts a validator acts on."""
    # Nested under its own key rather than flattened alongside: a spec model
    # named "properties" would otherwise overwrite the root's entry and take
    # the root's whole field list out of the snapshot silently.
    return {
        **_object_surface(schema),
        "$defs": {
            def_name: _object_surface(def_schema)
            for def_name, def_schema in schema.get("$defs", {}).items()
        },
    }


def _lead_sentence(description: str) -> str:
    """The description up to its first full stop."""
    return description.split(".", 1)[0].lower()


async def test_tools_list_and_both_are_annotated_read_only() -> None:
    async with mcp_session(build_app()) as session:
        listed = await session.list_tools()

    by_name = {tool.name: tool for tool in listed.tools}
    assert set(by_name) == {"create_document", "render_chart"}
    # Load-bearing, not cosmetic: the platform fails CLOSED on a missing
    # readOnlyHint and would route every document through a confirmation prompt.
    assert by_name["create_document"].annotations.readOnlyHint is True
    assert by_name["render_chart"].annotations.readOnlyHint is True


async def test_tool_descriptions_name_the_artifacts_they_produce() -> None:
    # Descriptions are read twice: the model picks a tool by them, and the
    # platform's top-K tool selection embeds them whole and ranks by similarity.
    #
    # Asserted against the LEAD SENTENCE because that is the part a HUMAN
    # skim and a truncating client both keep, and because it is the one part
    # guaranteed to survive any future shortening. Selection itself embeds the
    # whole string with no positional weighting, so this is a convention that
    # keeps the summary strong, not a mechanical property of the ranker.
    async with mcp_session(build_app()) as session:
        listed = await session.list_tools()

    by_name = {tool.name: _lead_sentence(tool.description or "") for tool in listed.tools}
    for word in ("spreadsheet", "document", "pdf", "csv", "markdown", "download"):
        assert word in by_name["create_document"], word
    for word in ("chart", "png", "svg", "download"):
        assert word in by_name["render_chart"], word


async def test_the_input_schemas_expose_the_full_capability_surface() -> None:
    # The token trim's whole claim: fewer bytes, nothing the model can no longer
    # say. See EXPECTED_SURFACE for what counts as capability and what counts as
    # prose.
    async with mcp_session(build_app()) as session:
        listed = await session.list_tools()

    surfaces = {tool.name: _capability_surface(tool.inputSchema) for tool in listed.tools}
    assert surfaces == EXPECTED_SURFACE


async def test_the_published_schemas_carry_no_inert_annotations() -> None:
    # The counterpart to the snapshot above, and the reason both exist. That one
    # is blind to `title` and a null `default` ON PURPOSE, so it cannot notice
    # the token trim being switched off - a spec model that forgets the
    # SpecModel base, or the base losing its config, would re-inflate every
    # agentic turn's prompt with the whole suite still green. This is the
    # assertion that fails instead.
    async with mcp_session(build_app()) as session:
        listed = await session.list_tools()

    for tool in listed.tools:
        # $defs only. The root arguments model is built by the SDK from the
        # handler signature, is not a SpecModel, and keeps its own title - see
        # SpecModel's docstring for why that one is left alone.
        for model_name, model_schema in tool.inputSchema.get("$defs", {}).items():
            assert "title" not in model_schema, f"{tool.name}.{model_name}"
            for field_name, field_schema in model_schema.get("properties", {}).items():
                where = f"{tool.name}.{model_name}.{field_name}"
                assert "title" not in field_schema, where
                # `false` and `0` are defaults that inform and must SURVIVE; only
                # a null one is the redundant restatement of being optional.
                assert not ("default" in field_schema and field_schema["default"] is None), where


@pytest.mark.parametrize(
    ("tool", "args", "suffix", "media_type"),
    [
        (
            "create_document",
            {"format": "xlsx", "spec": {"filename": "book", "rows": [["a"]]}},
            ".xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ),
        (
            "create_document",
            {"format": "docx", "spec": {"filename": "doc", "text": "hi"}},
            ".docx",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ),
        (
            "create_document",
            {"format": "pdf", "spec": {"filename": "doc", "text": "hi"}},
            ".pdf",
            "application/pdf",
        ),
        (
            "create_document",
            {"format": "csv", "spec": {"filename": "rows", "rows": [["a"]]}},
            ".csv",
            "text/csv",
        ),
        ("create_document", {"format": "md", "spec": {"text": "hi"}}, ".md", "text/markdown"),
        (
            "render_chart",
            {"spec": {"filename": "plot", "series": [{"values": [1, 2]}], "format": "png"}},
            ".png",
            "image/png",
        ),
        (
            "render_chart",
            {"spec": {"filename": "plot", "series": [{"values": [1, 2]}], "format": "svg"}},
            ".svg",
            "image/svg+xml",
        ),
    ],
)
async def test_every_file_result_follows_the_artifact_contract(
    tool: str, args: dict, suffix: str, media_type: str
) -> None:
    async with mcp_session(build_app()) as session:
        content = await call_content(session, tool, args)

    # ORDER: the one-line summary the model reads, then the file.
    assert isinstance(content[0], TextContent)
    assert content[0].text.startswith("Created ")
    assert isinstance(content[1], EmbeddedResource)
    # COUNT: one call, one file - the host reads the artifact positionally. The
    # literal 1, not the constant: asserting a constant against itself is a test
    # that moves with the value it is supposed to pin.
    embedded = [block for block in content if isinstance(block, EmbeddedResource)]
    assert len(embedded) == 1

    resource = embedded[0].resource
    assert isinstance(resource, BlobResourceContents)
    uri = str(resource.uri)
    assert uri.startswith("artifact://")
    assert uri.endswith(suffix)
    assert resource.mimeType == media_type
    # The name in the summary is the name on the artifact.
    assert uri.removeprefix("artifact://") in content[0].text


async def test_the_artifact_uri_keeps_the_filename_verbatim() -> None:
    # The URI is parsed by the host to name a download, so its case and
    # punctuation must survive the trip through pydantic's URL type.
    async with mcp_session(build_app()) as session:
        content = await call_content(
            session,
            "create_document",
            {"format": "md", "spec": {"filename": "Q1_Board-Report.v2", "text": "hi"}},
        )

    assert str(content[1].resource.uri) == "artifact://Q1_Board-Report.v2.md"


async def test_a_file_result_carries_no_structured_content() -> None:
    # structuredContent is the platform's citable-RECORDS contract. A file is
    # not a record, and returning both would have the host ingest the summary
    # line as a source.
    async with mcp_session(build_app()) as session:
        result = await session.call_tool("create_document", _MINIMAL_DOCUMENT)

    assert result.isError is False
    assert result.structuredContent is None


async def test_neither_tool_advertises_an_output_schema() -> None:
    # The mechanical half of the rule above: an output schema is what makes
    # FastMCP serialize a return value into structuredContent.
    async with mcp_session(build_app()) as session:
        listed = await session.list_tools()

    assert all(tool.outputSchema is None for tool in listed.tools)


async def test_a_refused_spec_returns_text_only_and_is_not_an_error() -> None:
    # No artifact, and NOT isError: the platform stops offering a tool that
    # errored, which would take away the retry along with the mistake.
    async with mcp_session(build_app()) as session:
        result = await session.call_tool("create_document", {"format": "csv", "spec": {}})

    assert result.isError is False
    assert len(result.content) == 1
    assert isinstance(result.content[0], TextContent)
    assert "rows" in result.content[0].text


async def test_healthz_is_open() -> None:
    async with docgen_http_client(build_app()) as http_client:
        response = await http_client.get("/healthz")

    assert response.status_code == 200
    assert response.text == "ok"


@pytest.mark.parametrize(
    "headers",
    [None, {"Authorization": "Bearer wrong-key"}],
    ids=["missing-header", "wrong-key"],
)
async def test_auth_rejects_calls_without_the_bearer_key(
    monkeypatch: pytest.MonkeyPatch, headers: dict[str, str] | None
) -> None:
    monkeypatch.setenv("DOCGEN_API_KEY", API_KEY)
    app = build_app()

    # The 401 lands on the initialize POST and the SDK surfaces it as an
    # httpx.HTTPStatusError wrapped in the client task group's exception
    # group(s) - never as a usable session.
    with pytest.raises(BaseExceptionGroup) as excinfo:
        async with mcp_session(app, headers=headers):
            pytest.fail("an unauthenticated session must not initialize")
    assert excinfo.group_contains(httpx.HTTPStatusError, match="401 Unauthorized")


async def test_auth_accepts_the_bearer_key_and_healthz_stays_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DOCGEN_API_KEY", API_KEY)
    app = build_app()

    async with mcp_session(app, headers={"Authorization": f"Bearer {API_KEY}"}) as session:
        listed = await session.list_tools()
    assert {tool.name for tool in listed.tools} == {"create_document", "render_chart"}

    # The liveness probe must not require the key. A fresh app is built because
    # a FastMCP session manager's run() is once-per-instance, so an app's
    # lifespan cannot be entered twice.
    async with docgen_http_client(build_app()) as http_client:
        response = await http_client.get("/healthz")
    assert response.status_code == 200


async def test_a_whitespace_api_key_is_treated_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    # A blank value is what a failed compose variable substitution produces; the
    # service boots open (with a warning) rather than locking on "Bearer ".
    # WHITESPACE, not "": an empty string is already falsy, so testing "" proves
    # only that `if api_key:` works and leaves the .strip() branch untested.
    monkeypatch.setenv("DOCGEN_API_KEY", "   ")

    async with mcp_session(build_app()) as session:
        listed = await session.list_tools()

    assert len(listed.tools) == 2


async def test_a_disallowed_host_is_rejected() -> None:
    # The SDK's DNS-rebinding protection rejects unlisted Host headers with a
    # 421 before the protocol layer sees the request.
    with pytest.raises(BaseExceptionGroup) as excinfo:
        async with mcp_session(build_app(), base_url="http://evil.example"):
            pytest.fail("a disallowed Host must not initialize")
    assert excinfo.group_contains(httpx.HTTPStatusError, match="421")


async def test_the_allowed_hosts_env_is_honored(monkeypatch: pytest.MonkeyPatch) -> None:
    # The env var is the whole reason transport security is configurable: a
    # deployment reaching this service under another hostname lists it here.
    monkeypatch.setenv("DOCGEN_ALLOWED_HOSTS", "docs.internal,docs.internal:*")

    async with mcp_session(build_app(), base_url="http://docs.internal") as session:
        listed = await session.list_tools()

    assert len(listed.tools) == 2


async def test_blank_allowed_hosts_falls_back_to_the_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An empty allow list would be the SDK's reject-everything state.
    monkeypatch.setenv("DOCGEN_ALLOWED_HOSTS", "  ,  ")

    async with mcp_session(build_app()) as session:
        listed = await session.list_tools()

    assert len(listed.tools) == 2


async def test_the_compose_service_name_is_reachable_by_default() -> None:
    # The backend dials http://doc-gen:8765/docgen/mcp, so the default allow
    # list must accept that Host or every call 421s in the compose network.
    async with mcp_session(build_app(), base_url="http://doc-gen:8765") as session:
        listed = await session.list_tools()

    assert len(listed.tools) == 2
