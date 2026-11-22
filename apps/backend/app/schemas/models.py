"""HTTP schema for the models endpoints.

Typed response models (instead of bare dicts) so the OpenAPI schema - and the
TypeScript types generated from it - carry the real shape of the model
selector payload and of the management view.

``POST /models/pull`` streams Server-Sent Events. Each event has a name and a
JSON ``data`` payload::

    event: progress  data: {"status","completed","total","percent","digest"}
    event: done      data: {"model","status"}            # terminal, success
    event: error     data: {"message","code"}            # terminal, failure

A download emits many ``progress`` frames and exactly one terminal frame -
``done`` when the weights landed, ``error`` when they did not. Comment frames
starting with ":" (e.g. ": keep-alive") are idle heartbeats; SSE clients ignore
them by spec, and a download that spends minutes on one blob relies on them to
keep the connection open.

``status`` is the backend's own progress wording, passed through for display
because it is genuinely more informative than any fixed vocabulary would be
(fetching a manifest, downloading a specific blob, verifying a digest). Clients
must render it rather than switch on it. ``completed``/``total`` are byte
counts and are present only while a blob is transferring; ``percent`` is
derived from them, rounded to one decimal, and is null whenever they are
absent - a client must not compute progress from a stage name.

Error ``code`` values: ``model_not_found`` (the registry has no such tag),
``provider_unavailable`` (the model server could not be reached, or failed
mid-download), ``internal_error`` (unexpected backend failure). The stream
never carries an exception's text.

``POST /models/verify`` streams the tool-check probe::

    event: verify_progress  data: {"case","model","passed","detail",
                                   "latency_s","completed","total"}
    event: verify_done      data: {"model","depth","earned",...}  # terminal
    event: verify_error     data: {"message","code"}              # terminal

One ``verify_progress`` frame per finished sample - a deep check runs
twenty-six of them over several minutes, a fast one four, which is why this is
a stream and not a silent POST - then exactly one terminal frame. ``verify_done`` carries the
verdicts the run earned plus what it observed (throughput, the weights digest,
the runtime version, and where the model actually sat while it ran);
``verify_error`` carries the same three codes as a download, and means the
measurement could not be MADE, which is emphatically not the same as a model
that measured badly - that is a ``verify_done`` with a modest ``earned``.

The verification frames keep these names on every stream that carries them,
which is what lets a download continue into one without two different ``done``
frames colliding on one connection.

**The terminal frame is not the end of the connection.** A check evicts the
chat model and the embedder to measure a fresh load, and puts both back
afterwards - which happens AFTER the verdict is sent, so the socket stays open
for the ten to thirty seconds that re-warm takes. That silence is the
deployment being made usable again, not a hang: a client renders the verdict as
final when it arrives and simply lets the stream close in its own time.

Everything a verification can be refused for is decided before the response
begins, so a refusal is problem+json: 422 (unusable name), 404 (the backend
does not hold those weights), 409 (another probe is already running), 403 (not
an admin).

``POST /models/embedder-check`` is the retrieval-side sibling and is a plain
JSON request, not a stream: it embeds a fixed probe corpus and answers with the
score bands that came out, which takes seconds rather than the minutes a
generation probe takes. Its answer either VALIDATES the relevance floors
already declared for that embedder or PROPOSES a pair derived from the bands.
Nothing is applied - the floors live in configuration, and a number that
changes what every retrieval returns is a human's decision.

**A download can continue into a check.** ``POST /models/pull`` accepts
``verify`` (default true) and, when the download succeeds, runs the FAST check
on the tag it just fetched as a continuation of the SAME stream - one user
action, one connection. The download's ``done`` frame stays terminal FOR THE
DOWNLOAD and still reports success on its own terms; it carries ``verifying``
to say whether verification frames follow it, and ``verify_skipped_reason`` to
say why not when they do not. A check that fails therefore never presents as a
failed download - by then the ``done`` frame reporting the successful download
has already been sent.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.providers.base import ModelInfo
from app.schemas.common import UtcDatetime
from app.schemas.rag import StrandedVectorsReport
from app.services.model_names import MAX_NAME_LENGTH
from app.services.model_resolution import AgencyMode, ThinkMode, ToolsMode


class ModelServerStatus(BaseModel):
    """Where the model server sits, and whether the app sends it credentials.

    Configuration, read straight from settings and never from a live call, so
    it answers the same under any provider. It exists so a client can say
    plainly that a model server outside this deployment is being reached with
    no credential: the server authenticates nothing of its own, so an
    unprotected endpoint on a public network answers whoever finds it. That is
    a WARNING, never a refusal - an open server on a trusted network is a
    legitimate deployment.

    Attributes:
        remote: The configured URL names a host outside this deployment.
        auth_configured: The app sends credentials with every call. It never
            says WHAT they are, only that something is set.
        host: The hostname alone - never the full URL, and never any
            credential a URL might carry in its userinfo. Empty when the
            configured value has no parseable host.
    """

    remote: bool = False
    auth_configured: bool = False
    host: str = ""


class ModelEntry(BaseModel):
    """One model the server holds, described by what the runtime reports of it.

    Attributes:
        role: What the model can serve, derived from the runtime's report:
            ``chat`` when it can complete, ``embedding`` when it can embed and
            cannot complete, and empty when the report establishes neither.
            Derived rather than declared, because a declaration written on
            another machine cannot know what this server holds.
        provider: The serving backend this deployment is configured with.
        model: The exact tag as the provider lists it.
        capabilities: What is known about the tag: ``tools`` and ``vision``
            from the runtime's own mechanical report, plus ``thinking`` and
            ``thinking_model`` when something answers for them - a stored
            confirmation, a measurement, the code-shipped quirk seed, or the
            model's own template. An absent key means nothing knows, which is
            a different claim from a key saying no.
    """

    role: str
    provider: str
    model: str
    capabilities: dict = Field(default_factory=dict)


class ModelsResponse(BaseModel):
    """Live provider models, and what is known about each of them.

    Attributes:
        models: Every model the provider currently holds, exactly as it listed
            them.
        entries: One row per held tag that answered a capability probe, in the
            provider's own order. The chat ones are what a conversation can be
            pinned to. ``role`` is NOT unique across this list: filter on it,
            never key on it. A tag whose probe FAILED is left out here and
            still present in ``models`` - "the runtime would not say" is the
            honest place for it, and the next fetch tries again.
        server: Where the model server is and whether it is being reached with
            credentials - enough for a client to warn about an unprotected
            remote endpoint, and nothing more.
    """

    models: list[ModelInfo]
    entries: list[ModelEntry]
    # Required, not defaulted: the one handler that builds this response always
    # knows the answer, and a default would generate an OPTIONAL field for
    # every client - each of which would then have to treat "absent" as some
    # third state that never occurs.
    server: ModelServerStatus


class WarmRequest(BaseModel):
    """Body for the warm endpoint; every field is optional.

    ``model`` defaults to the configured default chat model. ``keep_alive``
    is passed through to the provider as a residency hint: a duration string
    (``"5m"``) or integer seconds, where ``-1`` means indefinite on Ollama.
    """

    model: str | None = None
    keep_alive: str | int | None = None

    @field_validator("keep_alive", mode="before")
    @classmethod
    def _numeric_string_as_int(cls, value: object) -> object:
        """Coerce a bare numeric string (e.g. ``"-1"`` from make) to an int.

        Ollama parses string values with Go's duration parser, which rejects
        unit-less numbers (time: missing unit in duration), while a JSON
        number is read as seconds (negative = keep loaded indefinitely).
        Shell callers cannot easily emit an unquoted JSON number, so the
        boundary accepts the quoted form and forwards the working type.
        """
        if isinstance(value, str):
            try:
                return int(value.strip())
            except ValueError:
                return value
        return value


class WarmResponse(BaseModel):
    """Result of a warm run: what was primed and how long it took."""

    model: str
    took_ms: int
    # Size of the evaluated (now cached) prompt prefix, when reported.
    prompt_tokens: int | None = None


class WarmSearchResponse(BaseModel):
    """Result of a search-backend warm run: how long the priming query took."""

    took_ms: int
    # Results the throwaway query returned; informational only (zero is fine).
    results: int


# --- the model manager view ---


class ProviderInfo(BaseModel):
    """The serving backend and which management verbs it supports.

    Attributes:
        name: The ``MODEL_PROVIDER`` selector value currently serving.
        runtime_version: The runtime's version when it exposes one; None
            simply means it does not, never that it is old.
        can_pull: The backend can download weights on request.
        can_delete: The backend can remove weights on request.
        can_list_installed: The backend can enumerate what it holds.
    """

    name: str
    runtime_version: str | None = None
    can_pull: bool = False
    can_delete: bool = False
    can_list_installed: bool = False


class ResidentModel(BaseModel):
    """One model the serving runtime is holding in memory right now.

    Every number is one the runtime reported, and null where it reported none -
    there is nothing derived or predicted here beyond ``gpu_fraction``, which is
    the ratio of the two byte counts beside it.

    Attributes:
        model: The tag as the runtime knows it, which is not necessarily a tag
            this deployment ever configured.
        total_bytes: Bytes the loaded model occupies across all memory.
        vram_bytes: The part of that sitting in GPU memory.
        gpu_fraction: ``vram_bytes`` over ``total_bytes``, 0.0 to 1.0, or null
            when either count is missing. Near zero means the model spilled to
            the CPU - it still answers, an order of magnitude slower - and a
            reader must say so rather than showing a bare "0 GB".
    """

    model: str
    total_bytes: int | None = None
    vram_bytes: int | None = None
    gpu_fraction: float | None = None


class ModelResidencyResponse(BaseModel):
    """What the serving runtime holds in memory right now.

    Attributes:
        reported: The runtime answers residency questions at all. False means
            this payload says NOTHING about what is loaded - a process-per-model
            backend holds whatever it was started with and has no changing
            residency to report - and a reader must not render it as "nothing
            loaded", which is the empty list beside a true here.
        models: One entry per resident model, in the runtime's own order.
    """

    reported: bool = False
    models: list[ResidentModel] = []


class MeasuredFacts(BaseModel):
    """What a tool-check probe actually MEASURED on the installed weights.

    Present only when a digest-bound probe row exists; absent (null) means
    unmeasured, which a reader must render as such rather than filling in an
    estimate. Nothing here is predicted - there is no fit or throughput model
    anywhere in this payload.

    Attributes:
        depth: The probe depth these numbers come from ("fast" or "deep").
        tokens_per_second: Decode throughput the probe observed, if recorded.
            Null on a run that was not ``representative``: a CPU-bound decode
            rate describes the spill, not the deployment.
        vram_bytes: Resident VRAM the probe observed, if recorded.
        measured_at: When the probe ran, always with an explicit UTC offset -
            a reader compares it against the live runtime and against other
            timestamps in the payload, and an offset-less stamp would be read
            as local time.
        runtime_version: The runtime the probe ran under, if recorded.
        runtime_stale: The runtime has changed since the probe ran, so the
            numbers describe a different serving stack. A banner fact - it
            does not clamp anything.
        harness_stale: The tool-check cases have changed since the probe ran,
            so the verdict answers a slightly different question.
        endpoint_stale: The probe does not describe the model server being
            served from now - it named a different one, or named none at all.
            Unlike the two flags above, this one explains a verdict that has
            already been DISCOUNTED: throughput is the host's hardware and
            tool behaviour belongs to the runtime that host runs, so a verdict
            earned elsewhere certifies nothing here, and neither does one whose
            host went unrecorded (the shape of every probe taken before this
            was tracked). The numbers stay on show as the record of what that
            run measured; what the flag says is that they are not about this
            server. A reader that renders one staleness notice should treat
            this one as the reason a check is being asked for again.
        representative: Whether the probe describes how the deployment serves.
            False means the model spilled off the GPU while it ran, which is
            what a reader needs to explain the two numbers above: a full spill
            records ``vram_bytes`` of literally zero and no throughput at all,
            and "0 B, unmeasured" with nothing saying why reads as a broken
            page rather than as a model that ran on the CPU. Such a run still
            counts against a model that failed a case, but certifies nothing.
        gpu_fraction: The share of the model in GPU memory while it ran, 0.0
            to 1.0, when the runtime reported it. A partial split is ordinary
            on a small card and is not a fault.
        thinking_with_tools_passed: What a deep run ESTABLISHED about reasoning
            and calling tools in one turn, with no declaration folded in. True
            means a certificate holds, so storing a grant takes effect at once;
            False means a still-bound run measured the capability absent, so a
            grant would sit inert until a fresh check disagrees; null means
            nothing usable measured it here - no deep run, or one this
            deployment cannot count.

            Its own field rather than something a client derives, because the
            derivation a client would reach for is wrong. The effective
            capability on the row is False whenever nothing DECLARES it,
            however well the check went, so reading a failure out of it
            describes the ordinary earn-then-declare order backwards - a deep
            run that passed, waiting on a click, reported as a check that
            failed. Only the server can tell the two apart.
    """

    depth: str
    tokens_per_second: float | None = None
    vram_bytes: int | None = None
    measured_at: UtcDatetime
    runtime_version: str | None = None
    runtime_stale: bool = False
    harness_stale: bool = False
    endpoint_stale: bool = False
    representative: bool = True
    gpu_fraction: float | None = None
    thinking_with_tools_passed: bool | None = None


class WeightsChangeNote(BaseModel):
    """Tool-check verdicts the installed weights have outrun.

    A re-pull changes the digest, and every verdict measured on the previous
    weights stops counting - silently, because binding verdicts to weights is
    what makes them trustworthy. Silence is right for the resolution and wrong
    for a reader: a model's measured facts simply vanish, with an old failure
    clamp evaporating alongside them. This is the notice that says so.

    Derived, never stored. It is the probe log compared against the digest the
    tag points at right now, so it appears the moment weights change and clears
    itself the moment a check runs against the new ones - or the moment the tag
    comes back to weights a check already covered. There is no dismissal to
    record and nothing to clean up.

    Null means the notice does not apply - which includes the ordinary "nothing
    has ever been checked" case. That one is not a warning: it is what a fresh
    model looks like, and the row already says it in plain words.

    Attributes:
        depths: Every depth whose latest verdict was measured on other weights,
            weakest first ("fast" before "deep"). The last entry is the depth
            the two fields below describe - the fullest orphaned measurement of
            the set, on the same preference the measured facts use.
        earned: What that orphaned verdict earned, on the weights it actually
            measured. Reported so a reader can see what a re-check stands to
            recover; it is never a claim about the installed weights, and no
            part of the agency resolution reads it.
        measured_at: When that verdict was measured, always with an explicit
            UTC offset - it is rendered as a date beside other timestamps in
            this payload, and an offset-less stamp would be read as local time.
    """

    depths: list[str]
    earned: AgencyMode
    measured_at: UtcDatetime


class ThinkModeNote(BaseModel):
    """A disagreement about how this model treats the think flag.

    Three sources answer that question and they can contradict each other: what
    a tool check MEASURED on the installed weights, what the resolution chain
    DECLARES (a stored confirmation, the code-shipped seed, or the
    template inference), and what the runtime ADVERTISES about its own think
    plumbing. Agreement is the ordinary case and produces no note at all.

    Derived per request, never stored, exactly like
    :class:`WeightsChangeNote` - and for the same reason: the state it
    describes ends when a check runs against the installed weights and agrees,
    so there is no dismissal to record and nothing to clean up. A re-pull onto
    weights that behave differently is precisely what this catches, which is
    why it must not be a stored flag that could survive the thing it describes.

    Attributes:
        code: Which disagreement this is.

            - ``measured_mismatch`` - a check watched these weights do
              something the declaration does not say. The re-pull detector: a
              tag pointed at a checkpoint that reasons differently reads
              exactly like this.
            - ``nil_promotion`` - nothing has measured these weights, the
              declaration says they do not reason, and yet the runtime
              advertises think plumbing AND the model's own chat template
              carries it. A turn omits the flag entirely for a model declared
              never, and a serving stack that treats an absent flag as "on"
              then reasons anyway with nothing parsing the channel. It is a
              hazard rather than an observation, and a check that measures the
              installed weights clears it either way.
        measured: What the check measured, or null when nothing has.
        declared: What the resolution says WITHOUT the measurement - the
            declaration the note is disagreeing with.
        runtime_thinking: The runtime advertises think plumbing for this tag.
            A fact about the serving integration, never about the weights.
        message: The disagreement and what it means, in plain language.
        confirmable: A measurement bound to the installed weights exists, so
            confirming it would store a declaration that sticks. False on the
            hazard note, where there is nothing measured to confirm.
        routing_clamped: Turns are ALREADY sending ``think:true`` for this
            model on the strength of the measurement, without waiting for a
            confirmation. True only for a measured ``always``: that is the one
            direction where the alternative is raw reasoning markup in the
            answer, and where acting early can only ask a model to reason that
            was going to anyway.

            It has a SECOND consequence, and the message says so rather than
            leaving it to be discovered. A turn that will reason on a model
            without a ``thinking_with_tools`` grant drops its tools - reasoning
            and calling tools in the same turn fails as an empty answer on some
            model-and-runtime pairs, so the turn keeps the reasoning and takes
            the tools away. Nothing declares that grant by default, so this
            clamp also stops the model
            calling tools until the mode is confirmed and the grant earned. The
            turn is not left ungrounded by it: with no tool surviving, the
            deterministic pre-fetch and the ambient knowledge base run exactly
            as they did before tools existed.
    """

    code: Literal["measured_mismatch", "nil_promotion"]
    measured: ThinkMode | None = None
    declared: ThinkMode = "unknown"
    runtime_thinking: bool = False
    message: str
    confirmable: bool = False
    routing_clamped: bool = False


class AgencyReport(BaseModel):
    """How much this model may drive, and what is holding it back.

    ``effective`` is the CEILING of what a turn could run in, not a promise:
    a real turn folds in axes unknowable here (the request clamp, whether any
    tool survived the user's switches, whether the context window has room for
    a tool result), each of which can only clamp further DOWN.

    Attributes:
        ceiling: The ceiling for this model: a stored operator declaration when
            one is bound to the installed weights, else what the runtime's own
            report allows - ``assisted`` for a tool-taking template, and the
            safe floor otherwise.
        earned: The measured verdict, folded per the active regime. Reaching
            ``agentic`` takes a deep tool-check verdict bound to the installed
            weights in every regime; a declaration alone reaches ``assisted``.
        env_clamp: The ``AGENCY_MODE`` deployment clamp, or null for "auto".
        effective: The minimum over the axes above.
        clamp_code: Which axis to blame when ``effective`` is not ``agentic``;
            null when nothing clamps it. Most actionable first: a measured
            failure, then a chat template that cannot take a tool list at all,
            then a verdict discounted for naming another serving host, then "no
            check has proven this much", then the declaration, then the
            deployment clamp. The template sits above "not proven" because no
            check will ever lift it, so pointing a reader at one there is a
            dead end.

            ``endpoint_mismatch`` is a near-sibling of ``unverified_weights``
            and is separate for one reason: a passing verdict EXISTS for these
            weights, it was simply measured somewhere else, so the action is
            re-running the check here rather than running a first one.
        clamp_detail: A one-line human explanation of that blame, and nothing
            else. Null exactly when ``clamp_code`` is null - the two always
            agree, so a reader can treat either as the presence test. Why a
            row's Delete is unavailable is a different question with a different
            audience, and it lives in ``ManagerRow.delete_blocked_reason``.
    """

    ceiling: AgencyMode
    earned: AgencyMode
    env_clamp: AgencyMode | None = None
    effective: AgencyMode
    clamp_code: (
        Literal[
            "declared_ceiling",
            "uncatalogued",
            "failed_case",
            "unverified_weights",
            "endpoint_mismatch",
            "tools_none",
            "env_clamp",
        ]
        | None
    ) = None
    clamp_detail: str | None = None


class ManagerRow(BaseModel):
    """One model in the management view: declared facts joined to live state.

    Attributes:
        model: The exact tag as the provider knows it.
        provider: The serving backend this deployment is configured with.
        role: What the model can serve, derived from the runtime's report:
            ``chat``, ``embedding``, or empty when the report establishes
            neither.
        installed: The provider currently holds these weights. Always true
            today - the view is an inventory of what the server has - and kept
            as a field because a client should not have to infer from a row's
            presence what the row can state.
        installed_size_bytes: On-disk size the provider reports, when installed.
        digest: The weights the tag currently points at, when the runtime
            exposes one.
        num_ctx: The context window a turn on this tag would request - the
            smaller of the weights' native window and what this deployment
            affords. Resolved through the same helper the turn itself calls, so
            the number on the card is the number that runs.
        native_context_length: The window the WEIGHTS were trained for, as the
            runtime reports it. Reported beside ``num_ctx`` rather than
            instead of it, because the two answer different questions: this is
            the ceiling, that is what this deployment asks for - and a card
            that showed only the second cannot say whether a small window is a
            property of the model or a choice this machine made.
        quantization: How hard the installed weights were quantized, in the
            runtime's own wording ("Q4_K_M"). Null when nothing is installed
            or the runtime does not say.
        parameter_size: The parameter count as the runtime words it ("4.0B").
        family: The architecture family the runtime names ("qwen3").
        tools: Whether the chat template can take a tool list at all, as the
            runtime reports it - a mechanical fact only the serving stack can
            see, since it is the party that renders the template.
        think: How the model treats the think flag, strongest source first: an
            unconfirmed measurement of ``always`` (the one direction that acts
            before an operator agrees, because the alternative is reasoning
            markup in the visible answer), then a stored operator
            confirmation, then the code-shipped ledger of known model families,
            then the runtime's answer inferred from the chat template. Only
            that last one is a prior rather than a measurement, so it fills a
            gap and never overrules a declaration. This is what a TURN will
            actually send, so a disagreement between the sources shows up in
            ``think_mode`` rather than here.
        thinking_with_tools: The DECLARED claim that this model can reason and
            call tools in the same turn - a stored operator declaration, and
            nothing else. A declaration, not a measurement: what a probe found
            sits in ``measured``, and a turn will not act on this until one
            agrees. Uniquely among the capability columns it is NEVER inferred,
            because guessing it optimistically fails as silent empty output
            rather than as an error.
        thinking_with_tools_effective: What a TURN will actually do - the claim
            above folded against the deep verdict bound to these weights.
            Reported beside the declaration rather than instead of it because
            the pair is the honest thing to render: the claim alone over-states
            a granted-but-unproven model, and the fold alone gives a reader
            nothing to act on. Chat rows carry the resolved answer; every other
            role leaves it false, having no turn to run.
        thinking_with_tools_declared: The claim came from a stored operator
            declaration bound to the installed weights. The sibling of
            ``ceiling_declared`` and needed for the same reason: a console has
            to word its dialog around REPLACING a decision somebody made here
            rather than making a first one, and a re-pull retires it.
        thinking_model: The pair sibling that serves this tag's thinking turns,
            when one is set - the ``THINKING_CHAT_MODEL`` override where a
            deployment set one, else a stored operator declaration. The sibling
            has no row of its own; it is this entry's. This is the tag a TURN
            routes to, resolved the same way the turn resolves it, so a card
            cannot name a partner the chat path would ignore.
        thinking_model_declared: The sibling above came from a stored
            declaration rather than from the environment. Reported separately
            because the two are undone differently: a declared pair is changed
            from the console, an environment override from the deployment's
            configuration.
        vision: Whether this tag accepts images, as the runtime reports it. A
            tag is mutable, so nothing written down could ever have admitted
            pixels on its own - the measurement always had to agree, and now it
            is the whole answer.
        agency: The agency report; chat rows only (no other role drives a turn).
        measured: Probe measurements for the installed weights, or null for
            "unmeasured".
        weights_change: Verdicts a change of weights left behind, or null when
            none were. Distinct from ``measured``, which only ever describes
            the installed weights, and from that report's staleness flags,
            which are about the RUNTIME moving under a still-valid verdict -
            this one is about the weights themselves moving out from under it.
            Chat rows only, and only where the runtime exposes a digest to
            compare against.
        think_mode: A disagreement about how this model treats the think flag,
            or null when the sources agree - which is the ordinary case and not
            a warning. Distinct from ``think``, which is the single answer a
            turn acts on: this says the sources it was resolved from do not
            match, and which of them to believe.
        can_pull: This row can be pulled from here right now. True on an
            INSTALLED manageable row too: pulling it fetches whatever the
            registry now serves under the tag, which is how updates arrive.
        can_delete: This row can be deleted from here right now.
        checkable: A tool check can be run against this row right now - the
            weights are here, no declared role rules the check out, and where
            nothing declares a role the runtime says the model can hold a
            conversation. Mirrors what the verify endpoint actually refuses.
            Server-derived so that clients need not re-derive it: what may be
            done to a model depends on facts the server holds, and a client
            guessing from the role alone guesses wrongly the first time a
            server holds a chat model nothing has declared. The console
            still derives its own answer today and adopts this field in the
            follow-up slice.
        delete_blocked_reason: Why ``can_delete`` is false, when the reason is
            a POLICY one - something still depends on this model - rather than
            the mundane "it is not installed" or "this backend cannot delete".
            Deliberately its own field rather than a line appended to
            ``agency.clamp_detail``: that one answers "how much may this model
            drive a turn", which is a different question with a different
            audience, and fusing the two produced a sentence about buttons
            underneath a heading about autonomy. This is the text a disabled
            control explains itself with.
        pinned_conversations: How many of the tenant's conversations are SET to
            use this tag. Not a block - a delete clears those pins and the
            threads follow the workspace default - but the person deleting
            should read that consequence before committing to it, so the count
            rides on the row for the confirm dialog to say.
        workspace_default: This tag is the workspace's STORED default chat
            model. Deleting it requires naming a successor in the same request,
            so the dialog that offers the delete must offer that choice too.
        is_default: This tag is the one an unpinned turn ACTUALLY runs -
            resolved through the same chain a turn uses, so it is the stored
            workspace default where there is a usable one and the deployment
            binding otherwise. Exactly one row in a payload carries it.

            Distinct from ``workspace_default`` and not a duplicate of it: that
            one answers the delete path's question ("does removing this need a
            successor?") and is false on every row of a deployment where nobody
            has stored a default, even though one model plainly answers by
            default there. This one answers the grid's ("which model replies
            when nobody chose?"), which always has an answer.
        ceiling_declared: The ceiling in ``agency`` came from a stored operator
            declaration bound to the installed weights, rather than from the
            runtime's report. A console needs the difference to word its own
            dialog honestly: changing a declared ceiling replaces a decision
            somebody made here, and a re-pull retires it.
    """

    model: str
    provider: str
    role: str
    installed: bool = False
    installed_size_bytes: int | None = None
    digest: str | None = None
    num_ctx: int | None = None
    native_context_length: int | None = None
    quantization: str | None = None
    parameter_size: str | None = None
    family: str | None = None
    tools: ToolsMode = "none"
    think: ThinkMode = "unknown"
    thinking_with_tools: bool = False
    thinking_with_tools_effective: bool = False
    thinking_with_tools_declared: bool = False
    thinking_model: str | None = None
    thinking_model_declared: bool = False
    vision: bool = False
    agency: AgencyReport | None = None
    measured: MeasuredFacts | None = None
    weights_change: WeightsChangeNote | None = None
    think_mode: ThinkModeNote | None = None
    can_pull: bool = False
    can_delete: bool = False
    checkable: bool = False
    delete_blocked_reason: str | None = None
    pinned_conversations: int = 0
    workspace_default: bool = False
    is_default: bool = False
    ceiling_declared: bool = False


class RoleGroup(BaseModel):
    """The rows serving one role, and how much the UI may do to them.

    Attributes:
        role: The role these rows serve.
        management: ``manageable`` (pull and delete are offered),
            ``locked`` (deliberately not changeable from here), or
            ``status_only`` (not a model-provider model at all).
        lock_reason: Why a non-manageable group is not manageable.
        rows: The held tags serving this role, in the provider's own order.
    """

    role: str
    management: Literal["manageable", "locked", "status_only"]
    lock_reason: str | None = None
    rows: list[ManagerRow]


class RoleGap(BaseModel):
    """Something the server cannot do because of what it does not hold.

    A gap is about the INVENTORY and about what is SELECTED from it: it says
    what a deployment has lost, in terms of the features a user would notice
    losing. Both halves matter, because everything downstream of either fails
    quietly - uploads accept the file and never become searchable, knowledge
    lookups return nothing, and a web-grounded answer arrives ungrounded.

    Attributes:
        role: Which role the gap is about ("embedding", "chat").
        severity: ``fatal`` when a whole feature area is unavailable until
            somebody acts, ``note`` when the deployment works and could work
            better. Only the first is worth interrupting a page for. Both
            embedding gaps are fatal: retrieval embeds through the SELECTED
            model, so anything short of holding that model is a dead feature
            rather than a suboptimal one.
        code: The stable machine-readable reason, so a client can act on the
            distinction rather than parse the sentence. Each pair of codes
            separates gaps by WHAT WOULD FIX IT, which is the only distinction
            a reader can act on:

            - ``embedder_missing`` - a DOWNLOAD fixes it. Either the deployment
              selects an embedding model this server does not hold (other
              installed embedders do not soften that: nothing selects them), or
              nothing can embed and nothing is selected.
            - ``embedder_unselected`` - a CHOICE fixes it, and a download would
              not. Nothing selects an embedding model, and installed models
              could serve as one.
            - ``tools_incapable`` - no installed model's chat template can take
              a tool list, so nothing the model could drive is available to it.
              A different MODEL fixes it.
            - ``tools_clamped`` - models that CAN take tools are installed, and
              policy holds every one of them at ``deterministic``. A
              DECLARATION or a passing check fixes it; a download is wasted.
        message: The gap in plain language, naming what stops working and what
            would fix it.
        suggested_model: A tag that would close the gap, when exactly one is
            obviously right. Always a tag to DOWNLOAD - the embedding model the
            deployment already selected. Null whenever the fix is a choice
            rather than a download, because picking between several installed
            embedders would be a ranking, and this platform does not invent
            rankings.
    """

    role: str
    severity: Literal["fatal", "note"]
    code: Literal[
        "embedder_missing",
        "embedder_unselected",
        "tools_incapable",
        "tools_clamped",
    ]
    message: str
    suggested_model: str | None = None


class ModelNameRequest(BaseModel):
    """Body for the management verbs that act on a named model.

    A body rather than a query string because these are actions on a name, not
    filters over a collection, and because a pull will grow options (a forced
    re-pull, a mirror) that a body accommodates without changing the request's
    shape. Delete names its target in the path instead, as its method requires.

    The length cap is declared HERE as well as in the validator so it is
    enforced during parsing: without it the string is materialised in full and
    only then measured, and nothing else bounds a JSON body field.
    """

    name: str = Field(max_length=MAX_NAME_LENGTH)


class ModelPullRequest(ModelNameRequest):
    """Body for the download verb: the name, plus what to do once it lands.

    Attributes:
        verify: Run the FAST tool check on the tag as soon as the download
            succeeds, as a continuation of the same stream. On by default
            because a model nobody measured is a model whose agency ceiling is
            a guess - and the moment weights land is the one moment a check
            costs nothing extra to schedule. A caller pulling several models
            back to back, or one that will run a DEEP check itself, turns it
            off.
    """

    verify: bool = True


class ModelVerifyRequest(ModelNameRequest):
    """Body for the verify verb: which model to measure, and how thoroughly.

    Attributes:
        depth: ``fast`` runs the two load-bearing cases once each and can earn
            at most ``assisted``; ``deep`` runs every case three times and is
            the only depth that can earn ``agentic`` or ``thinking_with_tools``
            - and the only one that drives the pair's thinking tag. Fast is the
            default because deep costs minutes of GPU time.
    """

    depth: Literal["fast", "deep"] = "fast"


class ModelThinkModeResponse(BaseModel):
    """What confirming a measured think mode stored.

    The request carries only a model NAME: the value stored is the one a tool
    check measured on the weights installed right now, so there is no field
    through which a caller could declare a mode nothing observed. A model
    nothing has measured is refused (422) rather than declared.

    Attributes:
        model: The tag the declaration was recorded against, in the spelling
            the provider lists it under.
        think_mode: The mode that was stored.
        digest: The weights the declaration is bound to. It retires on its own
            when the tag moves past them - a declaration is a claim about
            weights, not about a name.
        probe_id: The measurement the declaration rested on.
    """

    model: str
    think_mode: Literal["always", "hybrid", "never"]
    digest: str
    probe_id: str


class ModelCeilingRequest(ModelNameRequest):
    """Body for declaring how much a model may be allowed to drive.

    Attributes:
        ceiling: The strongest agency mode an operator permits this model to
            ATTEMPT, ``agentic`` included (owner decision, 2026-08-08).

            Raising it is not the escalation it looks like, and the reason is
            structural: a declaration reaches a turn only through
            ``min(CEILING, EARNED)``, and ``agentic`` needs a passing DEEP
            certificate bound to the installed digest and measured on the
            serving endpoint before EARNED will reach it. Declaring it on an
            unproven model stores a durable permission that stays inert until a
            check earns it.

            One honest caveat, because an earlier draft of this text got it
            wrong: raising to ``assisted`` DOES move what a turn does with no
            measurement at all, through the reviewed-declaration grace floor.
            Lowering, in either direction, needs no evidence and applies to the
            next turn.
    """

    ceiling: Literal["deterministic", "assisted", "agentic"]


class ModelCeilingResponse(BaseModel):
    """What declaring an agency ceiling stored, and whether it changed anything.

    ``effective`` is reported alongside the stored value precisely because the
    two routinely differ: declaring ``agentic`` on a model no deep check has
    certified stores the permission and leaves the model running at
    ``assisted`` until one passes. A response that echoed only what was written
    would let a console claim a change that did not happen.

    Attributes:
        model: The tag the declaration was recorded against, in the spelling
            the provider lists it under.
        ceiling: The mode that was stored.
        digest: The weights the declaration is bound to. A re-pull past this
            digest retires it, and the model falls back to the runtime's report.
        earned: What measurement has actually established for these weights.
        effective: The minimum of the two, plus the deployment clamp - what a
            turn on this model may now assume.
    """

    model: str
    ceiling: AgencyMode
    digest: str
    earned: AgencyMode
    effective: AgencyMode


class ModelThinkingWithToolsRequest(ModelNameRequest):
    """Body for declaring whether a model may reason and call tools at once.

    Attributes:
        thinking_with_tools: True grants the claim, False revokes it.

            Granting is stored whatever any check has found, and reaches a turn
            only through the fold: a stored grant needs a passing DEEP verdict
            at the running harness version,
            bound to the pair's installed thinking weights and measured on the
            host being served from. Until one exists the grant is durable and
            inert, exactly as an ``agentic`` ceiling is.

            Revoking needs no evidence and applies to the next turn. The
            asymmetry is the point: an unearned grant fails as an EMPTY answer
            rather than as an error, so the safe direction has to be the cheap
            one.
    """

    thinking_with_tools: bool


class ModelThinkingWithToolsResponse(BaseModel):
    """What declaring a reasoning-with-tools claim stored, and what a turn does.

    ``effective`` is reported beside the stored value for the same reason the
    ceiling response reports it: the two routinely differ. Granting the claim on
    a pair no deep check has certified stores a durable permission and leaves
    every turn running without it. A response echoing only the write would let a
    console announce a capability the model was not given - and this is the axis
    where that lie is invisible, because the failure it hides behind is an empty
    answer rather than an error.

    Attributes:
        model: The tag the declaration was recorded against, in the spelling
            the provider lists it under.
        thinking_with_tools: The claim that was stored.
        digest: The weights the declaration is bound to. A re-pull past this
            digest retires it, and the model falls back to the runtime's report.
        effective: What a turn on this model will actually do - the declaration
            folded against what a deep check has measured on these weights.
    """

    model: str
    thinking_with_tools: bool
    digest: str
    effective: bool


class ModelThinkingPairRequest(ModelNameRequest):
    """Body for pairing a direct-answer model with a reasoning partner.

    Attributes:
        thinking_model: The installed tag that should serve ``name``'s thinking
            turns. Refused (422) unless its think mode already resolves to
            ``always`` or ``hybrid`` - a partner nothing has established can
            reason would produce a thinking toggle that quietly changes
            nothing.
    """

    thinking_model: str = Field(max_length=MAX_NAME_LENGTH)


class ModelThinkingPairResponse(BaseModel):
    """What declaring a thinking pair stored.

    No digest: this is a PREFERENCE axis. "Run these two tags together" says
    how this deployment wants a pair used, not what either checkpoint is, so it
    follows the tags across a re-pull instead of retiring with the weights.

    Attributes:
        model: The direct-answer tag, in the spelling the provider lists it
            under.
        thinking_model: The sibling that now serves its thinking turns.
    """

    model: str
    thinking_model: str


class ModelDefaultResponse(BaseModel):
    """What setting the workspace default model stored.

    Attributes:
        model: The normalized tag every conversation without a pin of its own
            now follows.
    """

    model: str


class ModelDeleteResponse(BaseModel):
    """What a delete actually did, beyond removing the weights.

    A delete is no longer only a removal: conversation pins on the tag are
    cleared in the same transaction, and deleting the workspace default writes
    the successor the caller named. Answering with the specifics rather than a
    bare 204 lets the page report what happened instead of re-deriving it from
    a refetch that races other tabs.

    Attributes:
        removed: The normalized tag whose weights were removed.
        unpinned_conversations: How many conversations had their pin on this
            tag cleared; those threads follow the workspace default now.
        new_default: The tag that took over as the workspace default, when this
            delete replaced it; null when the default was not involved.
    """

    removed: str
    unpinned_conversations: int = 0
    new_default: str | None = None


class VerifyProgressEvent(BaseModel):
    """One finished sample of a running check.

    Attributes:
        case: The harness case this sample ran.
        model: The exact tag it ran against - on a deep check the reasoning
            case drives the pair's thinking tag, so this is not constant.
        passed: Whether the exchange met the case's expectation.
        detail: One human-readable line saying why, pass or fail.
        latency_s: Wall time of the whole exchange, seconds.
        completed: Samples finished so far, including this one.
        total: Samples this depth will run in all, so a client can show
            progress without knowing the harness's case list.
    """

    case: str
    model: str
    passed: bool
    detail: str
    latency_s: float
    completed: int
    total: int


class VerifyResidencyReport(BaseModel):
    """Where the model actually sat while it was being measured.

    Attributes:
        gpu_resident: True when the weights were on the GPU, False when they
            spilled entirely to the CPU, and null when the backend does not
            report residency (or no longer held the model when it was asked).
            Null is "unknown", never "no".
        gpu_fraction: The share of the model in GPU memory, 0.0 to 1.0. A
            partial split is ordinary on a small card and is not a fault.
        vram_bytes: Bytes of the model in GPU memory.
        total_bytes: Bytes the loaded model occupied in all.
    """

    gpu_resident: bool | None = None
    gpu_fraction: float | None = None
    vram_bytes: int | None = None
    total_bytes: int | None = None


class VerifyDoneEvent(BaseModel):
    """The terminal frame of a check that ran: verdicts plus what was observed.

    Attributes:
        model: The chat tag the run drove.
        thinking_model: The pair's thinking tag, when the run had one to drive.
        depth: Which depth ran.
        earned: The agency verdict this run earned. The EARNED half of the
            two-key resolution only - the human-declared ceiling still clamps
            beside it, so nothing here can raise what a model may do.
        thinking_with_tools: True only when a deep run passed the
            reasoning-with-tools case on every sample.
        measured_think_mode: How the run watched the CHECKED tag treat the
            think flag - never the pair's thinking sibling, which the thinking
            cases deliberately do not drive. Null when the run could not tell
            (a provider error on either half of the pair of samples). A
            measurement, not a declaration: it becomes one only when an
            operator confirms it.
        tallies: Per-case ``[passed, run]`` counts.
        tokens_per_second: Decode throughput measured across the run. Null when
            nothing measurable was generated - and also when the run was not
            representative, because a CPU-bound decode rate is a fact about a
            spill, not about the deployment.
        harness_version: The case-spec fingerprint this run measured under.
        digest: The weights the tag pointed at, binding the verdict to them.
        runtime_version: The serving runtime the run measured under.
        residency: Where the model sat while it ran.
        representative: Whether this run describes how the deployment actually
            serves. False marks a run the model spilled off the GPU for: it is
            recorded and its failures still count, but it certifies nothing.
        warning: Why ``representative`` is false, or null.
        probe_id: The measurement row this run was recorded as, or null when
            the write failed - the measurement still happened and is reported
            here either way.
    """

    model: str
    thinking_model: str | None = None
    depth: Literal["fast", "deep"]
    earned: AgencyMode
    thinking_with_tools: bool = False
    measured_think_mode: Literal["always", "hybrid", "never"] | None = None
    tallies: dict[str, list[int]] = Field(default_factory=dict)
    tokens_per_second: float | None = None
    harness_version: str
    digest: str | None = None
    runtime_version: str | None = None
    residency: VerifyResidencyReport = Field(default_factory=VerifyResidencyReport)
    representative: bool = True
    warning: str | None = None
    probe_id: str | None = None


class VerifyErrorEvent(BaseModel):
    """The terminal frame of a check that could not be made.

    Attributes:
        message: A sentence safe to show a user; never exception text.
        code: Stable machine-readable reason (see the module docstring).
    """

    message: str
    code: Literal["model_not_found", "provider_unavailable", "internal_error"]


class EmbedderCheckRequest(BaseModel):
    """Body for the embedder check: which embedder to measure, if not the active one.

    Attributes:
        name: The tag to measure. Omitted or blank measures the embedder the
            resolution chain settles on, which is the case worth making the
            easy one - it is the only embedder whose bands the deployment's own
            relevance floors are supposed to describe.
    """

    name: str | None = Field(default=None, max_length=MAX_NAME_LENGTH)


class EmbedderScoreBand(BaseModel):
    """The range one family of similarity scores occupied.

    Attributes:
        count: How many scores went into the band. Zero leaves the rest null.
        minimum: The lowest score in it.
        maximum: The highest score in it.
        mean: The arithmetic mean.
    """

    count: int
    minimum: float | None = None
    maximum: float | None = None
    mean: float | None = None


class EmbedderTripletReport(BaseModel):
    """What one probe triplet scored.

    Attributes:
        name: The triplet's identifier.
        related: How the query scored against the passage that answers it.
        supporting: How it scored against the weaker true answer, when the
            triplet carries one - the passage the RELATIVE margin decides the
            fate of.
        best_unrelated: The highest score the query reached against any passage
            that does not answer it, anywhere in the corpus.
        best_unrelated_passage: Which passage that was, so a band that fails
            names the text to look at rather than only a number.
    """

    name: str
    related: float
    supporting: float | None = None
    best_unrelated: float
    best_unrelated_passage: str


class EmbedderArmReport(BaseModel):
    """One prefix arm's reading: the bands, the corridor between them, the detail.

    Attributes:
        related: The band genuinely relevant passages occupied.
        supporting: The band the weaker true answers occupied.
        unrelated: The band everything else occupied.
        separation: The width of the corridor a floor has to sit in - the
            weakest related score minus the strongest unrelated one. Negative
            means the bands overlap and no absolute floor separates them.
        triplets: The per-triplet detail behind the bands.
    """

    related: EmbedderScoreBand
    supporting: EmbedderScoreBand
    unrelated: EmbedderScoreBand
    separation: float
    triplets: list[EmbedderTripletReport] = Field(default_factory=list)


class EmbedderPrefixReport(BaseModel):
    """Whether the instruct wrapper earned its place on this embedder.

    Attributes:
        source: Which source supplied the prefix under test - the model-quirk
            seed, which records the wrapper the model card prescribes, or none.
        prefix: The wrapper itself, so a reader can see what was measured.
        wrapped_separation: The corridor the wrapped queries produced.
        raw_separation: The corridor the same queries produced unwrapped.
        delta: Wrapped minus raw.
        improves_separation: True when wrapping widened the corridor. False on
            an asymmetric embedder is a finding rather than a formality: the
            prefix and the floors are calibrated together, so a wrapper that
            does not help means the floors beside it describe a scale nobody is
            querying on.
        degenerate: True when there is no prefix, which makes the two arms one
            measurement and the delta necessarily zero.
    """

    source: Literal["known_models", "none"]
    prefix: str
    wrapped_separation: float
    raw_separation: float
    delta: float
    improves_separation: bool
    degenerate: bool


class EmbedderFloorCheck(BaseModel):
    """One floor measured against one band - a DIAGNOSTIC, never the verdict.

    Retrieval composes its two floors, so what a single floor does to a single
    band decides nothing on its own. These say WHERE a composed failure comes
    from; ``composed`` says whether there is one. A failing band check beside a
    composed gate that holds is an ordinary, healthy reading - it usually means
    the relative margin is doing the work.

    Attributes:
        name: Which property was measured.
        passed: Whether the measurement supports the declaration.
        detail: One line carrying the numbers behind the answer.
    """

    name: Literal[
        "min_score_admits_related",
        "min_score_excludes_unrelated",
    ]
    passed: bool
    detail: str


class EmbedderQueryGate(BaseModel):
    """What the composed floors would actually retrieve for one query.

    Attributes:
        name: The triplet's identifier.
        top_score: The best-scoring passage of any kind - what the relative
            margin measures down from.
        top_passage: Which passage that was. When it is not the real answer,
            something else is setting the margin's floor.
        effective_floor: The score a passage had to reach to survive both
            floors.
        related_retrieved: Whether the query's real answer came back. False is
            the worse failure: the deployment answers that question with no
            grounding and says nothing.
        supporting_retrieved: Whether the weaker true answer came back, or null
            when the triplet has none. Informational - pruning weak context is
            what a relative margin is for.
        noise_retrieved: Noise passages that survived both floors. Non-empty is
            a leak.
        noise_clearance: How far the strongest noise passage sat below the
            effective floor. Negative on a leak; a small positive number is a
            gate that holds today and would not survive much drift.
    """

    name: str
    top_score: float
    top_passage: str
    effective_floor: float
    related_retrieved: bool
    supporting_retrieved: bool | None = None
    noise_retrieved: list[str] = Field(default_factory=list)
    noise_clearance: float


class EmbedderComposedGate(BaseModel):
    """The verdict: what the two floors do when applied TOGETHER.

    Retrieval applies ``min_score`` as a coarse gate and then drops anything
    more than ``rel_margin`` below the top survivor, so neither floor ever acts
    alone. This simulates exactly that per query and answers the two questions
    a user of retrieval would notice - was the real answer retrieved, and did
    any noise come with it.

    Attributes:
        min_score: The absolute floor that was composed.
        rel_margin: The relative margin that was composed.
        queries: The per-query simulation, in corpus order.
        leaks: ``triplet: passage`` for every noise passage retrieved.
        related_lost: Triplets whose real answer the floors excluded.
        supporting_pruned: Triplets whose weaker true answer the margin pruned.
            A reported TRADE, never a failure.
        holds: True when nothing leaked and no real answer was lost.
        detail: One line summarising the above.
    """

    min_score: float
    rel_margin: float
    queries: list[EmbedderQueryGate] = Field(default_factory=list)
    leaks: list[str] = Field(default_factory=list)
    related_lost: list[str] = Field(default_factory=list)
    supporting_pruned: list[str] = Field(default_factory=list)
    holds: bool
    detail: str


class EmbedderFloorProposal(BaseModel):
    """Floors derived from the measured bands, for a human to apply or refuse.

    Attributes:
        min_score: The proposed absolute floor, or null when the bands overlap
            too far for any floor to separate them.
        rel_margin: The proposed relative margin, or null when nothing in the
            corpus measured a drop to derive one from.
        derivation: The rule, stated with the numbers it was applied to.
        validates: Whether the proposed pair HOLDS when composed - derived and
            then simulated by one code path, so a proposal can never be
            recommended on a rule the gate would reject.
        caveat: Why these numbers must not be copied into configuration
            unexamined. The probe corpus is purpose-written question-and-answer
            text and scores systematically higher than production prose, so a
            floor derived from it transplants downward badly. A separate field
            rather than a sentence buried in the derivation, so nothing can
            render the numbers without it.
        applied: Always false, and present to say so: floors are configuration
            (``RAG_MIN_SCORE`` / ``RAG_REL_MARGIN``), and nothing in the
            backend writes them from a measurement.
    """

    min_score: float | None = None
    rel_margin: float | None = None
    derivation: str
    validates: bool
    caveat: str = ""
    applied: Literal[False] = False


class EmbedderFloorsReport(BaseModel):
    """What the run concluded about this embedder's relevance floors.

    Attributes:
        declared_min_score: The absolute floor already declared for this tag.
        declared_rel_margin: The relative margin already declared for it.
        source: Where those came from, or ``none`` when nothing describes this
            embedder - which is exactly when a proposal is the whole answer.
        composed: What the declared pair would actually retrieve. Null when
            fewer than both floors are declared, since retrieval applies both.
        checks: Band diagnostics for the declared absolute floor. Empty when
            nothing is declared, and never the verdict.
        hold: The composed gate's answer, or null when there was nothing to
            compose. NOT derived from ``checks``.
        proposal: Present whenever ``hold`` is not true - an undeclared pair, a
            half-declared one, or a composed gate that failed.
    """

    declared_min_score: float | None = None
    declared_rel_margin: float | None = None
    source: Literal["known_models", "none"]
    composed: EmbedderComposedGate | None = None
    checks: list[EmbedderFloorCheck] = Field(default_factory=list)
    hold: bool | None = None
    proposal: EmbedderFloorProposal | None = None


class EmbedderConfiguredFloors(BaseModel):
    """How the floors this deployment RETRIEVES with fare against the bands.

    Distinct from the declared pair on purpose. What a model-quirk seed
    declares is a statement about the model; ``RAG_MIN_SCORE`` and
    ``RAG_REL_MARGIN`` are what retrieval actually applies, and the two coincide
    only until somebody edits an environment file. Reporting the declared
    verdict alone could answer "the floors hold" while the numbers in force sat
    above the related band and dropped true answers on every turn, which is the
    exact failure this check exists to catch.

    Attributes:
        min_score: The absolute floor in force in this process.
        rel_margin: The relative margin in force in this process.
        composed: What these two, applied together, would actually retrieve -
            the verdict, simulated exactly as retrieval composes them.
        checks: Band diagnostics for the same floor. Informational.
        hold: The composed gate's answer.
        matches_declared: Whether these are the declared numbers. When true the
            two verdicts are one measurement reported twice; when false, this
            is the one that describes what the deployment does.
    """

    min_score: float
    rel_margin: float
    composed: EmbedderComposedGate
    checks: list[EmbedderFloorCheck] = Field(default_factory=list)
    hold: bool
    matches_declared: bool


class EmbedderCheckResponse(BaseModel):
    """What one embedder check measured, and what it recommends.

    Attributes:
        model: The embedder tag the run drove.
        active: Whether that tag is the embedder this deployment queries with.
            False means the reading describes a candidate, and the floors
            verdict is advice about switching rather than about today.
        digest: The weights the tag pointed at, binding the bands to them - a
            re-pull changes it, and these bands correctly stop describing what
            is installed.
        runtime_version: The serving runtime the run measured under.
        harness_version: The corpus fingerprint this run measured under.
        prefix: The wrapped-versus-raw comparison.
        wrapped: The arm run with the query prefix applied - the one live
            retrieval produces, and therefore the one the floors are judged on.
        raw: The same corpus with unwrapped queries, for the comparison.
        floors: What the DECLARED floors for this embedder do against the
            bands - the validation, the proposal, or both.
        configured: What the floors actually in force do against the same
            bands. Read this one to answer "is my deployment retrieving
            correctly"; read ``floors`` to answer "is what the repo believes
            about this model still true".
        probe_id: The measurement row this run was recorded as, or null when
            the write failed - the measurement happened either way.
    """

    model: str
    active: bool
    digest: str | None = None
    runtime_version: str | None = None
    harness_version: str
    prefix: EmbedderPrefixReport
    wrapped: EmbedderArmReport
    raw: EmbedderArmReport
    floors: EmbedderFloorsReport
    configured: EmbedderConfiguredFloors
    probe_id: str | None = None


class ModelPreflightResponse(BaseModel):
    """What the registry says about a tag, asked on demand and never guessed.

    Attributes:
        name: The validated name, normalized (whitespace trimmed).
        kind: Which naming scheme the name follows.
        status: ``available`` (the registry has it), ``not_found`` (the
            registry answered and does not), or ``unknown``. Only a real answer
            from the registry ever produces the first two - a failed lookup is
            ``unknown``, because presenting a network blip as a missing model
            would send someone to correct a spelling that was already right.
        download_size_bytes: Exact bytes the download would transfer, summed
            from the registry manifest. Null whenever ``status`` is not
            ``available`` - never an estimate, and never a figure carried over
            from a similarly-named tag.
        quantization: How hard the weights were quantized, in the registry's
            own wording ("Q4_K_M"), or null where it publishes none. Reported
            beside the size because it is what makes the size mean something:
            two tags of one model can be a gigabyte apart on quantization
            alone, and choosing between them on bytes is choosing blind.
        parameter_size: The parameter count as the registry words it ("8.2B"),
            or null. The other half of the same judgement - a size with no
            parameter count beside it cannot be weighed against the card it has
            to fit on.

            Both are read from the config blob the manifest points at, which is
            a convention of the model registry rather than part of the OCI
            spec. A registry that publishes neither still answers the size
            question completely, and neither is ever inferred from the tag
            string, where a "q4_K_M" in a name is a naming habit rather than a
            declaration.
        warning: A caveat to show alongside the answer, or null.
        detail: Why the answer is what it is, when that needs saying (nothing
            found, nothing checkable); null on a clean ``available``.
    """

    name: str
    kind: Literal["library", "huggingface"]
    status: Literal["available", "not_found", "unknown"]
    download_size_bytes: int | None = None
    quantization: str | None = None
    parameter_size: str | None = None
    warning: str | None = None
    detail: str | None = None


class PullProgressEvent(BaseModel):
    """One ``progress`` frame of a download stream.

    Attributes:
        status: The backend's own progress wording, for display.
        completed: Bytes transferred so far, while a blob is transferring.
        total: Bytes in the blob being transferred, when known.
        percent: ``completed`` over ``total`` as a percentage, one decimal.
            Null whenever either byte count is absent - a stage with no
            transfer behind it has no progress to report, and inventing one
            would make the bar jump backwards on the next blob.
        digest: The blob being transferred, when the backend names it.
    """

    status: str
    completed: int | None = None
    total: int | None = None
    percent: float | None = None
    digest: str | None = None


class PullDoneEvent(BaseModel):
    """The terminal frame of a download that succeeded.

    Terminal for the DOWNLOAD: when ``verifying`` is true the connection stays
    open and the automatic check's frames follow (see the module docstring).
    Sending this frame first is what keeps the two outcomes separate - the
    download is reported successful before anything can go wrong in the check.

    Attributes:
        model: The tag that was downloaded.
        status: The last progress wording the backend sent.
        verifying: A tool check follows on this same stream.
        verify_skipped_reason: Why no check follows, when the caller might
            reasonably have expected one; null when they asked for none.
    """

    model: str
    status: str
    verifying: bool = False
    verify_skipped_reason: str | None = None


class PullErrorEvent(BaseModel):
    """The terminal frame of a download that failed.

    Attributes:
        message: A sentence safe to show a user; never exception text.
        code: Stable machine-readable reason (see the module docstring).
    """

    message: str
    code: Literal["model_not_found", "provider_unavailable", "internal_error"]


class ModelManagerResponse(BaseModel):
    """The whole management view: one request-time join, nothing stored.

    An INVENTORY: one row per tag the model server actually holds, and nothing
    else. There is no second list of models a deployment might have downloaded
    but did not - a row that names weights nobody has is a row whose every
    verb, measurement and verdict is empty, and a page full of them buries the
    ones a reader came for.

    Attributes:
        provider: The serving backend and its management verbs.
        groups: Role groups, each with its rows. A tag whose runtime report
            establishes neither chat nor embedding lands in the group with the
            empty role - described from what the report DID say, and running at
            the safe agency floor until an operator declares otherwise.
        vectors: Whether stored vectors are still reachable by the configured
            embedding model - the other half of the embedding role's lock,
            which prevents a swap from HERE but cannot undo one already made.
            Null when the view was built without a vector store to ask.
        role_gaps: What this server's inventory cannot do, worst first. Empty
            is the healthy answer and the common one. Reported on the response
            rather than on a row because a gap is the absence of a row: there
            is no embedder card to hang "there is no embedder" on. The console
            renders these in the follow-up slice; the data ships first so the
            two can land separately.
    """

    provider: ProviderInfo
    groups: list[RoleGroup]
    vectors: StrandedVectorsReport | None = None
    role_gaps: list[RoleGap] = Field(default_factory=list)
