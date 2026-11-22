"""Typed application errors mapped to RFC 7807 problem+json responses.

Services and repositories raise these instead of ``HTTPException`` so business
code stays HTTP-agnostic. ``app.main`` registers a MINIMAL handler that turns
any :class:`AppError` into a problem+json body ``{"type": "about:blank",
"title", "status", "detail", "code"}``; the cross-cutting middleware unit
extends that handler with ``request_id`` and structured logging - keep the
handler in ``main.py`` small and obvious so that extension stays trivial.
"""

from __future__ import annotations


class AppError(Exception):
    """Base class for errors that map to a problem+json HTTP response.

    Attributes:
        status: HTTP status code for the response.
        code: Stable machine-readable code (e.g. "not_found") for clients.
        title: Short human-readable summary, constant per error class.
        detail: Instance-specific explanation, safe to show to a user.
    """

    status: int = 500
    code: str = "internal_error"
    title: str = "Internal Server Error"

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


class NotFoundError(AppError):
    """The requested resource does not exist (or belongs to another tenant)."""

    status = 404
    code = "not_found"
    title = "Not Found"


class AttachmentNotFoundError(NotFoundError):
    """An attachment id names nothing in this conversation.

    A NotFoundError (still HTTP 404) with a distinct code, so a client can tell
    this apart from a missing conversation. The two look identical over the wire
    otherwise, and they call for opposite reactions: a missing conversation means
    the pane is bound to something deleted and should unbind, while a missing
    attachment means one stale chip on a live conversation, which should be
    dropped without disturbing the thread.
    """

    code = "attachment_not_found"
    title = "Attachment Not Found"


class MessageNotFoundError(NotFoundError):
    """A message id names nothing in this conversation.

    The same reasoning as :class:`AttachmentNotFoundError`, for the turns that
    address a stored message by id (editing a question, regenerating a named
    reply). Without a code of its own it is indistinguishable from a missing
    CONVERSATION, and the two call for opposite reactions: a missing conversation
    means the pane is bound to something deleted and should unbind, while a
    missing message means one stale row in a live thread - which happens the
    moment a second tab rewinds it - and unbinding there would detach the pane
    from a conversation that is perfectly alive.
    """

    code = "message_not_found"
    title = "Message Not Found"


class BadRequestError(AppError):
    """The request is syntactically valid but semantically wrong."""

    status = 400
    code = "bad_request"
    title = "Bad Request"


class UnprocessableEntityError(AppError):
    """The request parsed cleanly but a value in it is not acceptable.

    Distinct from :class:`BadRequestError` so a caller can tell "this field is
    wrong" from "this request does not make sense here", and it carries the
    same ``validation_error`` code as the framework's own body-validation
    failures - a client should not have to learn two codes for one 422 just
    because the check that rejected the value lives in a service rather than
    in a schema.
    """

    status = 422
    code = "validation_error"
    title = "Validation Error"


class AuthRequiredError(AppError):
    """No usable session: the caller must sign in before this can be answered.

    Only ever raised where login is CONFIGURED - a deployment with no issuer set
    acts as the seeded principal and never reaches this. The ``auth_required``
    code is the contract the frontend keys on: one shared response check turns
    any reply carrying it into a hop to the sign-in page, so a session that
    expires mid-use lands somewhere honest rather than on a broken screen.

    Distinct from :class:`ForbiddenError` for the reason those two statuses
    exist: 401 says "we do not know who you are, sign in and try again" and 403
    says "we know exactly who you are, and no". Collapsing them would send an
    admitted member to the login page every time they touched an admin surface.
    """

    status = 401
    code = "auth_required"
    title = "Authentication Required"


class ForbiddenError(AppError):
    """The caller is authenticated but not allowed to perform this action.

    Real once login is configured: ``get_current_tenant`` then carries the role
    from the signed-in person's row, so the ``require_role`` gate this backs
    genuinely refuses a member. With no issuer configured every request is the
    seeded admin and the gate always passes, which is the pre-login behaviour
    preserved deliberately. Distinct from 404: a forbidden action on a resource
    the caller may not even see still answers 404 (the repository's tenant
    filter), so this is for role gates, not ownership.
    """

    status = 403
    code = "forbidden"
    title = "Forbidden"


class ConflictError(AppError):
    """The request conflicts with existing state (e.g. a duplicate name)."""

    status = 409
    code = "conflict"
    title = "Conflict"


class AttachmentNotReadyError(ConflictError):
    """The attachment exists but its bytes are not servable yet.

    A ConflictError (still HTTP 409) with a distinct code, for the same reason
    :class:`AttachmentNotFoundError` has one: a client that polls an upload
    through ``pending`` and ``parsing`` needs "come back in a moment" to look
    different from "this request does not make sense here", because only the
    first one is worth retrying.
    """

    code = "attachment_not_ready"
    title = "Attachment Not Ready"


class GoneError(AppError):
    """The resource existed but is deliberately no longer served.

    Distinct from :class:`NotFoundError`: 404 says nothing ever addressed this,
    while 410 says something did and the answer will not change on a retry. The
    difference is what lets a client replace a stale affordance with an honest
    explanation instead of quietly dropping it.
    """

    status = 410
    code = "gone"
    title = "Gone"


class AttachmentExpiredError(GoneError):
    """The attachment's bytes aged out of retention.

    Its own code because this is the one 410 a client acts on differently: the
    row is still there, the file is not, and a file the assistant generated can
    simply be asked for again. Anything less specific would have the UI offering
    a retry for a file that was never coming back.
    """

    code = "attachment_expired"
    title = "Attachment Expired"


class AttachmentUnreadableError(GoneError):
    """The attachment has no trustworthy bytes to serve, and never will.

    Narrower than it once was. A failed INGESTION no longer raises this: the
    bytes of an upload or a paste are written before its row is committed and
    no failure path removes them, so a file a person sent is still theirs to
    fetch even when the platform could not read it.

    What remains are the two cases with nothing behind them. A file the
    ASSISTANT generated that somehow reached ``failed`` has no original - it
    exists only because the platform finished producing it - and a row carrying
    a status this build does not recognise is refused rather than served on the
    strength of not being understood.

    Separated from the expiry code so a client does not offer "ask for it again"
    over a file that never came into being, and so both stay distinguishable
    from :class:`AttachmentNotReadyError`, which is the one worth retrying.
    """

    code = "attachment_unreadable"
    title = "Attachment Unreadable"


class UnsupportedMediaTypeError(AppError):
    """The uploaded file is a type no parser can read."""

    status = 415
    code = "unsupported_media_type"
    title = "Unsupported Media Type"


class PayloadTooLargeError(AppError):
    """The request body exceeds the allowed size (e.g. an oversized upload)."""

    status = 413
    code = "payload_too_large"
    title = "Payload Too Large"


class TooManyRequestsError(AppError):
    """A per-tenant limit (e.g. concurrent streams) was exceeded."""

    status = 429
    code = "too_many_requests"
    title = "Too Many Requests"


class ServiceUnavailableError(AppError):
    """A required backing service (model provider, DB) is unreachable."""

    status = 503
    code = "service_unavailable"
    title = "Service Unavailable"
