"""Curated integration presets: prefill values for well-known MCP servers.

A preset is pure PREFILL - the same add-server form and connect form, with the
provider facts already typed in. Nothing here grants anything: the operator
still registers their own OAuth app with the provider, pastes its client id
and secret, and approves the consent screen. The facts below were verified
against primary provider documentation and live endpoint probes (2026-07-21
for GitHub and Gmail, 2026-07-22 for Google Drive, Slack, and Asana); a
preset also carries the honest caveats that decide whether it fits a given
deployment, because a prefill that hides a provider's conditions is a trap.

Static by design: presets are curated in code review, not stored in the
database - there is no write path, and a deployment edits them by editing this
module.
"""

from __future__ import annotations

from app.models import McpAuthType, McpCredentialScope
from app.schemas.integrations import (
    IntegrationOAuthEndpoints,
    IntegrationOAuthIdentity,
    IntegrationPresetOAuth,
    IntegrationPresetOut,
    IntegrationPresetServer,
)

_GITHUB = IntegrationPresetOut(
    id="github",
    name="GitHub",
    description="GitHub's hosted MCP server: repositories, issues, pull requests, actions.",
    server=IntegrationPresetServer(
        name_suggestion="github",
        base_url="https://api.githubcopilot.com/mcp/",
        auth_type=McpAuthType.OAUTH,
        credential_scope=McpCredentialScope.WORKSPACE,
    ),
    oauth=IntegrationPresetOAuth(
        # No endpoints: GitHub's server is MCP-native - discovery finds the
        # authorization server (and its RFC 8414 metadata) from the server
        # itself, verified live against both well-known documents.
        scopes=["repo", "read:org", "read:user"],
        endpoints=None,
        # GitHub has no OIDC userinfo; the REST profile call's login field is
        # the account label, present regardless of scope.
        identity=IntegrationOAuthIdentity(
            userinfo_endpoint="https://api.github.com/user",
            label_field="login",
        ),
        extra_authorize_params={},
    ),
    registration_url="https://github.com/settings/apps/new",
    setup_notes=[
        "Register a GitHub App, not a classic OAuth App: only GitHub App "
        "tokens come with a refresh token, so only they can renew "
        "automatically (classic OAuth App tokens never expire and never "
        "refresh).",
        "Keep 'Expiring user access tokens' enabled (the default) and set "
        "the callback URL to this deployment's redirect URI.",
        "A GitHub App ignores the scope list at sign-in: what the "
        "connection may touch is fixed by the App's own permissions, chosen "
        "when it is registered. It also needs no extra sign-in parameters - "
        "the refresh token comes with expiring tokens automatically.",
        "If Connect fails at the provider, tick the manual-endpoints option "
        "and use: issuer https://github.com/login/oauth, authorization "
        "https://github.com/login/oauth/authorize, token "
        "https://github.com/login/oauth/access_token.",
    ],
)

_GMAIL = IntegrationPresetOut(
    id="gmail",
    name="Gmail",
    description="Google's Gmail MCP server (developer preview): search, read, label, draft.",
    server=IntegrationPresetServer(
        name_suggestion="gmail",
        base_url="https://gmailmcp.googleapis.com/mcp/v1",
        auth_type=McpAuthType.OAUTH,
        credential_scope=McpCredentialScope.WORKSPACE,
    ),
    oauth=IntegrationPresetOAuth(
        scopes=[
            "https://www.googleapis.com/auth/gmail.readonly",
            "https://www.googleapis.com/auth/gmail.compose",
            "openid",
            "email",
        ],
        # Google's servers are not MCP-native: the endpoints are prefilled
        # from Google's own OIDC discovery document.
        endpoints=IntegrationOAuthEndpoints(
            issuer="https://accounts.google.com",
            authorization_endpoint="https://accounts.google.com/o/oauth2/v2/auth",
            token_endpoint="https://oauth2.googleapis.com/token",
            revocation_endpoint="https://oauth2.googleapis.com/revoke",
        ),
        # Standard OIDC userinfo; the email claim needs the openid + email
        # scopes above.
        identity=IntegrationOAuthIdentity(
            userinfo_endpoint="https://openidconnect.googleapis.com/v1/userinfo",
            label_field="email",
        ),
        # Google issues a refresh token ONLY when asked via these non-standard
        # parameters; without them the connection dies when the first access
        # token expires. Prefilling them VISIBLY (editable in the form) is the
        # owner-approved reading of the no-silent-standing-access guardrail.
        extra_authorize_params={"access_type": "offline", "prompt": "consent"},
    ),
    registration_url="https://console.cloud.google.com/apis/credentials",
    setup_notes=[
        "Create an OAuth client of type 'Web application' in Google Cloud "
        "Console, add this deployment's redirect URI to it, and enable the "
        "Gmail API for the project.",
        "The Gmail scopes here are in Google's RESTRICTED tier. Inside a "
        "Google Workspace organization, set the OAuth consent screen to "
        "'Internal' and no Google review is needed. Without a Workspace "
        "org, 'Testing' mode works for up to 100 listed users with a "
        "warning screen - and its refresh tokens expire every 7 DAYS, so "
        "connections need weekly reconnects. Public production use requires "
        "Google's paid security assessment.",
        "The access_type=offline and prompt=consent entries ask Google for "
        "a renewal (refresh) token so the connection can renew itself; "
        "remove them to grant one-hour access only.",
        "Google's Gmail MCP server is a developer preview: joining the "
        "Google Workspace Developer Preview Program may be required, and "
        "its terms can change.",
    ],
)

_GDRIVE = IntegrationPresetOut(
    id="gdrive",
    name="Google Drive",
    description="Google's Drive MCP server (developer preview): search, read, and manage files.",
    server=IntegrationPresetServer(
        name_suggestion="gdrive",
        base_url="https://drivemcp.googleapis.com/mcp/v1",
        auth_type=McpAuthType.OAUTH,
        credential_scope=McpCredentialScope.WORKSPACE,
    ),
    oauth=IntegrationPresetOAuth(
        scopes=[
            "https://www.googleapis.com/auth/drive.file",
            "https://www.googleapis.com/auth/drive.readonly",
            "openid",
            "email",
        ],
        # Same Google authorization server as the Gmail preset: the endpoints
        # come from Google's OIDC discovery document, not from the MCP server.
        endpoints=IntegrationOAuthEndpoints(
            issuer="https://accounts.google.com",
            authorization_endpoint="https://accounts.google.com/o/oauth2/v2/auth",
            token_endpoint="https://oauth2.googleapis.com/token",
            revocation_endpoint="https://oauth2.googleapis.com/revoke",
        ),
        identity=IntegrationOAuthIdentity(
            userinfo_endpoint="https://openidconnect.googleapis.com/v1/userinfo",
            label_field="email",
        ),
        extra_authorize_params={"access_type": "offline", "prompt": "consent"},
    ),
    registration_url="https://console.cloud.google.com/apis/credentials",
    setup_notes=[
        "Create an OAuth client of type 'Web application' in Google Cloud "
        "Console, add this deployment's redirect URI to it, and enable BOTH "
        "the Google Drive API and the Google Drive MCP API "
        "(drive.googleapis.com and drivemcp.googleapis.com) for the project.",
        "The drive.readonly scope is in Google's RESTRICTED tier, with the "
        "same conditions as the Gmail preset (Internal Workspace org: no "
        "review; Testing mode: 100 users and 7-day refresh tokens; public "
        "production: Google's paid security assessment). The drive.file "
        "scope alone is non-sensitive - dropping drive.readonly avoids "
        "review at the cost of reaching only files the connection creates "
        "or is explicitly handed.",
        "The access_type=offline and prompt=consent entries ask Google for "
        "a renewal (refresh) token so the connection can renew itself; "
        "remove them to grant one-hour access only.",
        "Google's Drive MCP server is a developer preview: joining the "
        "Google Workspace Developer Preview Program may be required, and "
        "its terms can change.",
    ],
)

_SLACK = IntegrationPresetOut(
    id="slack",
    name="Slack",
    description="Slack's hosted MCP server: search, read, and post in a workspace.",
    server=IntegrationPresetServer(
        name_suggestion="slack",
        base_url="https://mcp.slack.com/mcp",
        auth_type=McpAuthType.OAUTH,
        credential_scope=McpCredentialScope.WORKSPACE,
    ),
    oauth=IntegrationPresetOAuth(
        # A starter set of USER scopes, each confirmed against the server's
        # published authorization metadata; the field is editable.
        scopes=[
            "search:read.public",
            "channels:history",
            "users:read",
            "users:read.email",
            "chat:write",
            "files:read",
            "canvases:read",
        ],
        # No endpoints: Slack's server is MCP-native - discovery finds its
        # authorization server (RFC 9728 + RFC 8414, both verified live).
        endpoints=None,
        # Slack's identity check; the user field names the signed-in account.
        identity=IntegrationOAuthIdentity(
            userinfo_endpoint="https://slack.com/api/auth.test",
            label_field="user",
        ),
        extra_authorize_params={},
    ),
    registration_url="https://api.slack.com/apps",
    setup_notes=[
        "Create a Slack app for the workspace and add this deployment's "
        "redirect URI. The app must ALSO declare a bot user with at least "
        "one bot scope (users:read works) in its own settings - Slack "
        "refuses the user sign-in without one, even though the bot is never "
        "used here.",
        "The scope list is a starter set of user scopes; edit it to match "
        "what the deployment should reach (private channels and direct "
        "messages need their own history scopes).",
        "Slack issues renewable (expiring) tokens only when token rotation "
        "is enabled in the app's settings - a one-way switch. Without "
        "rotation the connected token simply never expires, so there is "
        "nothing to renew.",
        "The sign-in sends Slack the standard scope parameter; if the "
        "consent screen rejects the requested scopes, trim the list and "
        "retry.",
    ],
)

_ASANA = IntegrationPresetOut(
    id="asana",
    name="Asana",
    description="Asana's hosted MCP server: tasks, projects, comments, and search.",
    server=IntegrationPresetServer(
        name_suggestion="asana",
        base_url="https://mcp.asana.com/v2/mcp",
        auth_type=McpAuthType.OAUTH,
        credential_scope=McpCredentialScope.WORKSPACE,
    ),
    oauth=IntegrationPresetOAuth(
        # Asana's single full-access scope; granular scopes are rejected for
        # MCP apps ("Invalid scope(s) requested").
        scopes=["default"],
        # No endpoints: Asana's server is MCP-native - its metadata declares
        # the resource identifier .../v2, which the flow binds tokens to.
        endpoints=None,
        # /users/me answers inside a data envelope; the dotted field walks it.
        identity=IntegrationOAuthIdentity(
            userinfo_endpoint="https://app.asana.com/api/1.0/users/me",
            label_field="data.name",
        ),
        extra_authorize_params={},
    ),
    registration_url="https://app.asana.com/0/my-apps",
    setup_notes=[
        "Register an app in Asana's developer console (choose the MCP app "
        "type) and add this deployment's redirect URI. A client secret is "
        "REQUIRED - Asana refuses the token exchange without one.",
        "Asana's only OAuth scope is 'default' (full account access as the "
        "connected user); granular scopes are rejected for MCP apps, so "
        "the scope field should stay as prefilled.",
        "Access tokens last one hour and come with a refresh token by "
        "default; the platform renews them automatically before use.",
        "On Enterprise plans, workspace admins can allow or block MCP "
        "clients - a blocked workspace fails at Connect, not at "
        "registration.",
    ],
)

# Display order. Curated: additions land through code review with their facts
# verified against the provider's primary documentation, never scraped.
PRESETS: tuple[IntegrationPresetOut, ...] = (_GITHUB, _GMAIL, _GDRIVE, _SLACK, _ASANA)
