"""Service layer: orchestration between the HTTP routers and the providers.

Routers stay thin (HTTP/SSE mapping only); services own prompt assembly, policy
decisions, and driving the provider seams.
"""
