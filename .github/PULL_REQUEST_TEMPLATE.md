<!-- Title must follow Conventional Commits: type(scope): subject -->

## What & why
<!-- What does this change and why. Link the issue: Closes #NN -->

## Roadmap phase
<!-- e.g. Phase 3 - RAG subsystem -->

## Checklist
- [ ] Conventional Commit title (`type(scope): subject`)
- [ ] `make up` still works; app runs (`make up PROFILE=cpu` without a GPU)
- [ ] Tests added/updated and green (`make backend-test` / `make frontend-test`)
- [ ] Lint/format/type-check pass
- [ ] **Dependency audit clean** (if JS deps changed - `npm audit`)
- [ ] Touches the tenant/auth or MCP/secrets boundary? → ran a security review
- [ ] Docs updated if decisions/behavior changed
