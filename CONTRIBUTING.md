# Contributing Guide

## Development Workflow

All development must follow this workflow:

1. Pull the latest `main` branch.
2. Create a feature branch.
3. Implement the assigned task.
4. Run relevant tests.
5. Update documentation if required.
6. Commit changes.
7. Push the feature branch.
8. Create a Pull Request.
9. Request review.
10. Resolve review comments.
11. Merge into `main` after approval.
12. Update the relevant task status and handoff documentation.

## Branch Naming

Use:

- `feature/m1-dataset`
- `feature/m2-annotation`
- `feature/m3-vector-db`
- `feature/m4-rag-evaluation`
- `feature/m5-fastapi`
- `feature/m6-testing-deployment`

For smaller changes:

- `fix/<description>`
- `docs/<description>`
- `test/<description>`

## Commit Convention

Use:

- `feat:` — new functionality
- `fix:` — bug fix
- `docs:` — documentation
- `test:` — tests
- `refactor:` — code restructuring
- `chore:` — maintenance/configuration

Examples:

feat(dataset): implement canonical dataset schema

feat(vector-db): implement ChromaDB ingestion

feat(rag): implement calibration retrieval

feat(evaluation): implement C1-C6 scoring

feat(api): add evaluation endpoint

test(rag): add retrieval tests

docs(api): update API documentation

## Pull Requests

Every feature branch must be submitted through a Pull Request.

Do not directly push development work to `main`.

At least one team member should review the Pull Request before merging.

## Secrets

Never commit:

- `.env`
- API keys
- passwords
- private credentials
- access tokens
- private certificates

Use `.env.example` to document required environment variables.
