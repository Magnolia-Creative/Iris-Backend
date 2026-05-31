# AGENTS.md

## Cursor Cloud specific instructions

### Workspace layout

This cloud workspace is a **multi-repo** setup. This repo is `Iris-Backend` (Python FastAPI). The iOS app lives in sibling repo `Iris-Main` under the same `repos/` parent.

### Infrastructure (not in the update script)

Start Postgres (pgvector) and Redis once per VM session:

```bash
sudo docker rm -f iris-postgres iris-redis 2>/dev/null
sudo docker run -d --name iris-postgres \
  -e POSTGRES_USER=iris -e POSTGRES_PASSWORD=iris -e POSTGRES_DB=iris \
  -p 5432:5432 pgvector/pgvector:pg16
sudo docker run -d --name iris-redis -p 6379:6379 redis:7-alpine
```

### Environment

Create `.env` in this directory (not committed) with at least:

- `DATABASE_URL=postgresql+asyncpg://iris:iris@127.0.0.1:5432/iris`
- `REDIS_URL=redis://127.0.0.1:6379/0`
- Optional: `SEMANTIC_INDEXING_ENABLED=false` for local dev without Gemini

### Python

Use `.venv` with `PYTHONPATH=.` from this directory. The `pyproject.toml` package is not setuptools-discoverable; the VM update script installs runtime deps via `pip` from the dependency list in `pyproject.toml`, not `pip install -e .`.

### Schema

On a fresh DB, `Base.metadata.create_all` (runs on app startup) plus `alembic stamp head` is the practical local bootstrap. Running `alembic upgrade head` alone on an empty database can fail because early revisions assume existing tables.

### Run API

```bash
export PYTHONPATH=.
.venv/bin/uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

### Lint / test

```bash
export PYTHONPATH=.
.venv/bin/ruff check .
.venv/bin/pytest -q
```

`GET /` is public (`{"status":"ok"}`). Product routes require a **Clerk** Bearer token; tests mock auth via `main.app.dependency_overrides`.

### External keys (optional by feature)

`OPENAI_API_KEY`, `GEMINI_API_KEY`, Modal or `ASSEMBLYAI_API_KEY` for transcription, Clerk JWKS (defaults to hosted `clerk.irisvideo.app`).

### Iris-Main (macOS only)

The iOS app cannot build on Linux (no Xcode/CocoaPods). On macOS: `pod install` in `Iris-Main`, open `Iris-Main.xcworkspace`, Run; backend URL defaults to `http://127.0.0.1:8000`.
