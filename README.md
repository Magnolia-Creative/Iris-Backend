# Iris Backend

The API and AI orchestration layer for **Iris**, a prompt-driven video-editing experience. It powers project ingest, transcription and captions, semantic source search, and structured editing-agent runs for the native iOS app.

> This repository is one half of Iris. The SwiftUI client lives in [Iris-Main](https://github.com/Magnolia-Creative/Iris-Main).

## What it does

- Accepts and processes project video sources
- Stores project, clip, transcript, caption, and agent-run data in PostgreSQL
- Produces structured edit intents from natural-language prompts
- Supports transcript and semantic search across a project's source media
- Streams agent progress and realtime voice transcription over WebSockets
- Protects user-owned resources with Clerk authentication

## Architecture

The FastAPI application is organized around a small API surface and domain-focused services:

```text
app/
├── api/        # FastAPI application factory and route modules
├── agent/      # Intent compilation and agent workflow logic
├── auth/       # Clerk authentication and ownership checks
├── database/   # SQLAlchemy models and database sessions
├── domains/    # Shared domain models
└── services/   # Ingest, transcription, captions, and semantic-search services
```

Database migrations are maintained in `alembic/`; the automated test suite is in `tests/`.

## Local development

**Requirements:** Python 3.11+, PostgreSQL, and Redis. `ffmpeg` is also needed for audio chunking during media processing.

```bash
git clone https://github.com/Magnolia-Creative/Iris-Backend.git
cd Iris-Backend

python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env
# Set DATABASE_URL and any provider/auth credentials required for your environment.

alembic upgrade head
uvicorn main:app --reload
```

The development server runs at `http://127.0.0.1:8000`. The companion iOS app is configured to use that address for debug builds.

## Configuration

Start with [`.env.example`](.env.example). At minimum, local development requires `DATABASE_URL`. Depending on the workflow you are exercising, configure:

- `OPENAI_API_KEY` for realtime transcription and voice intent flows
- `REDIS_URL` for transcript caching (defaults to local Redis)
- Clerk settings for authenticated environments
- Google GenAI credentials for Gemini-backed embedding workflows

Never commit real credentials or production connection strings.

## Testing

```bash
pytest
ruff check .
```

## Iris repositories

| Repository | Purpose |
| --- | --- |
| [Iris-Backend](https://github.com/Magnolia-Creative/Iris-Backend) | FastAPI API, agent workflows, data persistence, and media-processing services. |
| [Iris-Main](https://github.com/Magnolia-Creative/Iris-Main) | Native SwiftUI iOS client for the Iris editing workflow. |

## License

Copyright 2026 Magnolia Creative. This project is licensed under the [Apache License 2.0](LICENSE). See [NOTICE](NOTICE) for attribution information.
