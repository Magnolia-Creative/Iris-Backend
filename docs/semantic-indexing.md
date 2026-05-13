# Semantic / multimodal clip indexing

Clip ingest can schedule a **vector index** job (Gemini `gemini-embedding-2` @ 3072 dims) that stores per-chunk embeddings in Postgres via **pgvector** (`halfvec(3072)`), independent of transcription.

## Environment

- `GEMINI_API_KEY` — required for embeddings and `POST /projects/{id}/semantic-search`.
- `SEMANTIC_INDEXING_ENABLED` — default `true`; set to `false` to skip scheduling index tasks.
- `DATABASE_URL` — must point to a Postgres instance where the `vector` extension can be enabled.

## Postgres / pgvector

The `vector` extension must be **installed on the server** (see [pgvector](https://github.com/pgvector/pgvector)). Alembic runs `CREATE EXTENSION IF NOT EXISTS vector` and creates `clip_chunk_embeddings` with an HNSW index on `halfvec(3072)`.

If the extension is unavailable on your host, migrations or inserts will fail until you use a pgvector-enabled image or package.

## Client upload

`POST /projects/{project_id}/clips/process` accepts optional:

- `visual_frame_manifest` — JSON string `{ "clips": [ { "local_key", "frames": [ { chunk_index, start_time_seconds, end_time_seconds, center_time_seconds, filename } ] } ] }`
- `visual_frames` — multipart file parts (field name `visual_frames`) whose filenames match the manifest.

Compressed audio continues to use the `videos` multipart field (unchanged contract).
