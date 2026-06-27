# Endpoint Refactor Migration

This refactor is a breaking backend API change. It removes the old mixed endpoint layout and groups the public API around project lifecycle, project sources, and agent runs.

## New taxonomy

- `/projects` owns project lifecycle only.
- `/projects/{project_id}/sources` owns media/context-base ingestion, source status, transcript retrieval, and source search.
- `/agent` owns agent invocation, automake run status/debug, voice/streaming agent protocols, and standalone sentence transcription.
- Health endpoints remain outside the product taxonomy: `GET /` and `GET /db-health`.

## Old to new endpoint mapping

| Old endpoint | New endpoint | Notes |
| --- | --- | --- |
| `POST /projects/{project_id}/clips/process` | `POST /projects/{project_id}/sources` | Multipart fields are unchanged: `videos`, repeated `local_key`, optional `session_id`, optional visual frame fields. |
| `GET /projects/{project_id}/clips/status` | `GET /projects/{project_id}/sources` | Response shape is unchanged. |
| `DELETE /projects/{project_id}/clips/{local_key}` | `DELETE /projects/{project_id}/sources/{local_key}` | Optional `session_id` query remains supported. |
| `GET /captions?project_id=...&local_key=...` | `GET /projects/{project_id}/sources/{local_key}/transcript` | Response shape is unchanged. |
| `POST /projects/{project_id}/semantic-search` | `POST /projects/{project_id}/sources/search` | Add body field `mode: "semantic"`. |
| `POST /projects/{project_id}/transcript-search` | `POST /projects/{project_id}/sources/search` | Add body field `mode: "transcript"`. |
| `POST /agent/intent` | `POST /agent/runs` | Add body field `kind: "intent"`; existing intent fields remain. |
| `POST /projects/agent-sessions` | `POST /agent/runs` | Use `kind: "automake"` and omit `project_id` to create a project/session pair. |
| `POST /projects/{project_id}/agent-sessions` | `POST /agent/runs` | Use `kind: "automake"` with `project_id`. |
| `GET /sessions/{session_id}` | `GET /agent/runs/{run_id}` | `run_id` is currently the existing session id. |
| `GET /sessions/{session_id}/debug` | `GET /agent/runs/{run_id}/debug` | `run_id` is currently the existing session id. |
| `WS /ws/sessions/{session_id}` | `WS /agent/runs/{run_id}/stream` | WebSocket message protocol is unchanged. |
| `WS /ws/transcribe` | `WS /agent/voice/transcribe` | Query `model` behavior is unchanged. |
| `WS /ws/intent/voice` | `WS /agent/voice/intent` | Start payload and streamed response events are unchanged. |
| `POST /transcriptions/sentences` | `POST /agent/transcriptions/sentences` | Multipart `audio` request and response shape are unchanged. |
| `POST /sessions/upload` | No direct one-call replacement | Create an automake run with `POST /agent/runs`, then upload sources to its project with `POST /projects/{project_id}/sources` and optional `session_id`. |

## Request shape changes

Source search is now mode-based:

```json
{
  "query": "car driving away",
  "mode": "semantic",
  "limit": 5
}
```

Intent/UI modification runs are now agent runs:

```json
{
  "kind": "intent",
  "prompt": "make the audio louder",
  "context": {}
}
```

Automake runs are also created through `POST /agent/runs`:

```json
{
  "kind": "automake",
  "project_id": 123,
  "session_name": "Launch edit"
}
```

To create a new project and automake run together, omit `project_id` and optionally pass `project_name`.

## Frontend migration checklist

- Update API constants for all old paths listed above.
- Update clip upload/status/cancel call sites from `clips` terminology to `sources` paths.
- Update transcript/caption fetches to use `/projects/{project_id}/sources/{local_key}/transcript`.
- Replace separate semantic/transcript search URLs with `/projects/{project_id}/sources/search` and pass `mode`.
- Replace direct intent calls with `POST /agent/runs` plus `kind: "intent"`.
- Replace automake session creation calls with `POST /agent/runs` plus `kind: "automake"`.
- Rename client variables carefully: `run_id` maps to the existing backend `session_id` for now.
- Update automake websocket URLs to `/agent/runs/{run_id}/stream`.
- Update voice websocket URLs to `/agent/voice/intent` and `/agent/voice/transcribe`.
- Remove assumptions that `/sessions/*` endpoints still exist.

No frontend migration is implemented in this backend refactor.
