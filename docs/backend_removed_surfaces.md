# Backend Removed Surfaces

This file records backend surfaces that are candidates for deletion during the cleanup.
The default decision is to remove redundant endpoints and import facades once canonical
behavior exists under `app.agent`.

## Intent Run HTTP + WebSocket

- Old surface: `POST /intent-runs`
- Old surface: `WS /ws/intent-runs/{run_id}`
- Canonical replacement: `POST /agent/runs` with `kind: "intent"`, or a future canonical agent streaming route if
  streaming remains product-required.
- Reason for removal: the old flow is edit-only and exists beside the combined agent response.
  Keeping both makes `app.intent_compiler.runs` a permanent second intent boundary.
- Required proof before deletion: canonical tests cover edit result generation, UI plan generation,
  transcript hydration metadata, ownership checks, and any retained streaming behavior.

## UI Workspace Plan Endpoint

- Old surface: `POST /projects/{project_id}/ui-workspace-plan`
- Canonical replacement: `POST /agent/runs` with `kind: "intent"` returning the canonical `ui` plan from
  `app.agent.intent.ui`.
- Reason for removal: the old planner duplicates the canonical agent UI planner and keeps
  `app.ui_workspace` alive as a second UI model family.
- Required proof before deletion: canonical UI tests cover default workspace, visual/audio slice
  selection, effect-parameter controls, catalog validation, and route-level ownership checks.

## Intent Compiler Import Facade

- Old surface: `app.intent_compiler.*`
- Canonical replacement: `app.agent.intent.editing.*` and `app.agent.intent.*`.
- Reason for removal: most modules are wildcard re-exports and do not own behavior.
- Required proof before deletion: production import search returns no `app.intent_compiler`
  imports, and tests import canonical modules directly.

## UI Workspace Package

- Old surface: `app.ui_workspace.*`
- Canonical replacement: `app.agent.intent.ui.*`.
- Reason for removal: useful typed models, catalog, validation, and planner behavior should have
  one owner under the canonical agent UI package.
- Required proof before deletion: production import search returns no `app.ui_workspace` imports,
  and canonical UI tests cover the moved model/catalog/validator behavior.
