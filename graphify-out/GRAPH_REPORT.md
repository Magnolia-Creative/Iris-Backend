# Graph Report - .  (2026-06-27)

## Corpus Check
- Corpus is ~43,550 words - fits in a single context window. You may not need a graph.

## Summary
- 956 nodes · 2804 edges · 56 communities (46 shown, 10 thin omitted)
- Extraction: 92% EXTRACTED · 8% INFERRED · 0% AMBIGUOUS · INFERRED: 231 edges (avg confidence: 0.57)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Intent Agent Graph Build|Intent Agent Graph Build]]
- [[_COMMUNITY_Chunk Time Search Range|Chunk Time Search Range]]
- [[_COMMUNITY_Websocket Clerk Request Voice|Websocket Clerk Request Voice]]
- [[_COMMUNITY_Transcript Hydrate Context Transcripts|Transcript Hydrate Context Transcripts]]
- [[_COMMUNITY_Intent Agent Runs Workspace|Intent Agent Runs Workspace]]
- [[_COMMUNITY_Plan Payload Timeline Agent|Plan Payload Timeline Agent]]
- [[_COMMUNITY_Session Clip Create Agent|Session Clip Create Agent]]
- [[_COMMUNITY_Effect Clip Plan Prompt|Effect Clip Plan Prompt]]
- [[_COMMUNITY_Session State Run Context|Session State Run Context]]
- [[_COMMUNITY_Effect Capability Intent Capabilities|Effect Capability Intent Capabilities]]
- [[_COMMUNITY_Resolve Apply Clip Remove|Resolve Apply Clip Remove]]
- [[_COMMUNITY_Clip Effect Plan Context|Clip Effect Plan Context]]
- [[_COMMUNITY_Route Project Get Fake|Route Project Get Fake]]
- [[_COMMUNITY_Assemblyai From Segments Transcript|Assemblyai From Segments Transcript]]
- [[_COMMUNITY_Action Compile Clip Tier|Action Compile Clip Tier]]
- [[_COMMUNITY_Timeline State Build Clip|Timeline State Build Clip]]
- [[_COMMUNITY_Create Health Project Fake|Create Health Project Fake]]
- [[_COMMUNITY_Pos Projects Semantic Search|Pos Projects Semantic Search]]
- [[_COMMUNITY_Clip Cleanup Ranges Get|Clip Cleanup Ranges Get]]
- [[_COMMUNITY_Get Decision Trace Elapsed|Get Decision Trace Elapsed]]
- [[_COMMUNITY_Chunk Windows Time Audio|Chunk Windows Time Audio]]
- [[_COMMUNITY_Project Get Source Require|Project Get Source Require]]
- [[_COMMUNITY_Owner Require Owned Auth|Owner Require Owned Auth]]
- [[_COMMUNITY_Names Add Column Index|Names Add Column Index]]
- [[_COMMUNITY_Run Migrations Offline Online|Run Migrations Offline Online]]
- [[_COMMUNITY_Duration Expression Time Resolve|Duration Expression Time Resolve]]
- [[_COMMUNITY_Phrase Transcript Matches Word|Phrase Transcript Matches Word]]
- [[_COMMUNITY_Visual Vector Index Frame|Visual Vector Index Frame]]
- [[_COMMUNITY_Task Clip Registry Registered|Task Clip Registry Registered]]
- [[_COMMUNITY_Embed Bytes Gemini Sync|Embed Bytes Gemini Sync]]
- [[_COMMUNITY_Session Debug Entry Prompt|Session Debug Entry Prompt]]
- [[_COMMUNITY_Sentence Intent Jsonb Transcript|Sentence Intent Jsonb Transcript]]
- [[_COMMUNITY_Parser Intent Agent Probe|Parser Intent Agent Probe]]
- [[_COMMUNITY_Clip Embedding Halfvec Project|Clip Embedding Halfvec Project]]
- [[_COMMUNITY_Get For Clips Persisted|Get For Clips Persisted]]
- [[_COMMUNITY_Route Inventory Http Routes|Route Inventory Http Routes]]
- [[_COMMUNITY_Graph Elapsed Get Configurable|Graph Elapsed Get Configurable]]
- [[_COMMUNITY_Run Agent Runs Maps|Run Agent Runs Maps]]
- [[_COMMUNITY_Health Endpoints|Health Endpoints]]
- [[_COMMUNITY_Agent Entry Points And|Agent Entry Points And]]
- [[_COMMUNITY_Fast Transport Layer For|Fast Transport Layer For]]
- [[_COMMUNITY_Endpoint Refactor Migration Breaking|Endpoint Refactor Migration Breaking]]
- [[_COMMUNITY_Captions Projects Project Sources|Captions Projects Project Sources]]
- [[_COMMUNITY_Domain Facing And Presenters|Domain Facing And Presenters]]
- [[_COMMUNITY_Fast Route Modules|Fast Route Modules]]
- [[_COMMUNITY_Request And Websocket Payload|Request And Websocket Payload]]
- [[_COMMUNITY_Layer Modules For Non|Layer Modules For Non]]
- [[_COMMUNITY_Semantic Constants Chunking Aligned|Semantic Constants Chunking Aligned]]
- [[_COMMUNITY_Fastapi|Fastapi]]

## God Nodes (most connected - your core abstractions)
1. `IntentCompilerContext` - 87 edges
2. `IntentCompiler` - 66 edges
3. `IntentCompileResult` - 47 edges
4. `SemanticEditPlan` - 42 edges
5. `SemanticEditOperation` - 40 edges
6. `SessionGraphState` - 39 edges
7. `IntentCompilerService` - 38 edges
8. `ClerkPrincipal` - 36 edges
9. `IntentTimelineSimulator` - 35 edges
10. `ExperimentalEffectOperation` - 29 edges

## Surprising Connections (you probably didn't know these)
- `_FakeEmbeddingClient` --uses--> `IntentCompiler`  [INFERRED]
  tests/test_intent_compiler.py → app/agent/intent/editing/compiler.py
- `_FakeEmbeddingClient` --uses--> `TranscriptWord`  [INFERRED]
  tests/test_intent_compiler.py → app/agent/intent/editing/models.py
- `_FakeIntentCompilerService` --uses--> `IntentCompilerContext`  [INFERRED]
  tests/test_intent_agent.py → app/agent/intent/editing/models.py
- `_FakeIntentUIPlannerService` --uses--> `IntentCompilerContext`  [INFERRED]
  tests/test_intent_agent.py → app/agent/intent/editing/models.py
- `_FakeEmbeddingClient` --uses--> `IntentCompilerContext`  [INFERRED]
  tests/test_intent_compiler.py → app/agent/intent/editing/models.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Public API Taxonomy Group** — docs_endpoint_refactor_migration_projects_endpoint, docs_endpoint_refactor_migration_projects_sources_endpoint, docs_endpoint_refactor_migration_agent_endpoint, docs_endpoint_refactor_migration_health_endpoints [EXTRACTED 1.00]
- **Agent Runs Consolidation** — docs_endpoint_refactor_migration_post_agent_runs, docs_endpoint_refactor_migration_kind_intent, docs_endpoint_refactor_migration_kind_automake, docs_endpoint_refactor_migration_run_id_session_id_mapping [EXTRACTED 1.00]
- **Semantic Index Storage Pipeline** — docs_semantic_indexing_clip_ingest, docs_semantic_indexing_vector_index_job, docs_semantic_indexing_gemini_embedding_2, docs_semantic_indexing_per_chunk_embeddings, docs_semantic_indexing_pgvector, docs_semantic_indexing_clip_chunk_embeddings, docs_semantic_indexing_hnsw_index [EXTRACTED 1.00]

## Communities (56 total, 10 thin omitted)

### Community 0 - "Intent Agent Graph Build"
Cohesion: 0.07
Nodes (54): build_editing_intent_graph(), _handle_openai_event(), Event, IntentAgentBaseModel, IntentAgentMeta, IntentAgentRequest, IntentAgentResponse, IntentAgentTiming (+46 more)

### Community 1 - "Chunk Time Search Range"
Cohesion: 0.07
Nodes (55): search_response_dict(), SearchDomainModel, SearchMatch, SearchResponse, _build_merged_group(), ChunkHit, _dedupe_hits_by_chunk_index(), _interp_falling_threshold_time() (+47 more)

### Community 2 - "Websocket Clerk Request Voice"
Cohesion: 0.07
Nodes (42): request_validation_exception_handler(), _clerk_frontend_api_url(), _env_bool(), _env_list(), Settings, _extract_bearer_token(), _get_jwks_client(), _local_auth_bypass_principal() (+34 more)

### Community 3 - "Transcript Hydrate Context Transcripts"
Cohesion: 0.09
Nodes (40): TranscriptPauseRange, current_clip_at_playhead_id(), needs_phrase_matching(), needs_transcript_hydration(), Deterministic rules for when intent compilation should hydrate transcript data., Clip IDs that should receive hydrated transcript context for this prompt., resolve_transcript_target_clip_ids(), _attach_phrase_matches() (+32 more)

### Community 4 - "Intent Agent Runs Workspace"
Cohesion: 0.06
Nodes (40): app.agent.intent.*, app.agent.intent.editing.*, app.agent.intent.ui, app.intent_compiler.*, app.intent_compiler.runs, app.ui_workspace, Backend Removed Surfaces, Canonical Agent Behavior (+32 more)

### Community 5 - "Plan Payload Timeline Agent"
Cohesion: 0.09
Nodes (31): CleanupPlan, ClipState, DecisionAgentOutput, EditPlan, EditPlanModel, RetrievalPlan, RetrievalPlanModel, TimelineEntry (+23 more)

### Community 6 - "Session Clip Create Agent"
Cohesion: 0.17
Nodes (33): AgentRunCreatePayload, AsyncSession, Clip, Session, HTTPException, create_agent_run(), _create_automake_agent_run(), create_sentence_transcription() (+25 more)

### Community 7 - "Effect Clip Plan Prompt"
Cohesion: 0.13
Nodes (27): IntentCompilerContext, _apply_contextual_assumptions(), _assume_effect_operation_targets(), _assume_operation_target(), _compact_transcript_context(), _current_clip_at_playhead_id(), _default_effect_direction(), _editor_context() (+19 more)

### Community 8 - "Session State Run Context"
Cohesion: 0.13
Nodes (26): context_id(), intent_context_log_summary(), intent_result_log_summary(), is_timeline_approval_message(), elapsed_ms(), build_session_graph(), approve_session_timeline(), _default_llm() (+18 more)

### Community 9 - "Effect Capability Intent Capabilities"
Cohesion: 0.16
Nodes (25): ClipTranscriptContext, EditingIntentBaseModel, EffectCapability, EffectCapabilityParameter, ExperimentalEffectOperation, ExperimentalEffectPlan, IntentCompileRequest, IntentRunCreateResponse (+17 more)

### Community 10 - "Resolve Apply Clip Remove"
Cohesion: 0.18
Nodes (8): _dead_space_terms(), IntentTimelineSimulator, _invalid_remove_ranges(), _normalized_remove_ranges(), Resolution, _survivor_ranges(), TimeRange, SemanticEditTarget

### Community 11 - "Clip Effect Plan Context"
Cohesion: 0.21
Nodes (29): IntentCompiler, SemanticEditOperation, SemanticEditPlan, _multi_clip_context(), _sample_context(), test_color_filter_effect_groups_multiple_operations_per_clip(), test_color_filter_effect_plan_emits_update_clip_color_filter_action(), test_color_filter_effect_with_unresolved_target_clarifies() (+21 more)

### Community 12 - "Route Project Get Fake"
Cohesion: 0.10
Nodes (24): set_db_override(), test_get_captions_404_project(), test_get_captions_409_no_transcript(), test_get_captions_ok(), _fake_db_semantic_200(), _fake_db_semantic_404(), _FakeExecResult, _FakeSessionSemantic200 (+16 more)

### Community 13 - "Assemblyai From Segments Transcript"
Cohesion: 0.17
Nodes (26): Any, _numeric_effect_parameter(), _get_llm(), _assemblyai_fetch_sentences(), _assemblyai_full_text_from_segments(), _assemblyai_headers(), _assemblyai_ms_to_seconds(), _assemblyai_poll_transcript() (+18 more)

### Community 14 - "Action Compile Clip Tier"
Cohesion: 0.22
Nodes (18): action_execution_tier(), DurationExpression, Translates validated effect operations into concrete `updateClipColorFilter`, Lower tier runs first on the client (color before clip topology)., ResolvedOperation, _swift_reference_date_seconds(), TimeExpression, Action (+10 more)

### Community 15 - "Timeline State Build Clip"
Cohesion: 0.17
Nodes (21): _route_from_decision(), _route_from_validator(), SessionGraphState, TimelineValidationResult, ValidatedTimelineEntry, _timeline_context(), _clip_duration_lookup(), _clip_local_key_lookup() (+13 more)

### Community 16 - "Create Health Project Fake"
Cohesion: 0.12
Nodes (10): create_app(), lifespan(), get_db(), FastAPI, db_health(), create_project_endpoint(), ProjectCreatePayload, create_project() (+2 more)

### Community 17 - "Pos Projects Semantic Search"
Cohesion: 0.11
Nodes (21): mode semantic, mode transcript, POST /projects/{project_id}/semantic-search, POST /projects/{project_id}/sources/search, POST /projects/{project_id}/transcript-search, clip_chunk_embeddings, Clip Ingest, DATABASE_URL (+13 more)

### Community 18 - "Clip Cleanup Ranges Get"
Cohesion: 0.22
Nodes (19): ClipCleanupOutput, _build_clip_ranges_with_change_flag(), clip_cleanup_node(), _clip_context(), _clip_duration_lookup(), _elapsed_ms(), _emit_event(), _extract_prompt_entity_hits() (+11 more)

### Community 19 - "Get Decision Trace Elapsed"
Cohesion: 0.22
Nodes (17): ChatOpenAI, _clip_digest(), decision_agent_node(), _elapsed_ms(), _emit_event(), _get_configurable(), _get_llm(), _preview_focus_terms() (+9 more)

### Community 20 - "Chunk Windows Time Audio"
Cohesion: 0.15
Nodes (15): Path, _ffprobe_duration_seconds(), Slice audio into overlapping WAV chunks using ffmpeg., Return list of (chunk_index, start, end, center, wav_bytes)., slice_audio_to_wav_chunks(), chunk_time_windows(), 4s windows with 0.5s overlap (3.5s stride), matching SemanticSearchPipeline.chun, Return (start, end, center) for each chunk window. (+7 more)

### Community 21 - "Project Get Source Require"
Cohesion: 0.28
Nodes (16): ClerkPrincipal, _not_found(), require_owned_project(), require_owned_project_clip(), require_owned_session(), get_agent_run_status(), _run_intent_agent(), cancel_project_source() (+8 more)

### Community 22 - "Owner Require Owned Auth"
Cohesion: 0.17
Nodes (9): _OwnershipDb, _principal(), _ScalarResult, test_create_project_receives_clerk_owner(), test_require_owned_project_accepts_matching_owner(), test_require_owned_project_hides_other_owner(), test_require_owned_project_skips_owner_filter_for_local_auth_bypass(), test_require_owned_session_hides_other_owner() (+1 more)

### Community 23 - "Names Add Column Index"
Cohesion: 0.23
Nodes (15): do_run_migrations(), Connection, _column_names(), downgrade(), _index_names(), upgrade(), _column_names(), downgrade() (+7 more)

### Community 24 - "Run Migrations Offline Online"
Cohesion: 0.21
Nodes (12): Run migrations in 'offline' mode.      This configures the context with just a U, Run migrations in 'online' mode., run_migrations_offline(), run_migrations_online(), Base, Clip, Project, Standalone sentence-level transcription rows (no clip), e.g. POST /agent/transcr (+4 more)

### Community 25 - "Duration Expression Time Resolve"
Cohesion: 0.32
Nodes (4): DurationUnit, _float_value(), Durations phrased as 'the first second' (one second), 'first two seconds', etc., _spoken_number_value()

### Community 26 - "Phrase Transcript Matches Word"
Cohesion: 0.33
Nodes (11): TranscriptPhraseMatch, TranscriptWord, collect_phrase_matches(), _dedupe_phrase_matches(), extract_phrase_queries(), find_phrase_time_ranges(), _norm_token(), Phrase extraction and deterministic transcript word-window matching. (+3 more)

### Community 27 - "Visual Vector Index Frame"
Cohesion: 0.29
Nodes (9): ffmpeg_available(), schedule_vector_index_tasks(), Background multimodal embedding indexing for a clip., run_clip_vector_index(), build_visual_frames_by_local_key(), _coerce_float(), parse_visual_frame_manifest_json(), Parse visual frame manifest and match uploaded image parts. (+1 more)

### Community 28 - "Task Clip Registry Registered"
Cohesion: 0.25
Nodes (4): ClipTaskRegistry, RegisteredClipTask, Task, test_clip_task_registry_registers_cancels_and_clears_tasks()

### Community 29 - "Embed Bytes Gemini Sync"
Cohesion: 0.25
Nodes (6): _client(), embed_audio_bytes_sync(), embed_image_bytes_sync(), embed_text_sync(), _format_retrieval_query(), Gemini Embedding 2 client for multimodal clip indexing.

### Community 30 - "Session Debug Entry Prompt"
Cohesion: 0.35
Nodes (10): _analysis_excerpt(), build_session_debug_snapshot(), _clip_state_lookup(), _ensure_entry(), _extract_prompt_hits(), get_session_debug_data(), _prompt_entity_terms(), _public_entry() (+2 more)

### Community 31 - "Sentence Intent Jsonb Transcript"
Cohesion: 0.39
Nodes (6): _as_seconds(), intent_jsonb_from_sentence_api_result(), Normalize /agent/transcriptions/sentences API output into JSONB shape used by in, _segments_for_intent_hydration(), test_intent_jsonb_sentence_level_coarse_word(), test_intent_jsonb_uses_nested_words_when_present()

### Community 32 - "Parser Intent Agent Probe"
Cohesion: 0.48
Nodes (6): ArgumentParser, _build_parser(), _load_json(), main(), _print_summary(), _widget_ids_from_layout()

### Community 33 - "Clip Embedding Halfvec Project"
Cohesion: 0.47
Nodes (5): count_embeddings_for_project(), _halfvec_literal(), insert_embedding_row(), Persist and query clip chunk embeddings in Postgres (pgvector halfvec)., search_project_nearest()

### Community 34 - "Get For Clips Persisted"
Cohesion: 0.60
Nodes (5): _aggregate_clip_status(), get_persisted_project_data(), get_persisted_session_data(), get_transcripts_for_clips(), _video_payloads_for_clips()

### Community 35 - "Route Inventory Http Routes"
Cohesion: 0.60
Nodes (5): _http_routes(), test_public_http_route_inventory_uses_projects_sources_and_agent_domains(), test_public_websocket_route_inventory_uses_agent_domain(), test_removed_endpoint_paths_are_not_mounted(), _websocket_routes()

### Community 36 - "Graph Elapsed Get Configurable"
Cohesion: 0.70
Nodes (4): compile_edit_node(), _elapsed_ms(), _get_configurable(), EditingIntentGraphState

### Community 37 - "Run Agent Runs Maps"
Cohesion: 0.67
Nodes (3): WS /agent/runs/{run_id}/stream, GET /agent/runs/{run_id}, run_id maps to session_id

### Community 38 - "Health Endpoints"
Cohesion: 0.67
Nodes (3): GET /db-health, GET /, Health Endpoints

## Knowledge Gaps
- **32 isolated node(s):** `my-fastapi-app`, `Backend Removed Surfaces`, `Canonical Agent Streaming Route`, `app.intent_compiler.runs`, `app.agent.intent.editing.*` (+27 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **10 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `IntentCompilerContext` connect `Effect Clip Plan Prompt` to `Intent Agent Graph Build`, `Websocket Clerk Request Voice`, `Transcript Hydrate Context Transcripts`, `Plan Payload Timeline Agent`, `Session State Run Context`, `Effect Capability Intent Capabilities`, `Resolve Apply Clip Remove`, `Clip Effect Plan Context`, `Action Compile Clip Tier`, `Duration Expression Time Resolve`?**
  _High betweenness centrality (0.084) - this node is a cross-community bridge._
- **Why does `search_project_semantic()` connect `Chunk Time Search Range` to `Assemblyai From Segments Transcript`, `Project Get Source Require`, `Session Clip Create Agent`?**
  _High betweenness centrality (0.076) - this node is a cross-community bridge._
- **Why does `slice_audio_to_wav_chunks()` connect `Chunk Windows Time Audio` to `Visual Vector Index Frame`?**
  _High betweenness centrality (0.033) - this node is a cross-community bridge._
- **Are the 25 inferred relationships involving `IntentCompilerContext` (e.g. with `DurationExpression` and `IntentCompiler`) actually correct?**
  _`IntentCompilerContext` has 25 INFERRED edges - model-reasoned connections that need verification._
- **Are the 18 inferred relationships involving `IntentCompiler` (e.g. with `Action` and `ActionType`) actually correct?**
  _`IntentCompiler` has 18 INFERRED edges - model-reasoned connections that need verification._
- **Are the 27 inferred relationships involving `IntentCompileResult` (e.g. with `DurationExpression` and `IntentCompiler`) actually correct?**
  _`IntentCompileResult` has 27 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Run migrations in 'offline' mode.      This configures the context with just a U`, `Run migrations in 'online' mode.`, `Agent entry points and graph-backed workflows.` to the rest of the system?**
  _86 weakly-connected nodes found - possible documentation gaps or missing edges._