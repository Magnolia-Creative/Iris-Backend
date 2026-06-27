import main


def _http_routes() -> set[tuple[str, str]]:
    routes: set[tuple[str, str]] = set()
    for route in main.app.routes:
        methods = getattr(route, "methods", None)
        if not methods:
            continue
        for method in methods:
            if method in {"HEAD", "OPTIONS"}:
                continue
            routes.add((method, route.path))
    return routes


def _websocket_routes() -> set[str]:
    return {
        route.path
        for route in main.app.routes
        if route.__class__.__name__ == "APIWebSocketRoute"
    }


def test_public_http_route_inventory_uses_projects_sources_and_agent_domains():
    routes = _http_routes()

    assert {
        ("GET", "/"),
        ("GET", "/db-health"),
        ("POST", "/projects"),
        ("GET", "/projects/{project_id}/sources"),
        ("POST", "/projects/{project_id}/sources"),
        ("DELETE", "/projects/{project_id}/sources/{local_key}"),
        ("GET", "/projects/{project_id}/sources/{local_key}/transcript"),
        ("POST", "/projects/{project_id}/sources/search"),
        ("POST", "/agent/transcriptions/sentences"),
        ("POST", "/agent/runs"),
        ("GET", "/agent/runs/{run_id}"),
        ("GET", "/agent/runs/{run_id}/debug"),
    }.issubset(routes)


def test_public_websocket_route_inventory_uses_agent_domain():
    assert {
        "/agent/runs/{run_id}/stream",
        "/agent/voice/transcribe",
        "/agent/voice/intent",
    }.issubset(_websocket_routes())


def test_removed_endpoint_paths_are_not_mounted():
    http_paths = {path for _, path in _http_routes()}
    websocket_paths = _websocket_routes()

    assert {
        "/captions",
        "/sessions/upload",
        "/sessions/{session_id}",
        "/sessions/{session_id}/debug",
        "/projects/{project_id}/clips/process",
        "/projects/{project_id}/clips/status",
        "/projects/{project_id}/clips/{local_key}",
        "/projects/{project_id}/semantic-search",
        "/projects/{project_id}/transcript-search",
        "/projects/agent-sessions",
        "/projects/{project_id}/agent-sessions",
        "/transcriptions/sentences",
        "/agent/intent",
    }.isdisjoint(http_paths)

    assert {
        "/ws/transcribe",
        "/ws/intent/voice",
        "/ws/sessions/{session_id}",
    }.isdisjoint(websocket_paths)
