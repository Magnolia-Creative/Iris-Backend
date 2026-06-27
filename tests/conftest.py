import pytest

import main
from app.api.routes import captions as captions_routes
from app.api.routes import clips as clip_routes
from app.api.routes import intent as intent_routes
from app.api.routes import projects as project_routes
from app.api.routes import search as search_routes
from app.api.routes import sessions as session_routes
from app.auth import ClerkPrincipal, require_clerk_user


@pytest.fixture(autouse=True)
def default_clerk_route_auth(monkeypatch):
    async def fake_require_clerk_user():
        return ClerkPrincipal(user_id="user_test", session_id="sess_test", claims={"sub": "user_test"})

    async def fake_require_owned_project(*args, **kwargs):
        return None

    async def fake_require_owned_session(*args, **kwargs):
        return None

    async def fake_require_owned_project_clip(*args, **kwargs):
        return None

    main.app.dependency_overrides[require_clerk_user] = fake_require_clerk_user
    monkeypatch.setattr(main, "require_owned_project", fake_require_owned_project, raising=False)
    monkeypatch.setattr(main, "require_owned_session", fake_require_owned_session, raising=False)
    monkeypatch.setattr(
        main,
        "require_owned_project_clip",
        fake_require_owned_project_clip,
        raising=False,
    )
    for route_module in (captions_routes, clip_routes, intent_routes, project_routes, search_routes):
        monkeypatch.setattr(route_module, "require_owned_project", fake_require_owned_project)
    for route_module in (clip_routes, intent_routes, session_routes):
        monkeypatch.setattr(route_module, "require_owned_session", fake_require_owned_session)
    monkeypatch.setattr(
        clip_routes,
        "require_owned_project_clip",
        fake_require_owned_project_clip,
    )
    yield
    main.app.dependency_overrides.pop(require_clerk_user, None)
