import pytest

import main
from app.auth import ClerkPrincipal


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

    main.app.dependency_overrides[main.require_clerk_user] = fake_require_clerk_user
    monkeypatch.setattr(main, "require_owned_project", fake_require_owned_project)
    monkeypatch.setattr(main, "require_owned_session", fake_require_owned_session)
    monkeypatch.setattr(main, "require_owned_project_clip", fake_require_owned_project_clip)
    yield
    main.app.dependency_overrides.pop(main.require_clerk_user, None)
