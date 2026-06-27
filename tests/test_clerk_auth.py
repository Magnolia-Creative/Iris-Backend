from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
import asyncio

from fastapi import HTTPException
from fastapi.testclient import TestClient

import main
from app.api.routes import projects as project_routes
from app.auth import ClerkPrincipal
from app.auth import require_clerk_user
from app.auth.ownership import require_owned_project, require_owned_session
from app.config import settings
from app.database import get_db
from app.database import models


@asynccontextmanager
async def _noop_lifespan(_app):
    yield


async def _fake_db() -> AsyncIterator[object]:
    yield object()


class _ScalarResult:
    def __init__(self, row):
        self._row = row

    def scalar_one_or_none(self):
        return self._row


class _OwnershipDb:
    def __init__(self, rows):
        self.rows = list(rows)
        self.statements = []

    async def execute(self, *args, **_kwargs):
        self.statements.extend(args)
        return _ScalarResult(self.rows.pop(0) if self.rows else None)


def _principal(user_id: str = "user_a") -> ClerkPrincipal:
    return ClerkPrincipal(user_id=user_id, session_id="sess_123", claims={"sub": user_id})


def test_protected_route_rejects_missing_token():
    original_lifespan = main.app.router.lifespan_context
    main.app.router.lifespan_context = _noop_lifespan
    main.app.dependency_overrides[get_db] = _fake_db
    main.app.dependency_overrides.pop(require_clerk_user, None)

    try:
        with TestClient(main.app) as client:
            response = client.post("/projects", json={"name": "Private"})
    finally:
        main.app.router.lifespan_context = original_lifespan
        main.app.dependency_overrides.clear()

    assert response.status_code == 401


def test_local_auth_bypass_allows_missing_token(monkeypatch):
    async def fake_create_project(db, *, owner_user_id, name=None):
        assert owner_user_id == "dev_user"
        assert name == "Local"
        return {"project_id": 43, "project_name": "Local"}

    monkeypatch.setattr(settings, "local_auth_bypass_enabled", True)
    monkeypatch.setattr(settings, "local_auth_bypass_user_id", "dev_user")
    monkeypatch.setattr(settings, "iris_env", "local")
    monkeypatch.setattr(project_routes, "create_project", fake_create_project)
    original_lifespan = main.app.router.lifespan_context
    main.app.router.lifespan_context = _noop_lifespan
    main.app.dependency_overrides[get_db] = _fake_db
    main.app.dependency_overrides.pop(require_clerk_user, None)

    try:
        with TestClient(main.app) as client:
            response = client.post("/projects", json={"name": "Local"})
    finally:
        main.app.router.lifespan_context = original_lifespan
        main.app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["project_id"] == 43


def test_health_route_stays_public():
    original_lifespan = main.app.router.lifespan_context
    main.app.router.lifespan_context = _noop_lifespan

    try:
        with TestClient(main.app) as client:
            response = client.get("/")
    finally:
        main.app.router.lifespan_context = original_lifespan

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_require_owned_project_accepts_matching_owner():
    project = models.Project(id=11, name="Launch Day", clerk_user_id="user_a")
    result = asyncio.run(
        require_owned_project(
            _OwnershipDb([project]),
            project_id=11,
            principal=_principal("user_a"),
        )
    )

    assert result is project


def test_require_owned_project_hides_other_owner():
    try:
        asyncio.run(
            require_owned_project(
                _OwnershipDb([None]),
                project_id=11,
                principal=_principal("user_b"),
            )
        )
    except HTTPException as exc:
        assert exc.status_code == 404
    else:
        raise AssertionError("Expected ownership check to raise HTTPException.")


def test_require_owned_project_skips_owner_filter_for_local_auth_bypass(monkeypatch):
    monkeypatch.setattr(settings, "local_auth_bypass_enabled", True)
    project = models.Project(id=11, name="Launch Day", clerk_user_id="someone_else")
    db = _OwnershipDb([project])

    result = asyncio.run(
        require_owned_project(
            db,
            project_id=11,
            principal=_principal("dev_user"),
        )
    )

    assert result is project
    assert "clerk_user_id" not in str(db.statements[0].whereclause)


def test_require_owned_session_hides_other_owner():
    try:
        asyncio.run(
            require_owned_session(
                _OwnershipDb([None]),
                session_id=7,
                principal=_principal("user_b"),
            )
        )
    except HTTPException as exc:
        assert exc.status_code == 404
    else:
        raise AssertionError("Expected ownership check to raise HTTPException.")


def test_require_owned_session_skips_owner_filter_for_local_auth_bypass(monkeypatch):
    monkeypatch.setattr(settings, "local_auth_bypass_enabled", True)
    session = models.Session(id=7, name="Local Session", clerk_user_id="someone_else")
    db = _OwnershipDb([session])

    result = asyncio.run(
        require_owned_session(
            db,
            session_id=7,
            principal=_principal("dev_user"),
        )
    )

    assert result is session
    assert "clerk_user_id" not in str(db.statements[0].whereclause)


def test_create_project_receives_clerk_owner(monkeypatch, app_client, set_db_override):
    async def fake_create_project(db, *, owner_user_id, name=None):
        assert owner_user_id == "user_test"
        assert name == "Owned"
        return {"project_id": 42, "project_name": "Owned"}

    monkeypatch.setattr(project_routes, "create_project", fake_create_project)
    set_db_override(_fake_db)
    response = app_client.post("/projects", json={"name": "Owned"})

    assert response.status_code == 200
    assert response.json()["project_id"] == 42
