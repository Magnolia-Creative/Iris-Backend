from app.config import Settings


def test_clerk_instance_development_selects_test_issuer(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///test.db")
    monkeypatch.setenv("CLERK_INSTANCE", "development")
    monkeypatch.delenv("IRIS_ENV", raising=False)
    monkeypatch.delenv("CLERK_FRONTEND_API_URL", raising=False)
    monkeypatch.delenv("CLERK_ISSUER", raising=False)
    monkeypatch.delenv("CLERK_JWKS_URL", raising=False)

    settings = Settings()

    assert settings.clerk_frontend_api_url == "https://ethical-adder-19.clerk.accounts.dev"
    assert settings.clerk_issuer == "https://ethical-adder-19.clerk.accounts.dev"
    assert settings.clerk_jwks_url == "https://ethical-adder-19.clerk.accounts.dev/.well-known/jwks.json"


def test_clerk_instance_production_selects_live_issuer(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///test.db")
    monkeypatch.setenv("CLERK_INSTANCE", "production")
    monkeypatch.delenv("IRIS_ENV", raising=False)
    monkeypatch.delenv("CLERK_FRONTEND_API_URL", raising=False)
    monkeypatch.delenv("CLERK_ISSUER", raising=False)
    monkeypatch.delenv("CLERK_JWKS_URL", raising=False)

    settings = Settings()

    assert settings.clerk_frontend_api_url == "https://clerk.irisvideo.app"
    assert settings.clerk_issuer == "https://clerk.irisvideo.app"
    assert settings.clerk_jwks_url == "https://clerk.irisvideo.app/.well-known/jwks.json"


def test_iris_env_local_selects_test_issuer(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///test.db")
    monkeypatch.setenv("IRIS_ENV", "local")
    monkeypatch.delenv("CLERK_INSTANCE", raising=False)
    monkeypatch.delenv("CLERK_FRONTEND_API_URL", raising=False)
    monkeypatch.delenv("CLERK_ISSUER", raising=False)
    monkeypatch.delenv("CLERK_JWKS_URL", raising=False)

    settings = Settings()

    assert settings.iris_env == "local"
    assert settings.clerk_frontend_api_url == "https://ethical-adder-19.clerk.accounts.dev"
    assert settings.clerk_issuer == "https://ethical-adder-19.clerk.accounts.dev"
    assert settings.clerk_jwks_url == "https://ethical-adder-19.clerk.accounts.dev/.well-known/jwks.json"


def test_iris_env_prod_selects_live_issuer(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///test.db")
    monkeypatch.setenv("IRIS_ENV", "prod")
    monkeypatch.delenv("CLERK_INSTANCE", raising=False)
    monkeypatch.delenv("CLERK_FRONTEND_API_URL", raising=False)
    monkeypatch.delenv("CLERK_ISSUER", raising=False)
    monkeypatch.delenv("CLERK_JWKS_URL", raising=False)

    settings = Settings()

    assert settings.iris_env == "prod"
    assert settings.clerk_frontend_api_url == "https://clerk.irisvideo.app"
    assert settings.clerk_issuer == "https://clerk.irisvideo.app"
    assert settings.clerk_jwks_url == "https://clerk.irisvideo.app/.well-known/jwks.json"


def test_clerk_instance_overrides_iris_env(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///test.db")
    monkeypatch.setenv("IRIS_ENV", "prod")
    monkeypatch.setenv("CLERK_INSTANCE", "local")
    monkeypatch.delenv("CLERK_FRONTEND_API_URL", raising=False)
    monkeypatch.delenv("CLERK_ISSUER", raising=False)
    monkeypatch.delenv("CLERK_JWKS_URL", raising=False)

    settings = Settings()

    assert settings.clerk_frontend_api_url == "https://ethical-adder-19.clerk.accounts.dev"
    assert settings.clerk_issuer == "https://ethical-adder-19.clerk.accounts.dev"
    assert settings.clerk_jwks_url == "https://ethical-adder-19.clerk.accounts.dev/.well-known/jwks.json"


def test_clerk_frontend_api_url_overrides_instance(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///test.db")
    monkeypatch.setenv("IRIS_ENV", "local")
    monkeypatch.setenv("CLERK_INSTANCE", "production")
    monkeypatch.setenv("CLERK_FRONTEND_API_URL", "https://example.clerk.accounts.dev/")
    monkeypatch.delenv("CLERK_ISSUER", raising=False)
    monkeypatch.delenv("CLERK_JWKS_URL", raising=False)

    settings = Settings()

    assert settings.clerk_frontend_api_url == "https://example.clerk.accounts.dev"
    assert settings.clerk_issuer == "https://example.clerk.accounts.dev"
    assert settings.clerk_jwks_url == "https://example.clerk.accounts.dev/.well-known/jwks.json"


def test_local_auth_bypass_enabled_only_for_local_envs(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///test.db")
    monkeypatch.setenv("IRIS_ENV", "local")
    monkeypatch.setenv("LOCAL_AUTH_BYPASS", "true")
    monkeypatch.setenv("LOCAL_AUTH_BYPASS_USER_ID", "dev_user")

    settings = Settings()

    assert settings.local_auth_bypass is True
    assert settings.local_auth_bypass_enabled is True
    assert settings.local_auth_bypass_user_id == "dev_user"


def test_local_auth_bypass_ignored_in_production(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///test.db")
    monkeypatch.setenv("IRIS_ENV", "production")
    monkeypatch.setenv("LOCAL_AUTH_BYPASS", "true")

    settings = Settings()

    assert settings.local_auth_bypass is True
    assert settings.local_auth_bypass_enabled is False
