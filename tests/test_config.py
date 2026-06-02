from app.config import Settings


def test_clerk_instance_development_selects_test_issuer(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///test.db")
    monkeypatch.setenv("CLERK_INSTANCE", "development")
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
    monkeypatch.delenv("CLERK_FRONTEND_API_URL", raising=False)
    monkeypatch.delenv("CLERK_ISSUER", raising=False)
    monkeypatch.delenv("CLERK_JWKS_URL", raising=False)

    settings = Settings()

    assert settings.clerk_frontend_api_url == "https://clerk.irisvideo.app"
    assert settings.clerk_issuer == "https://clerk.irisvideo.app"
    assert settings.clerk_jwks_url == "https://clerk.irisvideo.app/.well-known/jwks.json"


def test_clerk_frontend_api_url_overrides_instance(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///test.db")
    monkeypatch.setenv("CLERK_INSTANCE", "production")
    monkeypatch.setenv("CLERK_FRONTEND_API_URL", "https://example.clerk.accounts.dev/")
    monkeypatch.delenv("CLERK_ISSUER", raising=False)
    monkeypatch.delenv("CLERK_JWKS_URL", raising=False)

    settings = Settings()

    assert settings.clerk_frontend_api_url == "https://example.clerk.accounts.dev"
    assert settings.clerk_issuer == "https://example.clerk.accounts.dev"
    assert settings.clerk_jwks_url == "https://example.clerk.accounts.dev/.well-known/jwks.json"
