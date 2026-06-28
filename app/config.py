from dotenv import load_dotenv
import os


load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_list(name: str) -> list[str]:
    raw = os.getenv(name)
    if raw is None:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


CLERK_INSTANCE_URLS = {
    "local": "https://ethical-adder-19.clerk.accounts.dev",
    "development": "https://ethical-adder-19.clerk.accounts.dev",
    "dev": "https://ethical-adder-19.clerk.accounts.dev",
    "test": "https://ethical-adder-19.clerk.accounts.dev",
    "production": "https://clerk.irisvideo.app",
    "prod": "https://clerk.irisvideo.app",
    "live": "https://clerk.irisvideo.app",
}


def _clerk_frontend_api_url() -> str:
    explicit_url = os.getenv("CLERK_FRONTEND_API_URL")
    if explicit_url:
        return explicit_url.rstrip("/")

    instance = os.getenv("CLERK_INSTANCE") or os.getenv("IRIS_ENV", "production")
    instance = instance.strip().lower()
    return CLERK_INSTANCE_URLS.get(instance, CLERK_INSTANCE_URLS["production"])


class Settings:
    def __init__(self) -> None:
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            raise ValueError("DATABASE_URL is not set")
        self.database_url = database_url
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.intent_openai_model = os.getenv("INTENT_OPENAI_MODEL", "gpt-5.4-nano")
        self.intent_ui_openai_model = os.getenv("INTENT_UI_OPENAI_MODEL", "gpt-5.4-mini")
        self.iris_env = os.getenv("IRIS_ENV", "production").strip().lower()
        self.local_auth_bypass = _env_bool("LOCAL_AUTH_BYPASS", False)
        self.local_auth_bypass_user_id = os.getenv(
            "LOCAL_AUTH_BYPASS_USER_ID",
            "local_dev_user",
        )
        self.local_auth_bypass_enabled = self.local_auth_bypass and self.iris_env in {
            "local",
            "development",
            "dev",
            "test",
        }
        self.transcript_cache_ttl_seconds = int(
            os.getenv("TRANSCRIPT_CACHE_TTL_SECONDS", "3600")
        )
        self.transcription_provider = os.getenv("TRANSCRIPTION_PROVIDER", "modal").lower()
        self.assemblyai_api_key = os.getenv("ASSEMBLYAI_API_KEY")
        self.assemblyai_base_url = os.getenv("ASSEMBLYAI_BASE_URL", "https://api.assemblyai.com")
        self.assemblyai_poll_interval_seconds = float(
            os.getenv("ASSEMBLYAI_POLL_INTERVAL_SECONDS", "2.0")
        )
        self.assemblyai_poll_timeout_seconds = float(
            os.getenv("ASSEMBLYAI_POLL_TIMEOUT_SECONDS", "600.0")
        )
        self.assemblyai_ssl_verify = _env_bool("ASSEMBLYAI_SSL_VERIFY", True)
        self.assemblyai_ca_bundle = os.getenv("ASSEMBLYAI_CA_BUNDLE")
        # For outbound TLS (e.g. OpenAI WebSocket): path to a PEM of trusted roots.
        # If unset, code uses certifi’s bundle. Set to your corporate root CA if behind SSL inspection.
        # Standard env: SSL_CERT_FILE; alias: SSL_CA_BUNDLE.
        self.outbound_ssl_cafile = os.getenv("SSL_CERT_FILE") or os.getenv("SSL_CA_BUNDLE")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.semantic_indexing_enabled = _env_bool("SEMANTIC_INDEXING_ENABLED", True)
        self.clerk_frontend_api_url = _clerk_frontend_api_url()
        self.clerk_jwks_url = os.getenv(
            "CLERK_JWKS_URL",
            f"{self.clerk_frontend_api_url}/.well-known/jwks.json",
        )
        self.clerk_issuer = os.getenv("CLERK_ISSUER", self.clerk_frontend_api_url).rstrip("/")
        self.clerk_authorized_parties = _env_list("CLERK_AUTHORIZED_PARTIES")


settings = Settings()
