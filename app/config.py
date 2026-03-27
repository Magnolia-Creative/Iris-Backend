from dotenv import load_dotenv
import os


load_dotenv()


class Settings:
    def __init__(self) -> None:
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            raise ValueError("DATABASE_URL is not set")
        self.database_url = database_url
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.transcript_cache_ttl_seconds = int(
            os.getenv("TRANSCRIPT_CACHE_TTL_SECONDS", "3600")
        )


settings = Settings()
