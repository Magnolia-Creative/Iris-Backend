from dotenv import load_dotenv
import os


load_dotenv()


class Settings:
    def __init__(self) -> None:
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            raise ValueError("DATABASE_URL is not set")
        self.database_url = database_url


settings = Settings()
