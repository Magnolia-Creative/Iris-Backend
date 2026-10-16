"""Bring an Iris database to the schema required by the deployed service.

The first Render database was initialized by SQLAlchemy create_all, which does
not create the pgvector embedding table and does not record an Alembic version.
Stamp only that known baseline, then let Alembic run the remaining migrations.
"""

import asyncio
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import inspect

from app.database import Base, engine
from app.database import models  # noqa: F401 - register models on Base


BASELINE_REVISION = "b2e8f1a4c9d0"
REQUIRED_BASELINE_COLUMNS = {
    "projects": {"id", "name"},
    "sessions": {"id", "name", "graph_state"},
    "clips": {"id", "project_id", "session_id", "local_key", "processing_status"},
    "sentence_upload_transcripts": {"id", "transcript"},
}


def is_known_unversioned_baseline(connection) -> bool:
    inspector = inspect(connection)
    tables = set(inspector.get_table_names())
    if "alembic_version" in tables:
        return False
    if not set(REQUIRED_BASELINE_COLUMNS).issubset(tables):
        raise RuntimeError("Unversioned database does not match the Iris baseline tables")
    for table, required in REQUIRED_BASELINE_COLUMNS.items():
        columns = {column["name"] for column in inspector.get_columns(table)}
        if not required.issubset(columns):
            raise RuntimeError(f"Unversioned database has an incomplete {table} table")
    return True


async def prepare_baseline() -> bool:
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
        return await connection.run_sync(is_known_unversioned_baseline)


def main() -> None:
    config = Config(str(Path(__file__).resolve().parents[1] / "alembic.ini"))
    if asyncio.run(prepare_baseline()):
        command.stamp(config, BASELINE_REVISION)
    command.upgrade(config, "head")


if __name__ == "__main__":
    main()
