from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.database import Base, engine
from app.database import models  # noqa: F401
from app.api.routes import (
    captions,
    clips,
    health,
    intent,
    projects,
    realtime_ws,
    search,
    session_ws,
    sessions,
    transcriptions,
)


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    # Keep simple table creation for local development; use Alembic for production migrations.
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield


async def request_validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    errors = exc.errors()
    logger.warning(
        "[validation] request=%s errors=%s",
        request.url.path,
        errors,
    )
    return JSONResponse(status_code=422, content={"detail": errors})


def create_app() -> FastAPI:
    app = FastAPI(lifespan=lifespan)
    app.add_exception_handler(RequestValidationError, request_validation_exception_handler)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[],
        allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health.router)
    app.include_router(transcriptions.router)
    app.include_router(projects.router)
    app.include_router(sessions.router)
    app.include_router(clips.router)
    app.include_router(captions.router)
    app.include_router(search.router)
    app.include_router(intent.router)
    app.include_router(session_ws.router)
    app.include_router(realtime_ws.router)
    return app
