from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import (
    agent,
    health,
    projects,
    realtime_ws,
    session_ws,
    sources,
    transcriptions,
)


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    # Alembic owns schema creation and migration; app startup should not mutate schema.
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
    app.include_router(sources.router)
    app.include_router(agent.router)
    app.include_router(session_ws.router)
    app.include_router(realtime_ws.router)
    return app
