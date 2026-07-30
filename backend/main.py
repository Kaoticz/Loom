from typing import Any

from environment import load_environment

# Loaded before anything reads a variable, so imports that resolve configuration
# at module scope see the file's values.
load_environment()
import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import HTTPException as FastAPIHTTPException
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
import logging
import traceback
from collections.abc import AsyncGenerator
from dishka import make_async_container, AsyncContainer
from dishka.integrations.fastapi import setup_dishka

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DEFAULT_CORS_ORIGINS: list[str] = [
    'http://localhost:5173',
    'http://127.0.0.1:5173',
]

from providers.main import MainProvider
from api.v1.router import api_router


def allowed_origins() -> list[str]:
    """
    Determine which browser origins may call the API.

    Defaults to the addresses the bundled frontend is served from. Set
    LOOM_CORS_ORIGINS to a comma separated list to serve it from anywhere else.

    :returns: The permitted origins.
    """
    configured: str = os.getenv('LOOM_CORS_ORIGINS', '')
    if configured.strip():
        return [origin.strip() for origin in configured.split(',') if origin.strip()]
    return DEFAULT_CORS_ORIGINS


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    :returns: The configured FastAPI instance.
    """
    app: FastAPI = FastAPI(title='Loom API')

    # The frontend is served from a different origin than the API, so every
    # browser fetch to it is a cross-origin request. Without this the model list,
    # the repository picker and the task history all fail in the browser, while
    # the agent session keeps working because WebSockets are exempt from CORS.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins(),
        allow_credentials=True,
        allow_methods=['*'],
        allow_headers=['*'],
    )

    app.include_router(api_router, prefix='/api/v1')

    @app.exception_handler(StarletteHTTPException)
    @app.exception_handler(FastAPIHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException | FastAPIHTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "detail": exc.detail,
                "type": "HTTPException"
            }
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception(f"Unhandled exception: {str(exc)}")
        return JSONResponse(
            status_code=500,
            content={
                "detail": str(exc),
                "type": type(exc).__name__
            }
        )
    
    container: AsyncContainer = make_async_container(MainProvider())
    setup_dishka(container, app)
    
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        """
        Lifecycle events for the FastAPI application.
        """
        from tortoise import Tortoise
        from database import vector_client as vector_client  # noqa: F401  installs the pgvector client
        from database.setup import prepare_database
        from database.tortoise_config import build_config, database_url

        # Migrations own the schema. Generating it from the models instead would
        # create missing tables but never alter existing ones, so an upgraded
        # installation would silently run against an out-of-date schema.
        # Built from the environment now, not frozen at import time.
        config: dict[str, Any] = build_config()
        await prepare_database(database_url(), config)

        # The lifespan runs in a different task from request handlers, and
        # Tortoise 1.x keeps its connections in a ContextVar. Without the global
        # fallback, every database query outside this task fails with
        # "No TortoiseContext is currently active".
        await Tortoise.init(config=config, _enable_global_fallback=True)

        # Sessions exist only in memory, so anything still marked running belongs
        # to a process that is gone.
        from services.task_recorder import fail_interrupted_tasks

        await fail_interrupted_tasks()

        yield
        await Tortoise.close_connections()

    app.router.lifespan_context = lifespan
    
    return app

app: FastAPI = create_app()
