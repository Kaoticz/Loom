from fastapi import FastAPI, Request
from fastapi.exceptions import HTTPException as FastAPIHTTPException
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from dishka import make_async_container
from dishka.integrations.fastapi import setup_dishka

from providers.main import MainProvider
from api.v1.router import api_router
from database.tortoise_config import TORTOISE_ORM

def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    :returns: The configured FastAPI instance.
    """
    app: FastAPI = FastAPI(title='Loom API')
    
    app.include_router(api_router, prefix='/api/v1')

    @app.exception_handler(StarletteHTTPException)
    @app.exception_handler(FastAPIHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException | FastAPIHTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "detail": exc.detail,
                "type": "HTTPException"
            }
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=500,
            content={
                "detail": str(exc),
                "type": type(exc).__name__
            }
        )
    
    container = make_async_container(MainProvider())
    setup_dishka(container, app)
    
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """
        Lifecycle events for the FastAPI application.
        """
        from tortoise import Tortoise
        # Initialize Tortoise
        await Tortoise.init(config=TORTOISE_ORM)
        
        # Ensure pgvector is initialized BEFORE generating schemas
        connection = Tortoise.get_connection('default')
        await connection.execute_script('CREATE EXTENSION IF NOT EXISTS vector;')
        
        await Tortoise.generate_schemas()
        
        yield
        await Tortoise.close_connections()

    app.router.lifespan_context = lifespan
    
    return app

app: FastAPI = create_app()
