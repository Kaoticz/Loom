from typing import Annotated

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """
    Schema for the service health check.
    """
    status: Annotated[str, Field(description="'ok' when the service can serve requests.")]
    database: Annotated[bool, Field(description='Whether the database answered a trivial query.')]
