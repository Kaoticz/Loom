from typing import Any, override

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelMessage

from pydantic_ai.models import ModelRequestContext
from pydantic_ai.tools import RunContext


class RequestHistoryCapture(AbstractCapability[Any]):
    """
    Retain the most recent compacted request history for recovery.

    The agent discards its graph state when a model request raises. Capturing the
    history after ordinary capabilities run lets the session retry from a forced
    summary instead of throwing that useful context away before narrowing.
    """

    def __init__(self) -> None:
        """
        Create an empty request-history capture.
        """
        super().__init__()
        self._messages: list[ModelMessage] = []

    @property
    def messages(self) -> list[ModelMessage]:
        """
        Return a copy of the last captured request history.

        :returns: The history prepared for the most recent model request.
        """
        return list(self._messages)

    @override
    async def before_model_request(
        self,
        ctx: RunContext[Any],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        """
        Save the already-compacted history without changing the request.

        :param ctx: The active model run context, unused by this observer.
        :param request_context: The request whose prepared history is retained.
        :returns: The original request context, unchanged.
        """
        self._messages = list(request_context.messages)
        return request_context
