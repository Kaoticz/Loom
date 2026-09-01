from collections.abc import Awaitable, Callable
from typing import Any, override

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.tools import RunContext
from pydantic_ai_harness.compaction import estimate_token_count


class ObservedCompaction(AbstractCapability[Any]):
    """
    Observe a compaction capability without changing its decisions.

    :param inner: The compaction capability that may rewrite request messages.
    :param on_compaction: Receives the estimated token counts before and after a rewrite.
    """

    def __init__(
        self,
        inner: AbstractCapability[Any],
        on_compaction: Callable[[int, int], Awaitable[None]],
    ) -> None:
        """
        Create a compaction observer.

        :param inner: The capability that performs compaction.
        :param on_compaction: Receives counts when the capability shortened a request.
        """
        super().__init__()
        self._inner: AbstractCapability[Any] = inner
        self._on_compaction: Callable[[int, int], Awaitable[None]] = on_compaction

    @override
    async def before_model_request(
        self,
        ctx: RunContext[Any],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        """
        Delegate compaction and report a genuine reduction when one occurred.

        Observer failures are deliberately contained: a missing notice must not
        prevent the model request it was meant to describe.

        :param ctx: The active model run context.
        :param request_context: The model request that may be compacted.
        :returns: The request context returned by the wrapped capability.
        """
        try:
            before: int = estimate_token_count(request_context.messages)
        except Exception:
            # Imported lazily because the parent module imports this capability.
            from services.compaction import logger

            logger.exception('Could not measure request tokens before compaction.')
            return await self._inner.before_model_request(ctx, request_context)

        compacted: ModelRequestContext = await self._inner.before_model_request(ctx, request_context)
        try:
            after: int = estimate_token_count(compacted.messages)
            from services.compaction import COMPACTION_REPORT_MINIMUM_RECLAIM_TOKENS

            if before - after < COMPACTION_REPORT_MINIMUM_RECLAIM_TOKENS:
                return compacted
            await self._on_compaction(before, after)
        except Exception:
            from services.compaction import logger

            logger.exception('Could not report a compaction event.')
        return compacted
