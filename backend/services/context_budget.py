from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, override

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelResponse

from services.conversation_history import estimate_tokens
from services.provider_usage_divergence import ProviderUsageDivergence

if TYPE_CHECKING:
    from pydantic_ai.models import ModelRequestContext
    from pydantic_ai.tools import RunContext

WORKING_HEADROOM_TOKENS: int = 4_096
"""
Room a request needs beyond its fixed overhead to be worth making at all.

Enough for the task, one step's prompt, a couple of file reads and a reply. A
window that leaves less than this is not tight, it is unusable: the model has
nowhere to put the work, and fails before generating anything.
"""

PROVIDER_UNDER_ESTIMATE_RATIO: float = 0.5
"""
The largest provider-to-Loom input ratio that signals likely server truncation.

Small rounding differences are normal for a character-based estimator. A provider
seeing at most half of Loom's estimate instead points to it discarding part of the
prompt, commonly because its actual ``num_ctx`` is smaller than requested.
"""

PROVIDER_OVER_ESTIMATE_RATIO: float = 1.5
"""
The smallest provider-to-Loom input ratio that signals unseen prompt growth.

At this distance the provider is accounting for material content Loom's heuristic
did not see, so later compaction budgets need an additional reserve.
"""


def provider_usage_divergence(
    estimated_input_tokens: int, provider_input_tokens: int
) -> ProviderUsageDivergence:
    """
    Classify a provider's prompt-token accounting against Loom's estimate.

    :param estimated_input_tokens: Loom's estimated tokens in the sent prompt.
    :param provider_input_tokens: Tokens the provider reports processing.
    :returns: The meaningful direction of divergence, if there is one.
    """
    if estimated_input_tokens <= 0 or provider_input_tokens <= 0:
        return ProviderUsageDivergence.unavailable
    if provider_input_tokens <= estimated_input_tokens * PROVIDER_UNDER_ESTIMATE_RATIO:
        return ProviderUsageDivergence.input_truncated
    if provider_input_tokens >= estimated_input_tokens * PROVIDER_OVER_ESTIMATE_RATIO:
        return ProviderUsageDivergence.estimate_too_low
    return ProviderUsageDivergence.agreement


def provider_usage_correction(estimated_input_tokens: int, provider_input_tokens: int) -> int:
    """
    Calculate the extra context reserve justified by provider usage.

    :param estimated_input_tokens: Loom's estimated tokens in the sent prompt.
    :param provider_input_tokens: Tokens the provider reports processing.
    :returns: Extra tokens later compaction budgets should reserve.
    """
    if (
        provider_usage_divergence(estimated_input_tokens, provider_input_tokens)
        is not ProviderUsageDivergence.estimate_too_low
    ):
        return 0
    return provider_input_tokens - estimated_input_tokens


class ReportProviderUsage(AbstractCapability[Any]):
    """
    Report provider input-token usage after each successful model response.

    The existing harness monitor reports the matching estimate before each request.
    Keeping this capability deliberately narrow lets the session reconcile those
    two measurements without changing the request, response, or usage-gauge
    contracts.

    :param on_usage: Receives positive provider-reported input token counts.
    """

    def __init__(self, on_usage: Callable[[int], Awaitable[None]]) -> None:
        """
        Create the response-usage reporter.

        :param on_usage: Receives positive provider-reported input token counts.
        """
        super().__init__()
        self._on_usage: Callable[[int], Awaitable[None]] = on_usage

    @override
    async def after_model_request(
        self,
        ctx: RunContext[Any],
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        """
        Pass a successful response's prompt usage to the host application.

        :param ctx: The active agent run context, unused by this observer.
        :param request_context: The request that received the response, unused.
        :param response: The successful provider response.
        :returns: The original response, unchanged.
        """
        input_tokens: int = response.usage.input_tokens
        if input_tokens > 0:
            await self._on_usage(input_tokens)
        return response


def estimate_fixed_overhead(system_prompt: str, agent: Any) -> int:
    """
    Estimate the tokens every request carries before any conversation.

    The system prompt and the tool schemas are resent on each request and are
    invisible to compaction, which budgets the message history alone. They are
    therefore a floor: no amount of trimming can bring a request below them, and
    a window smaller than this one cannot be worked in at all.

    Reads the agent's registered tools directly. Pydantic-ai exposes no public
    accessor for the rendered schemas, and estimating from the source functions
    instead would miss exactly what is sent over the wire.

    :param system_prompt: The instructions sent with every request.
    :param agent: The agent whose tools will be offered.
    :returns: The estimated token count.
    """
    tools: dict[str, Any] = getattr(agent, '_function_toolset').tools
    rendered: str = json.dumps([
        {
            'name': name,
            'description': tool.tool_def.description,
            'parameters': tool.tool_def.parameters_json_schema,
        }
        for name, tool in tools.items()
    ])
    return estimate_tokens(system_prompt) + estimate_tokens(rendered)


def context_window_complaint(window: int, overhead: int) -> str:
    """
    Explain that a context window cannot fit what every request must carry.

    Written for the person, not the model: the provider's own error for this
    ("token limit exceeded before any response was generated") names neither the
    window nor what filled it, so it reads as a bug in Loom rather than as a
    setting to change.

    :param window: The context window being budgeted against.
    :param overhead: The fixed per-request overhead.
    :returns: The explanation, or an empty string if the window is workable.
    """
    if window >= overhead + WORKING_HEADROOM_TOKENS:
        return ''

    return (
        f'The context window in use is about {window:,} tokens, but every request carries '
        f'roughly {overhead:,} tokens of instructions and tool definitions before any of your '
        'task is added. Compaction cannot help with that part: it trims the conversation, and '
        'this is sent fresh every time.\n\n'
        f'Runs are likely to fail before the model replies at all. Either raise the model\'s '
        f'context size (for Ollama, num_ctx on the model itself), or set the coder context '
        f'window in Settings to match what the model really allows. Around '
        f'{overhead + WORKING_HEADROOM_TOKENS:,} tokens is the practical minimum.'
    )
