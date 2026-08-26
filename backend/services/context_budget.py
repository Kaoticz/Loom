from __future__ import annotations

import json
from typing import Any

from services.conversation_history import estimate_tokens

WORKING_HEADROOM_TOKENS: int = 4_096
"""
Room a request needs beyond its fixed overhead to be worth making at all.

Enough for the task, one step's prompt, a couple of file reads and a reply. A
window that leaves less than this is not tight, it is unusable: the model has
nowhere to put the work, and fails before generating anything.
"""


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
