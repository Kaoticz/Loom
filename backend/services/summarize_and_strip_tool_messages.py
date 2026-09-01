from dataclasses import dataclass, replace
from typing import Any

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelRequestPart, ModelResponsePart, ToolCallPart, ToolReturnPart, UserPromptPart
from pydantic_ai.tools import RunContext
from pydantic_ai_harness.compaction import SummarizingCompaction


@dataclass
class SummarizeAndStripToolMessages(AbstractCapability[Any]):
    """
    Summarize history and then remove replaceable tool messages in one tier.

    The enclosing `TieredCompaction` calls this tier only after the history is
    above its target. The tool-part strip lives inside `compact`, rather than as
    a later tier, so it always follows a summary even when that summary alone
    brings the history back below the target.

    :param summarizer: The strategy that summarizes the older portion of history.
    :param exclude_tools: Tool names whose call/result parts remain intact.
    """

    summarizer: SummarizingCompaction[Any]
    exclude_tools: frozenset[str]

    def with_focus(self, focus: str) -> SummarizeAndStripToolMessages:
        """
        Prioritize focus in the summary without changing stripping behavior.

        :param focus: The topic the summary must preserve.
        :returns: An equivalent combined tier whose summary receives the focus.
        """
        return replace(self, summarizer=self.summarizer.with_focus(focus))

    async def compact(
        self,
        messages: list[ModelMessage],
        ctx: RunContext[Any],
    ) -> list[ModelMessage]:
        """
        Summarize history and remove nonessential tool parts.

        Retain the final request intact when stripping its parts would otherwise
        remove the current step from the history. A final request that is empty
        or contains only replaceable tool parts is backfilled with the latest
        original user prompt so it remains a wire-mappable user message after
        tool-pair cleanup.

        :param messages: The history that exceeded the enclosing ladder's target.
        :param ctx: The running model context used by the summarizer.
        :returns: The summarized history with replaceable tool parts removed.
        """
        # Imported lazily because the parent module instantiates this capability.
        from services.compaction import (
            EMPTY_TERMINAL_REQUEST_FALLBACK,
            _latest_user_prompt,
            _strip_redundant_system_prompts,
        )

        summarized: list[ModelMessage] = await self.summarizer.compact(messages, ctx)
        stripped: list[ModelMessage] = []
        final_message_index: int = len(summarized) - 1
        for index, message in enumerate(summarized):
            if isinstance(message, ModelRequest):
                parts: list[ModelRequestPart] = [
                    part for part in message.parts if not self._is_replaceable_tool_part(part)
                ]
                if parts:
                    stripped.append(replace(message, parts=parts))
                elif index == final_message_index:
                    stripped.append(message)
                continue

            response_parts: list[ModelResponsePart] = [
                part for part in message.parts if not self._is_replaceable_tool_part(part)
            ]
            if response_parts:
                stripped.append(replace(message, parts=response_parts))

        if (
            stripped
            and isinstance(stripped[-1], ModelRequest)
            and all(self._is_replaceable_tool_part(part) for part in stripped[-1].parts)
        ):
            prompt: UserPromptPart | None = _latest_user_prompt(messages)
            terminal_part: UserPromptPart = prompt or UserPromptPart(EMPTY_TERMINAL_REQUEST_FALLBACK)
            stripped[-1] = replace(stripped[-1], parts=[terminal_part])
        return _strip_redundant_system_prompts(stripped)

    def _is_replaceable_tool_part(self, part: ModelRequestPart | ModelResponsePart) -> bool:
        """
        Identify a tool part that can be removed from the working history.

        :param part: One request or response message part.
        :returns: True when the part belongs to a nonessential tool.
        """
        return (
            isinstance(part, (ToolCallPart, ToolReturnPart))
            and part.tool_name not in self.exclude_tools
        )
