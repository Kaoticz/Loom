from __future__ import annotations

from collections.abc import Iterable, Sequence
from enum import Enum
from typing import Annotated, Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

CHARS_PER_PROSE_TOKEN: int = 4
CHARS_PER_CODE_TOKEN: int = 2
DEFAULT_MAX_TOKENS: int = 4096
DEFAULT_RECENT_TURNS_KEPT: int = 3

FILE_WRITING_TOOLS: frozenset[str] = frozenset({'edit_file', 'create_file'})
TEST_RUNNING_TOOLS: frozenset[str] = frozenset({'run_test', 'run_test_command'})


class TurnRole(str, Enum):
    """
    The origin of a single turn in the accumulated conversation.
    """

    user = 'user'
    agent = 'agent'
    tool_call = 'tool_call'
    tool_result = 'tool_result'
    reviewer = 'reviewer'
    system = 'system'


def estimate_tokens(text: str, is_code: bool = False) -> int:
    """
    Approximate the number of tokens a piece of text occupies.

    A character-count heuristic is used deliberately: this runs locally on every
    agent iteration, so speed matters more than the accuracy a real tokeniser
    would buy.

    :param text: The text to measure.
    :param is_code: Whether the text is source code, which tokenises more densely.
    :returns: The estimated token count.
    """
    if not text:
        return 0
    divisor: int = CHARS_PER_CODE_TOKEN if is_code else CHARS_PER_PROSE_TOKEN
    return max(1, -(-len(text) // divisor))


def truncate_to_tokens(text: str, max_tokens: int, is_code: bool = False) -> str:
    """
    Shorten text so that its estimated token count fits within a budget.

    :param text: The text to shorten.
    :param max_tokens: The maximum number of tokens the result may occupy.
    :param is_code: Whether the text is source code, which tokenises more densely.
    :returns: The original text, or a truncated copy marked with an ellipsis notice.
    """
    if max_tokens <= 0:
        return ''
    if estimate_tokens(text, is_code) <= max_tokens:
        return text

    notice: str = '\n... (truncated)'
    divisor: int = CHARS_PER_CODE_TOKEN if is_code else CHARS_PER_PROSE_TOKEN
    allowed_chars: int = max(0, max_tokens * divisor - len(notice))
    if allowed_chars == 0:
        return notice.strip()
    return text[:allowed_chars] + notice


class ConversationTurn(BaseModel):
    """
    A single exchange in the accumulated agent conversation.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    role: Annotated[TurnRole, Field()]
    content: Annotated[str, Field()]
    iteration: Annotated[int, Field(ge=0)] = 0
    tool_name: Annotated[str | None, Field()] = None
    metadata: Annotated[dict[str, Any], Field(default_factory=dict)]
    is_code: Annotated[bool, Field()] = False

    def render(self) -> str:
        """
        Render this turn as a labelled block for injection into a prompt.

        :returns: The rendered representation of the turn.
        """
        if self.role is TurnRole.tool_call:
            return f'[tool_call] {self.tool_name or "unknown"}: {self.content}'
        if self.role is TurnRole.tool_result:
            return f'[tool_result] {self.tool_name or "unknown"}: {self.content}'
        return f'[{self.role.value}] {self.content}'

    def estimated_tokens(self) -> int:
        """
        Approximate the token cost of rendering this turn.

        :returns: The estimated token count of the rendered turn.
        """
        return estimate_tokens(self.render(), self.is_code)


class ConversationHistory:
    """
    Accumulates the agent's turns across iterations and renders them back into a
    prompt-sized context block.

    PydanticAI agents are stateless per `run()` call, so the iterative
    plan-act-review loop has to carry its own memory. The most recent turns are
    preserved verbatim while older ones collapse into per-iteration summaries,
    which keeps the freshest reasoning intact under a fixed token budget.
    """

    _turns: list[ConversationTurn]
    _recent_turns_kept: int
    _iteration: int

    def __init__(self, recent_turns_kept: int = DEFAULT_RECENT_TURNS_KEPT) -> None:
        """
        Create an empty conversation history.

        :param recent_turns_kept: How many trailing turns to preserve verbatim when compressing.
        :raises ValueError: If fewer than one recent turn would be preserved.
        """
        if recent_turns_kept < 1:
            raise ValueError('At least one recent turn must be preserved.')
        self._turns = []
        self._recent_turns_kept = recent_turns_kept
        self._iteration = 0

    @property
    def turns(self) -> Sequence[ConversationTurn]:
        """
        Every turn recorded so far, in order.

        :returns: An immutable view over the recorded turns.
        """
        return tuple(self._turns)

    @property
    def iteration(self) -> int:
        """
        The index of the iteration currently being recorded.

        :returns: The current iteration index.
        """
        return self._iteration

    def advance_iteration(self) -> int:
        """
        Begin a new iteration of the plan-act-review loop.

        :returns: The new iteration index.
        """
        self._iteration += 1
        return self._iteration

    def add_turn(self, turn: ConversationTurn) -> ConversationTurn:
        """
        Record a pre-built turn.

        :param turn: The turn to record.
        :returns: The recorded turn.
        """
        self._turns.append(turn)
        return turn

    def add(
        self,
        role: TurnRole,
        content: str,
        tool_name: str | None = None,
        metadata: dict[str, Any] | None = None,
        is_code: bool = False,
    ) -> ConversationTurn:
        """
        Record a new turn in the current iteration.

        :param role: Who or what produced this turn.
        :param content: The textual content of the turn.
        :param tool_name: The tool involved, for tool call and tool result turns.
        :param metadata: Structured details used when the turn is later summarised.
        :param is_code: Whether the content is source code, which tokenises more densely.
        :returns: The recorded turn.
        """
        return self.add_turn(
            ConversationTurn(
                role=role,
                content=content,
                iteration=self._iteration,
                tool_name=tool_name,
                metadata=metadata or {},
                is_code=is_code,
            )
        )

    def build_context(self, max_tokens: int = DEFAULT_MAX_TOKENS) -> str:
        """
        Render the accumulated history into a context block that fits a token budget.

        Turns beyond the most recent few are collapsed into one summary line per
        iteration. If the budget is still exceeded, further recent turns are
        folded into the summary, and as a last resort the remaining turn is
        truncated.

        :param max_tokens: The maximum estimated token count of the returned block.
        :returns: The rendered context, or an empty string when there is no history.
        """
        if not self._turns:
            return ''
        if max_tokens <= 0:
            return ''

        if self._fits(self._turns, max_tokens):
            return self._render_block([], self._turns)

        keep: int = min(self._recent_turns_kept, len(self._turns))
        while keep >= 1:
            older: list[ConversationTurn] = self._turns[: len(self._turns) - keep]
            recent: list[ConversationTurn] = self._turns[len(self._turns) - keep :]
            candidate: str = self._render_block(older, recent)
            if estimate_tokens(candidate) <= max_tokens:
                return candidate
            keep -= 1

        summary_only: str = self._render_block(self._turns[:-1], [])
        last_turn: ConversationTurn = self._turns[-1]
        remaining: int = max_tokens - estimate_tokens(summary_only)
        if remaining <= 0:
            return truncate_to_tokens(summary_only, max_tokens)
        return summary_only + '\n' + truncate_to_tokens(
            last_turn.render(), remaining, last_turn.is_code
        )

    def _fits(self, turns: Sequence[ConversationTurn], max_tokens: int) -> bool:
        """
        Check whether rendering the given turns in full stays within budget.

        :param turns: The turns to measure.
        :param max_tokens: The token budget.
        :returns: True if the rendered turns fit within the budget.
        """
        return estimate_tokens(self._render_block([], turns)) <= max_tokens

    def _render_block(
        self,
        older: Sequence[ConversationTurn],
        recent: Sequence[ConversationTurn],
    ) -> str:
        """
        Render a summary of older turns followed by recent turns in full.

        :param older: The turns to summarise.
        :param recent: The turns to render verbatim.
        :returns: The rendered context block.
        """
        sections: list[str] = []
        if older:
            sections.append('## Summary of earlier work\n' + self._summarise(older))
        if recent:
            body: str = '\n'.join(turn.render() for turn in recent)
            sections.append('## Recent turns\n' + body if older else body)
        return '\n\n'.join(sections)

    def _summarise(self, turns: Sequence[ConversationTurn]) -> str:
        """
        Collapse turns into one compact line per iteration.

        :param turns: The turns to summarise.
        :returns: The summary text.
        """
        iterations: list[int] = sorted({turn.iteration for turn in turns})
        lines: list[str] = []
        for iteration in iterations:
            slice_: list[ConversationTurn] = [t for t in turns if t.iteration == iteration]
            lines.append(self._summarise_iteration(iteration, slice_))
        return '\n'.join(lines)

    def _summarise_iteration(
        self,
        iteration: int,
        turns: Sequence[ConversationTurn],
    ) -> str:
        """
        Summarise a single iteration's worth of turns into one line.

        :param iteration: The iteration index being summarised.
        :param turns: The turns belonging to that iteration.
        :returns: A single-line summary of the iteration.
        """
        facts: list[str] = []

        written: list[str] = _unique(
            str(turn.metadata.get('path', '')).strip()
            for turn in turns
            if turn.role is TurnRole.tool_call
            and turn.tool_name in FILE_WRITING_TOOLS
            and str(turn.metadata.get('path', '')).strip()
        )
        if written:
            facts.append(f'modified {", ".join(written)}')

        test_results: list[ConversationTurn] = [
            turn
            for turn in turns
            if turn.role is TurnRole.tool_result and turn.tool_name in TEST_RUNNING_TOOLS
        ]
        if test_results:
            failures: int = sum(
                1 for turn in test_results if turn.metadata.get('success') is False
            )
            facts.append(
                f'ran {len(test_results)} test command(s)'
                + (f', {failures} failed' if failures else ', all passed')
            )

        other_tools: list[str] = _unique(
            turn.tool_name or ''
            for turn in turns
            if turn.role is TurnRole.tool_call
            and turn.tool_name not in FILE_WRITING_TOOLS
            and turn.tool_name not in TEST_RUNNING_TOOLS
            and turn.tool_name
        )
        if other_tools:
            facts.append(f'used {", ".join(other_tools)}')

        for turn in turns:
            if turn.role is not TurnRole.reviewer:
                continue
            approved: Any = turn.metadata.get('approved')
            verdict: str = 'approved' if approved is True else 'rejected'
            facts.append(f'reviewer {verdict}: {_shorten(turn.content)}')

        for turn in turns:
            if turn.role is TurnRole.user and turn.metadata.get('rejection') is True:
                facts.append(f'user rejected the plan: {_shorten(turn.content)}')

        if not facts:
            agent_turns: list[ConversationTurn] = [t for t in turns if t.role is TurnRole.agent]
            if agent_turns:
                facts.append(_shorten(agent_turns[-1].content))
            else:
                facts.append(f'{len(turns)} turn(s) with no notable actions')

        return f'- In iteration {iteration}, you {"; ".join(facts)}.'


def _unique(values: Iterable[str]) -> list[str]:
    """
    Deduplicate strings while preserving their first-seen order.

    :param values: The values to deduplicate.
    :returns: The unique values in their original order.
    """
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _shorten(text: str, limit: int = 160) -> str:
    """
    Collapse text to a single line no longer than the given limit.

    :param text: The text to shorten.
    :param limit: The maximum number of characters to keep.
    :returns: The shortened, single-line text.
    """
    collapsed: str = ' '.join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[:limit].rstrip() + '...'
