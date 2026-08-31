from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from agents.planner import PLANNER_SYSTEM_PROMPT, StructuredPlan, planner_agent
from pydantic_ai.agent import AgentRunResult
from services.compaction import looks_like_context_overflow
from services.conversation_history import TurnRole
from services.event_bus import CompactionRunPath, ErrorEvent, SessionStatus
from services.plan_parser import NO_STEPS_CORRECTION, ParsedPlan, PlanFormat, PlanStep, parse_plan, render_plan
from services.session_fsm import SessionState

if TYPE_CHECKING:
    from .agent_session import AgentSession


logger = logging.getLogger(__name__)

CONTEXT_TOKEN_BUDGET: int = 4096


class PlanningCoordinator:
    """
    Coordinate model planning and turn its output into a session plan.

    :param session: The session owning planning state and event publication.
    """

    _session: AgentSession

    def __init__(self, session: AgentSession) -> None:
        """
        Bind the coordinator to one session.

        :param session: The owning session.
        """
        self._session = session

    async def handle_planning(self) -> SessionState:
        """
        Ask the planner for a plan and parse it.

        :returns: The next session state.
        """
        self._session._total_iterations += 1
        if self._session._total_iterations > self._session._limits.max_total_iterations:
            await self._session._events.emit(
                ErrorEvent(
                    content=(
                        f'Reached the ceiling of {self._session._limits.max_total_iterations} '
                        'iterations without completing the task. Stopping.'
                    )
                )
            )
            return SessionState.failed

        await self._session._announce('Generating plan...', SessionStatus.planning)
        self._session._history.advance_iteration()
        prompt: str = self.build_planning_prompt()
        plan: ParsedPlan = await self.plan_structured(prompt)
        if plan.is_empty:
            plan = parse_plan(await self.plan_as_text(f'{prompt}\n\n{NO_STEPS_CORRECTION}'))
        if plan.is_empty:
            await self._session._events.emit(
                ErrorEvent(content='The agent produced no actionable plan steps.')
            )
            return SessionState.failed

        self._session._plan = plan
        self._session._history.add(TurnRole.agent, render_plan(plan))
        return SessionState.waiting_plan_approval

    async def plan_structured(self, prompt: str) -> ParsedPlan:
        """
        Ask the planner for a structured plan, falling back to free text.

        :param prompt: The planning prompt.
        :returns: The plan, structured if it could be, parsed from text if not.
        """
        try:
            result: AgentRunResult[StructuredPlan] = await self._session._agent_run_helper.run_agent(
                prompt,
                planner_agent,
                lambda: self._session._planner_context_window,
                CompactionRunPath.structured_planning,
                PLANNER_SYSTEM_PROMPT,
            )
        except Exception as error:
            if looks_like_context_overflow(error):
                raise
            logger.warning('Structured planning failed, falling back to text: %s', error)
            return parse_plan(await self.plan_as_text(prompt))

        steps: list[PlanStep] = [
            PlanStep(title=step.title.strip(), description=step.description.strip())
            for step in result.output.steps
            if step.title.strip()
        ]
        if not steps:
            return parse_plan(await self.plan_as_text(prompt))
        return ParsedPlan(steps=steps, used_fallback=False, format=PlanFormat.structured)

    async def plan_as_text(self, prompt: str) -> str:
        """
        Ask the planner for a plan as prose, for `parse_plan` to read.

        :param prompt: The prompt to send.
        :returns: The planner's output, as text.
        """
        result: AgentRunResult[str] = await self._session._agent_run_helper.run_agent(
            prompt,
            planner_agent,
            lambda: self._session._planner_context_window,
            CompactionRunPath.text_planning,
            PLANNER_SYSTEM_PROMPT,
            output_type=str,
        )
        return result.output

    def build_planning_prompt(self) -> str:
        """
        Build the planning prompt, including everything learned so far.

        :returns: The prompt to send to the planner agent.
        """
        context: str = self._session._history.build_context(CONTEXT_TOKEN_BUDGET)
        sections: list[str] = [f'Task: {self._session._task}']
        if context:
            sections.append(f'What has happened so far:\n{context}')
        sections.append('Produce a step-by-step plan for this task.')
        return '\n\n'.join(sections)
