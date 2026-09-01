from __future__ import annotations

from collections import Counter
from logging import Logger, getLogger
from typing import Any

from database.enums import TaskLogEntryType, TaskStatus
from database.models.task import Task
from database.models.task_memory import TaskMemory
from services.events import (
    BaseAgentEvent,
    CommitEvent,
    PlanEvent,
    ReviewEvent,
    ToolObservationEvent,
)

logger: Logger = getLogger(__name__)

DEFAULT_MAX_ENTRIES: int = 200
DEFAULT_COMPRESS_BATCH: int = 100


class TaskRecorder:
    """
    Persists what happens during a session into the task's action log.

    Rather than sprinkling writes through the state machine, this subscribes to
    the session's event bus: tools, commits and reviews already announce
    themselves there, so nothing can be recorded in one place and forgotten in
    another. Events the log does not care about, such as streamed thoughts and
    file diffs, are ignored.
    """

    _task: Task
    _max_entries: int
    _compress_batch: int
    _written: int

    def __init__(
        self,
        task: Task,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        compress_batch: int = DEFAULT_COMPRESS_BATCH,
    ) -> None:
        """
        Create a recorder for one task.

        :param task: The task whose log is being written.
        :param max_entries: How many entries may accumulate before older ones are collapsed.
        :param compress_batch: How many of the oldest entries to collapse when the limit is passed.
        :raises ValueError: If the compression batch is not smaller than the limit.
        """
        if compress_batch < 1 or compress_batch >= max_entries:
            raise ValueError('The compression batch must be at least one and smaller than the limit.')
        self._task = task
        self._max_entries = max_entries
        self._compress_batch = compress_batch
        self._written = 0

    @property
    def task(self) -> Task:
        """
        The task being recorded.

        :returns: The task.
        """
        return self._task

    @property
    def written(self) -> int:
        """
        How many entries this recorder has written.

        :returns: The number of entries written, before any compression.
        """
        return self._written

    async def record(
        self,
        entry_type: TaskLogEntryType,
        content: dict[str, Any],
    ) -> TaskMemory:
        """
        Append an entry to the task's log.

        :param entry_type: What kind of event is being recorded.
        :param content: The details of the event.
        :returns: The stored entry.
        """
        entry: TaskMemory = await TaskMemory.create(
            task=self._task, entry_type=entry_type, content=content
        )
        self._written += 1
        await self._compress_if_needed()
        return entry

    async def handle_event(self, event: BaseAgentEvent) -> None:
        """
        Record an event emitted onto the session's event bus.

        Failures are logged rather than raised: losing a log entry must not
        interrupt the work the entry was describing.

        :param event: The event to consider recording.
        """
        try:
            if isinstance(event, ToolObservationEvent):
                await self.record(
                    TaskLogEntryType.tool_call,
                    {
                        'tool': event.tool_name,
                        'status': event.status.value,
                        'result': event.result[:500],
                    },
                )
            elif isinstance(event, CommitEvent):
                await self.record(
                    TaskLogEntryType.commit,
                    {
                        'hash': event.commit_hash,
                        'message': event.message,
                        'files': list(event.files_changed),
                    },
                )
            elif isinstance(event, ReviewEvent):
                await self.record(
                    TaskLogEntryType.review,
                    {'approved': event.approved, 'comments': event.comments},
                )
            elif isinstance(event, PlanEvent):
                # Stored structured. Flattening to titles here would lose the
                # descriptions from the task history, where they are most useful
                # long after the session that produced them has gone.
                await self.record(
                    TaskLogEntryType.plan,
                    {
                        'format': event.format.value,
                        'steps': [step.model_dump() for step in event.steps],
                    },
                )
        except Exception as error:
            logger.warning('Could not record a %s entry: %s', type(event).__name__, error)

    async def record_rejection(self, stage: str, reason: str) -> None:
        """
        Record that the user sent work back.

        :param stage: What was rejected, for example 'plan' or 'result'.
        :param reason: The feedback the user gave, if any.
        """
        await self.record(
            TaskLogEntryType.rejection,
            {'stage': stage, 'reason': reason or '(no reason given)'},
        )

    async def record_status(self, status: TaskStatus, summary: str = '') -> Task:
        """
        Update the task's status and note the change in the log.

        :param status: The status the task has moved to.
        :param summary: A summary to store alongside it, when one is available.
        :returns: The updated task.
        """
        self._task.status = status
        if summary:
            self._task.summary = summary
        await self._task.save()

        await self.record(TaskLogEntryType.status_change, {'status': status.value})
        return self._task

    async def _compress_if_needed(self) -> None:
        """
        Collapse the oldest entries once the log grows past its limit.

        Old detail is replaced with counts rather than deleted outright, so the
        shape of what happened survives even after the specifics are gone.
        """
        total: int = await TaskMemory.filter(task=self._task).count()
        if total <= self._max_entries:
            return

        oldest: list[TaskMemory] = (
            await TaskMemory.filter(task=self._task)
            .order_by('created_at', 'id')
            .limit(self._compress_batch)
        )
        if not oldest:
            return

        counts: Counter[str] = Counter(entry.entry_type.value for entry in oldest)
        files: list[str] = []
        for entry in oldest:
            for path in entry.content.get('files', []):
                if isinstance(path, str) and path not in files:
                    files.append(path)

        summary_content: dict[str, Any] = {
            'collapsed_entries': len(oldest),
            'counts': dict(counts),
            'files_touched': files[:50],
        }

        await TaskMemory.filter(id__in=[entry.id for entry in oldest]).delete()
        await TaskMemory.create(
            task=self._task,
            entry_type=TaskLogEntryType.summary,
            content=summary_content,
        )


async def fail_interrupted_tasks() -> int:
    """
    Mark tasks left running by a previous process as failed.

    Sessions live only in memory, so a task still marked `running` at startup
    belongs to a process that no longer exists. Leaving it be would show a task
    as in progress forever.

    This assumes a single backend instance. With more than one, a task belonging
    to a live sibling process would be wrongly marked failed.

    :returns: How many tasks were marked failed.
    """
    interrupted: list[Task] = await Task.filter(status=TaskStatus.running)
    for task in interrupted:
        task.status = TaskStatus.failed
        await task.save()
        await TaskMemory.create(
            task=task,
            entry_type=TaskLogEntryType.status_change,
            content={
                'status': TaskStatus.failed.value,
                'reason': 'The session was interrupted by a backend restart.',
            },
        )

    if interrupted:
        logger.info('Marked %d interrupted task(s) as failed.', len(interrupted))
    return len(interrupted)
