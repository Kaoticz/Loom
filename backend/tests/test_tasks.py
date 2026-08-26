from __future__ import annotations

import os
from collections.abc import AsyncIterator
from uuid import uuid4

import pytest
import pytest_asyncio
from fastapi import HTTPException
from tortoise import Tortoise

from api.v1.tasks_controller import MAX_PAGE_SIZE, TaskController
from database.enums import TaskLogEntryType, TaskStatus
from database.models.task import Task
from database.models.task_memory import TaskMemory
from schemas.task_request import TaskRequest
from schemas.task_response import TaskPatchRequest
from services.event_bus import CommitEvent, PlanEvent, ReviewEvent, ThoughtEvent, ToolObservationEvent, ToolStatus
from services.task_recorder import TaskRecorder
from services.plan_parser import PlanStep

pytestmark = pytest.mark.usefixtures('require_database')


@pytest_asyncio.fixture
async def task_db() -> AsyncIterator[None]:
    """
    Provide a Postgres database with the task tables freshly built.

    :returns: An iterator yielding once the schema is ready.
    """
    from tests.conftest import postgres_config
    from tortoise.backends.base.executor import EXECUTOR_CACHE

    EXECUTOR_CACHE.clear()
    await Tortoise.init(config=postgres_config(os.environ['TEST_DATABASE_URL']))
    connection = Tortoise.get_connection('default')
    await connection.execute_script(
        'DROP TABLE IF EXISTS code_embeddings, symbols, task_memory, tasks, files, '
        'embedding_models CASCADE;'
    )
    await Tortoise.generate_schemas(safe=True)
    yield
    await TaskMemory.all().delete()
    await Task.all().delete()
    await Tortoise.close_connections()
    EXECUTOR_CACHE.clear()


@pytest.fixture
def controller() -> TaskController:
    """
    Provide a task controller.

    :returns: The controller.
    """
    return TaskController()


async def make_task(prompt: str = 'Do the thing') -> Task:
    """
    Insert a task directly.

    :param prompt: The task's prompt.
    :returns: The created task.
    """
    return await Task.create(id=uuid4(), prompt=prompt, status=TaskStatus.initialized)


@pytest.mark.usefixtures('task_db')
async def test_create_task_persists_a_record(controller: TaskController) -> None:
    """
    Creating a task stores it with the initialized status.
    """
    response = await controller.create_task(
        TaskRequest(prompt='Add a timeout', repo_path='/repo', coder_model='ollama:llama3')
    )

    assert response.status is TaskStatus.initialized
    assert response.prompt == 'Add a timeout'

    stored = await Task.get_or_none(id=response.id)
    assert stored is not None
    assert stored.coder_model == 'ollama:llama3'


@pytest.mark.usefixtures('task_db')
async def test_list_tasks_is_newest_first_and_paginated(controller: TaskController) -> None:
    """
    Listing returns a page of tasks, newest first, with the total count.
    """
    for index in range(5):
        await make_task(f'task {index}')

    page = await controller.list_tasks(offset=0, limit=2)
    assert page.total == 5
    assert len(page.items) == 2
    assert page.items[0].created_at >= page.items[1].created_at

    second = await controller.list_tasks(offset=2, limit=2)
    assert {item.id for item in page.items}.isdisjoint({item.id for item in second.items})


@pytest.mark.usefixtures('task_db')
@pytest.mark.parametrize(
    'offset,limit', [(-1, 20), (0, 0), (0, MAX_PAGE_SIZE + 1)],
    ids=['negative_offset', 'zero_limit', 'oversized_limit'],
)
async def test_bad_pagination_is_rejected(
    controller: TaskController, offset: int, limit: int
) -> None:
    """
    Out-of-range pagination is refused rather than silently clamped.
    """
    with pytest.raises(HTTPException) as raised:
        await controller.list_tasks(offset=offset, limit=limit)
    assert raised.value.status_code == 422


@pytest.mark.usefixtures('task_db')
async def test_get_task_returns_its_log(controller: TaskController) -> None:
    """
    Fetching a task includes its action log, oldest entry first.
    """
    task = await make_task()
    recorder = TaskRecorder(task)
    await recorder.record(TaskLogEntryType.plan, {'steps': ['one']})
    await recorder.record(TaskLogEntryType.commit, {'hash': 'abc', 'message': 'Edit a.py'})

    detail = await controller.get_task(str(task.id))

    assert detail.task.id == str(task.id)
    assert [entry.entry_type for entry in detail.log] == [
        TaskLogEntryType.plan,
        TaskLogEntryType.commit,
    ]
    assert detail.log[1].content['hash'] == 'abc'


@pytest.mark.usefixtures('task_db')
@pytest.mark.parametrize('identifier', ['not-a-uuid', '00000000-0000-0000-0000-000000000000'])
async def test_missing_task_is_a_404(controller: TaskController, identifier: str) -> None:
    """
    An unknown or malformed identifier is a not-found, not a server error.
    """
    with pytest.raises(HTTPException) as raised:
        await controller.get_task(identifier)
    assert raised.value.status_code == 404


@pytest.mark.usefixtures('task_db')
async def test_patch_updates_status_and_summary(controller: TaskController) -> None:
    """
    A task's status and summary can be updated independently.
    """
    task = await make_task()

    updated = await controller.update_task(
        str(task.id), TaskPatchRequest(status=TaskStatus.completed)
    )
    assert updated.status is TaskStatus.completed

    updated = await controller.update_task(
        str(task.id), TaskPatchRequest(summary='All done.')
    )
    assert updated.summary == 'All done.'
    assert updated.status is TaskStatus.completed


@pytest.mark.usefixtures('task_db')
async def test_empty_patch_is_rejected(controller: TaskController) -> None:
    """
    A patch that changes nothing is refused rather than silently succeeding.
    """
    task = await make_task()
    with pytest.raises(HTTPException) as raised:
        await controller.update_task(str(task.id), TaskPatchRequest())
    assert raised.value.status_code == 422


@pytest.mark.usefixtures('task_db')
async def test_recorder_persists_events_from_the_bus() -> None:
    """
    Events emitted during a session become log entries.
    """
    task = await make_task()
    recorder = TaskRecorder(task)

    await recorder.handle_event(PlanEvent(steps=[PlanStep(title='read'), PlanStep(title='edit')]))
    await recorder.handle_event(
        ToolObservationEvent(tool_name='edit_file', result='Edited a.py')
    )
    await recorder.handle_event(
        CommitEvent(commit_hash='abc123', message='Edit a.py', files_changed=['a.py'])
    )
    await recorder.handle_event(ReviewEvent(approved=False, comments='Needs a guard.'))

    entries = await TaskMemory.filter(task=task).order_by('id')
    assert [entry.entry_type for entry in entries] == [
        TaskLogEntryType.plan,
        TaskLogEntryType.tool_call,
        TaskLogEntryType.commit,
        TaskLogEntryType.review,
    ]
    assert entries[2].content['files'] == ['a.py']


@pytest.mark.usefixtures('task_db')
async def test_recorder_ignores_uninteresting_events() -> None:
    """
    Streamed reasoning is not written to the log.

    A thought arrives for every sentence the model produces; recording them
    would bury the actions the log exists to show.
    """
    task = await make_task()
    recorder = TaskRecorder(task)

    await recorder.handle_event(ThoughtEvent(content='thinking out loud'))

    assert await TaskMemory.filter(task=task).count() == 0


@pytest.mark.usefixtures('task_db')
async def test_recorder_failure_does_not_propagate() -> None:
    """
    A logging failure never interrupts the work it was describing.
    """
    task = await make_task()
    recorder = TaskRecorder(task)
    await task.delete()  # The foreign key can no longer be satisfied.

    await recorder.handle_event(ReviewEvent(approved=True, comments='fine'))


@pytest.mark.usefixtures('task_db')
async def test_status_change_updates_the_task_and_logs_it() -> None:
    """
    Recording a status change writes both the task row and a log entry.
    """
    task = await make_task()
    recorder = TaskRecorder(task)

    await recorder.record_status(TaskStatus.completed, 'Finished the work.')

    refreshed = await Task.get(id=task.id)
    assert refreshed.status is TaskStatus.completed
    assert refreshed.summary == 'Finished the work.'

    entries = await TaskMemory.filter(task=task, entry_type=TaskLogEntryType.status_change)
    assert len(entries) == 1


@pytest.mark.usefixtures('task_db')
async def test_log_is_compressed_once_it_grows() -> None:
    """
    An overlong log collapses its oldest entries into a summary.

    A long session writes an entry per tool call, so the log has to stay bounded
    without losing the shape of what happened.
    """
    task = await make_task()
    recorder = TaskRecorder(task, max_entries=20, compress_batch=10)

    for index in range(25):
        await recorder.record(
            TaskLogEntryType.tool_call,
            {'tool': 'edit_file', 'files': [f'file_{index}.py']},
        )

    entries = await TaskMemory.filter(task=task).order_by('id')
    assert len(entries) < 25

    summaries = [e for e in entries if e.entry_type is TaskLogEntryType.summary]
    assert len(summaries) == 1
    assert summaries[0].content['collapsed_entries'] == 10
    assert summaries[0].content['counts']['tool_call'] == 10
    assert 'file_0.py' in summaries[0].content['files_touched']


@pytest.mark.usefixtures('task_db')
async def test_recent_entries_survive_compression() -> None:
    """
    Compression removes the oldest entries, never the newest.
    """
    task = await make_task()
    recorder = TaskRecorder(task, max_entries=15, compress_batch=10)

    for index in range(20):
        await recorder.record(TaskLogEntryType.tool_call, {'tool': f'tool_{index}'})

    entries = await TaskMemory.filter(task=task).order_by('id')
    tools = [e.content.get('tool') for e in entries if e.entry_type is TaskLogEntryType.tool_call]
    assert 'tool_19' in tools
    assert 'tool_0' not in tools


def test_compression_settings_are_validated() -> None:
    """
    A compression batch that is not smaller than the limit is rejected.
    """
    task = Task(id=uuid4(), prompt='x')
    with pytest.raises(ValueError):
        TaskRecorder(task, max_entries=10, compress_batch=10)


@pytest.mark.usefixtures('task_db')
async def test_crashed_session_marks_the_task_failed() -> None:
    """
    A session that raises leaves its task recorded as failed.

    Otherwise the task would stay at `initialized` and the history would show it
    as never having started.
    """
    from api.v1.agent_session import AgentSession

    task = await make_task()
    recorder = TaskRecorder(task)

    session = AgentSession.__new__(AgentSession)
    session._recorder = recorder

    await session.mark_failed()

    refreshed = await Task.get(id=task.id)
    assert refreshed.status is TaskStatus.failed


@pytest.mark.usefixtures('task_db')
async def test_marking_failed_without_a_recorder_is_safe() -> None:
    """
    A session that never registered a task can still be marked failed.
    """
    from api.v1.agent_session import AgentSession

    session = AgentSession.__new__(AgentSession)
    session._recorder = None

    await session.mark_failed()


@pytest.mark.usefixtures('task_db')
async def test_interrupted_tasks_are_failed_on_startup() -> None:
    """
    A task left running by a dead process is marked failed at boot.

    Sessions live only in memory, so a running task at startup belongs to a
    process that no longer exists and would otherwise show as in progress forever.
    """
    from services.task_recorder import fail_interrupted_tasks

    running = await Task.create(id=uuid4(), prompt='was running', status=TaskStatus.running)
    finished = await Task.create(id=uuid4(), prompt='was done', status=TaskStatus.merged)
    fresh = await Task.create(id=uuid4(), prompt='never started', status=TaskStatus.initialized)

    assert await fail_interrupted_tasks() == 1

    assert (await Task.get(id=running.id)).status is TaskStatus.failed
    assert (await Task.get(id=finished.id)).status is TaskStatus.merged
    assert (await Task.get(id=fresh.id)).status is TaskStatus.initialized


@pytest.mark.usefixtures('task_db')
async def test_the_interruption_is_recorded_in_the_log() -> None:
    """
    The reason a task was failed is written to its log, not just its status.
    """
    from services.task_recorder import fail_interrupted_tasks

    task = await Task.create(id=uuid4(), prompt='was running', status=TaskStatus.running)

    await fail_interrupted_tasks()

    entries = await TaskMemory.filter(task=task, entry_type=TaskLogEntryType.status_change)
    assert len(entries) == 1
    assert 'restart' in entries[0].content['reason']


@pytest.mark.usefixtures('task_db')
async def test_nothing_to_sweep_is_harmless() -> None:
    """
    A clean startup sweeps nothing.
    """
    from services.task_recorder import fail_interrupted_tasks

    assert await fail_interrupted_tasks() == 0


@pytest.mark.usefixtures('task_db')
async def test_create_task_persists_the_context_windows(controller: TaskController) -> None:
    """
    The two budgets are stored independently of one another.
    """
    response = await controller.create_task(
        TaskRequest(
            prompt='Add a timeout',
            repo_path='/repo',
            coder_model='ollama:llama3',
            coder_context_window=40_000,
            planner_context_window=8_192,
        )
    )

    stored = await Task.get_or_none(id=response.id)
    assert stored is not None
    assert stored.coder_context_window == 40_000
    assert stored.planner_context_window == 8_192


@pytest.mark.usefixtures('task_db')
async def test_an_unset_context_window_is_stored_as_zero(controller: TaskController) -> None:
    """
    Zero records "the user configured nothing", not "the user chose zero".

    The raw setting is stored rather than the resolved number so that changing
    the default later reads as unset on old tasks, instead of silently claiming
    they used a value they never saw.
    """
    response = await controller.create_task(TaskRequest(prompt='Add a timeout'))

    stored = await Task.get_or_none(id=response.id)
    assert stored is not None
    assert stored.coder_context_window == 0
    assert stored.planner_context_window == 0


@pytest.mark.usefixtures('task_db')
async def test_get_task_reports_the_context_windows(controller: TaskController) -> None:
    """
    A recorded budget has to be readable back, or the record is of no use.
    """
    task = await make_task()
    task.coder_context_window = 40_000
    task.planner_context_window = 8_192
    await task.save()

    detail = await controller.get_task(str(task.id))

    assert detail.coder_context_window == 40_000
    assert detail.planner_context_window == 8_192
