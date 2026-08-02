from __future__ import annotations

import socket
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from docker.errors import NotFound

from services.event_bus import SandboxStage
from services.docker_sandbox import (
    DEFAULT_SANDBOX_IMAGE,
    TIMEOUT_EXIT_CODE,
    CommandResult,
    DockerSandboxManager,
    SandboxError,
    host_path_within,
    mount_covering,
)


def attach_fake_container(
    manager: DockerSandboxManager,
    exit_code: int = 0,
    output: bytes = b'ok',
    probe_ok: bool = True,
    has_timeout: bool = True,
) -> tuple[list[Any], Any]:
    """
    Attach a fake Docker client that records the commands it is asked to run.

    :param manager: The sandbox manager to attach the client to.
    :param exit_code: The exit status ordinary commands should report.
    :param output: The output ordinary commands should return.
    :param probe_ok: Whether the readiness probe should succeed.
    :param has_timeout: Whether the image should appear to provide `timeout`.
    :returns: The captured command list and the fake Docker client.
    """
    captured: list[Any] = []

    def exec_run(command: Any, **kwargs: Any) -> Any:
        captured.append(command)
        script: str = command[2] if isinstance(command, list) and len(command) > 2 else ''
        result: Any = MagicMock()

        if script == 'echo ok':
            result.exit_code = 0 if probe_ok else 1
            result.output = b'ok\n' if probe_ok else b''
            return result
        if script == 'command -v timeout':
            result.exit_code = 0 if has_timeout else 1
            result.output = b'/usr/bin/timeout\n' if has_timeout else b''
            return result

        result.exit_code = exit_code
        result.output = output
        return result

    container: Any = MagicMock()
    container.exec_run = exec_run
    client: Any = MagicMock()

    def get(name: str) -> Any:
        """
        Look up a container by name.

        Asking for this process's own hostname raises `NotFound`, which is how
        the manager detects that it is not itself containerised. Tests that want
        the containerised path use `attach_own_container`.
        """
        if name == socket.gethostname():
            raise NotFound('no such container')
        return container

    client.containers.get.side_effect = get
    client.containers.run.return_value = container
    manager._client = client
    return captured, client


def attach_own_container(
    manager: DockerSandboxManager, client: Any, mounts: list[dict[str, str]]
) -> None:
    """
    Make the manager appear to be running inside a container with given mounts.

    :param manager: The sandbox manager being tested.
    :param client: The fake Docker client already attached to it.
    :param mounts: Mount records as the container runtime would report them.
    """
    own: Any = MagicMock()
    own.attrs = {'Mounts': mounts}
    sandbox_container: Any = client.containers.run.return_value

    def get(name: str) -> Any:
        if name == socket.gethostname():
            return own
        return sandbox_container

    client.containers.get.side_effect = get


async def test_container_name_is_per_session(sandbox: DockerSandboxManager) -> None:
    """
    Each session gets its own container.

    A shared name means one session's teardown destroys another's sandbox.
    """
    attach_fake_container(sandbox)
    await sandbox.spawn_sandbox('abc123')
    assert sandbox.container_name == 'loom-sandbox-abc123'


async def test_default_image_is_used_without_a_dockerfile(
    sandbox: DockerSandboxManager
) -> None:
    """
    No Dockerfile means the default image, and no build.
    """
    _, client = attach_fake_container(sandbox)
    await sandbox.spawn_sandbox('abc123')

    assert client.containers.run.call_args.args[0] == DEFAULT_SANDBOX_IMAGE
    client.images.build.assert_not_called()


async def test_custom_dockerfile_is_built_and_cleaned_up(
    sandbox: DockerSandboxManager
) -> None:
    """
    A supplied Dockerfile is built into a tagged image from a temporary context.
    """
    import pathlib

    _, client = attach_fake_container(sandbox)
    contexts: list[str] = []

    def record_build(path: str, tag: str, **kwargs: Any) -> Any:
        contexts.append(path)
        assert (pathlib.Path(path) / 'Dockerfile').read_text() == 'FROM alpine\n'
        return (MagicMock(), [])

    client.images.build = record_build

    await sandbox.spawn_sandbox('abc123', dockerfile='FROM alpine\n')

    assert client.containers.run.call_args.args[0] == 'loom-sandbox-abc123'
    assert len(contexts) == 1
    assert not pathlib.Path(contexts[0]).exists()


async def test_build_failure_is_reported(sandbox: DockerSandboxManager) -> None:
    """
    A Dockerfile that will not build produces a clear error.
    """
    _, client = attach_fake_container(sandbox)

    def failing_build(**kwargs: Any) -> Any:
        raise RuntimeError('unknown instruction FROMM')

    client.images.build = failing_build

    with pytest.raises(SandboxError, match='could not be built'):
        await sandbox.spawn_sandbox('abc123', dockerfile='FROMM alpine\n')


async def test_unresponsive_container_is_rejected(sandbox: DockerSandboxManager) -> None:
    """
    A container that starts but cannot run commands is reported, not ignored.

    Continuing silently would leave the agent unable to verify anything, with no
    explanation of why.
    """
    attach_fake_container(sandbox, probe_ok=False)

    with pytest.raises(SandboxError, match='did not respond'):
        await sandbox.spawn_sandbox('abc123')


async def test_secrets_are_injected_as_environment(sandbox: DockerSandboxManager) -> None:
    """
    Secrets reach the container as environment variables.
    """
    _, client = attach_fake_container(sandbox)
    await sandbox.spawn_sandbox('abc123', environment={'API_KEY': 's3cret'})

    assert client.containers.run.call_args.kwargs['environment'] == {'API_KEY': 's3cret'}
    assert sandbox.secret_names == ['API_KEY']


async def test_secret_values_are_stripped_from_output(
    sandbox: DockerSandboxManager
) -> None:
    """
    A command that prints a secret does not leak it to the agent or the client.

    Command output is echoed into the chat feed, so an unredacted `env` dump
    would publish every secret the user configured.
    """
    attach_fake_container(sandbox, output=b'API_KEY=s3cret\nTOKEN=abc123\n')
    await sandbox.spawn_sandbox('abc123', environment={'API_KEY': 's3cret', 'TOKEN': 'abc123'})

    result: CommandResult = await sandbox.run_test_command('env')

    assert 's3cret' not in result.output
    assert 'abc123' not in result.output
    assert result.output.count('***') == 2
    assert 'API_KEY' in result.output


async def test_overlapping_secrets_are_fully_redacted(
    sandbox: DockerSandboxManager
) -> None:
    """
    A secret that contains another is redacted whole.

    Replacing the shorter value first would leave the remainder of the longer
    one visible.
    """
    attach_fake_container(sandbox, output=b'value=supersecret123\n')
    await sandbox.spawn_sandbox('abc', environment={'A': 'secret', 'B': 'supersecret123'})

    result: CommandResult = await sandbox.run_test_command('echo $B')
    assert 'supersecret123' not in result.output
    assert 'secret123' not in result.output


async def test_commands_are_wrapped_with_a_timeout(sandbox: DockerSandboxManager) -> None:
    """
    Commands run under an in-container timeout so a hang is actually killed.
    """
    captured, _ = attach_fake_container(sandbox)
    await sandbox.spawn_sandbox('abc123')
    captured.clear()

    await sandbox.run_test_command('pytest')

    script: str = captured[0][2]
    assert script.startswith('timeout 60 ')
    assert 'pytest' in script


async def test_timeout_exit_code_is_reported(sandbox: DockerSandboxManager) -> None:
    """
    A command killed by the timeout is reported as timed out.
    """
    attach_fake_container(sandbox, exit_code=TIMEOUT_EXIT_CODE, output=b'')
    await sandbox.spawn_sandbox('abc123')

    result: CommandResult = await sandbox.run_test_command('sleep 999')
    assert result.timed_out is True
    assert result.success is False


async def test_images_without_timeout_still_run(sandbox: DockerSandboxManager) -> None:
    """
    An image lacking the timeout utility runs commands unwrapped.

    Wrapping unconditionally would make every command fail with 127 on a minimal
    image.
    """
    captured, _ = attach_fake_container(sandbox, has_timeout=False)
    await sandbox.spawn_sandbox('abc123')
    captured.clear()

    await sandbox.run_test_command('pytest')
    assert captured[0][2] == 'pytest'


async def test_shell_metacharacters_survive_wrapping(
    sandbox: DockerSandboxManager
) -> None:
    """
    Pipes and operators still work once the command is wrapped for the timeout.
    """
    captured, _ = attach_fake_container(sandbox)
    await sandbox.spawn_sandbox('abc123')
    captured.clear()

    command: str = "pytest -q | tail -5 && echo 'done' > out.txt"
    await sandbox.run_test_command(command)

    assert command in captured[0][2].replace("'\\''", "'")


async def test_successful_command_reports_success(sandbox: DockerSandboxManager) -> None:
    """
    A zero exit status is reported as success.
    """
    attach_fake_container(sandbox, output=b'2 passed')
    await sandbox.spawn_sandbox('abc123')

    result: CommandResult = await sandbox.run_test_command('pytest')
    assert result.success is True
    assert '2 passed' in result.output


async def test_failing_command_reports_failure(sandbox: DockerSandboxManager) -> None:
    """
    A non-zero exit status is surfaced rather than discarded.
    """
    attach_fake_container(sandbox, exit_code=1, output=b'2 failed')
    await sandbox.spawn_sandbox('abc123')

    result: CommandResult = await sandbox.run_test_command('pytest')
    assert result.exit_code == 1
    assert result.success is False


async def test_teardown_removes_container_and_built_image(
    sandbox: DockerSandboxManager
) -> None:
    """
    Stopping the sandbox removes both the container and any image built for it.

    A built image per session would otherwise accumulate on the host.
    """
    _, client = attach_fake_container(sandbox)
    client.images.build = lambda **kwargs: (MagicMock(), [])

    await sandbox.spawn_sandbox('abc123', dockerfile='FROM alpine\n')
    await sandbox.stop_sandbox()

    client.images.remove.assert_called_once()
    assert sandbox._built_image is None


async def test_teardown_closes_the_docker_client(sandbox: DockerSandboxManager) -> None:
    """
    The Docker client is closed with the sandbox.

    One client per session would otherwise accumulate connections to the Docker
    socket for the life of the process.
    """
    _, client = attach_fake_container(sandbox)
    await sandbox.spawn_sandbox('abc123')

    await sandbox.stop_sandbox()

    client.close.assert_called_once()
    assert sandbox._client is None


def test_a_path_outside_any_mount_is_refused() -> None:
    """
    A repository the host cannot see is reported rather than mounted wrongly.

    Requirement 7.5 and 7.6: the message names the container path and what is
    actually mounted, so the reader can see why discovery failed.
    """
    manager: DockerSandboxManager = DockerSandboxManager(repo_path='/home/user/elsewhere')
    _, client = attach_fake_container(manager)
    attach_own_container(
        manager, client, [{'Source': '/srv/loom/data/repo', 'Destination': '/tmp/repo'}]
    )

    with pytest.raises(SandboxError) as raised:
        manager.host_mount_source()

    assert raised.value.stage is SandboxStage.path_discovery
    assert '/home/user/elsewhere' in str(raised.value)
    assert '/tmp/repo' in str(raised.value)


def test_a_relative_mount_source_is_refused() -> None:
    """
    A relative source would become a named volume, leaving the sandbox empty.
    """
    manager: DockerSandboxManager = DockerSandboxManager(repo_path='/tmp/repo')
    _, client = attach_fake_container(manager)
    attach_own_container(manager, client, [{'Source': 'data/repo', 'Destination': '/tmp/repo'}])

    with pytest.raises(SandboxError) as raised:
        manager.host_mount_source()

    assert raised.value.stage is SandboxStage.path_discovery
    assert 'not' in str(raised.value) and 'absolute' in str(raised.value)


def test_the_host_path_is_discovered_from_this_container() -> None:
    """
    The host path comes from inspecting our own mounts, not from configuration.

    Requirement 7.2.
    """
    manager: DockerSandboxManager = DockerSandboxManager(repo_path='/tmp/repo')
    _, client = attach_fake_container(manager)
    attach_own_container(
        manager, client, [{'Source': '/srv/loom/data/repo', 'Destination': '/tmp/repo'}]
    )

    assert manager.host_mount_source() == '/srv/loom/data/repo'


def test_a_subdirectory_is_translated() -> None:
    """
    A repository beneath the mount maps onto the matching host path.
    """
    manager: DockerSandboxManager = DockerSandboxManager(repo_path='/tmp/repo/project-a')
    _, client = attach_fake_container(manager)
    attach_own_container(
        manager, client, [{'Source': '/srv/loom/data/repo', 'Destination': '/tmp/repo'}]
    )

    assert manager.host_mount_source() == '/srv/loom/data/repo/project-a'


def test_outside_a_container_no_translation_happens() -> None:
    """
    Running outside a container needs no translation, and detects that itself.

    Requirement 7.4: detected, not configured. The runtime reports no container
    matching our hostname, and that answer is the detection.
    """
    manager: DockerSandboxManager = DockerSandboxManager(repo_path='/srv/project')
    attach_fake_container(manager)

    assert manager.host_mount_source() == '/srv/project'


def test_an_unreachable_runtime_is_named_as_such() -> None:
    """
    Failing to reach the runtime is a different failure from failing to translate.

    Requirement 7.7.
    """
    manager: DockerSandboxManager = DockerSandboxManager(repo_path='/srv/project')
    client: Any = MagicMock()
    client.containers.get.side_effect = RuntimeError('daemon is not listening')
    manager._client = client

    with pytest.raises(SandboxError) as raised:
        manager.host_mount_source()

    assert raised.value.stage is SandboxStage.runtime_connectivity


async def test_spawn_mounts_the_host_path(sandbox: DockerSandboxManager) -> None:
    """
    The resolved host path is what actually reaches the container runtime.
    """
    _, client = attach_fake_container(sandbox)
    attach_own_container(
        sandbox, client, [{'Source': '/srv/loom/data/repo', 'Destination': sandbox.repo_path}]
    )

    await sandbox.spawn_sandbox('abc123')

    volumes = client.containers.run.call_args.kwargs['volumes']
    assert list(volumes) == ['/srv/loom/data/repo']
    assert volumes['/srv/loom/data/repo']['bind'] == '/workspace'


def test_mount_covering_picks_the_most_specific_mount() -> None:
    """
    Nested mounts resolve to the innermost one that covers the path.

    Translating through the outer mount would produce a host path pointing at a
    different directory that happens to have the same name.
    """
    mounts: list[dict[str, str]] = [
        {'Source': '/host/data', 'Destination': '/data'},
        {'Source': '/host/elsewhere/repo', 'Destination': '/data/repo'},
    ]

    assert mount_covering(mounts, '/data/repo/src') == ('/host/elsewhere/repo', '/data/repo')


def test_mount_covering_matches_the_mount_point_itself() -> None:
    """
    The mount point counts as covered by its own mount.
    """
    mounts: list[dict[str, str]] = [{'Source': '/host/repo', 'Destination': '/data/repo'}]

    assert mount_covering(mounts, '/data/repo') == ('/host/repo', '/data/repo')


def test_mount_covering_does_not_match_a_sibling_by_prefix() -> None:
    """
    `/data/repository` is not inside `/data/repo`, despite sharing a prefix.
    """
    mounts: list[dict[str, str]] = [{'Source': '/host/repo', 'Destination': '/data/repo'}]

    assert mount_covering(mounts, '/data/repository') is None


def test_mount_covering_tolerates_a_trailing_slash() -> None:
    """
    Runtimes differ on whether a destination carries a trailing slash.
    """
    mounts: list[dict[str, str]] = [{'Source': '/host/repo', 'Destination': '/data/repo/'}]

    assert mount_covering(mounts, '/data/repo/src') == ('/host/repo', '/data/repo/')


def test_mount_covering_ignores_incomplete_records() -> None:
    """
    A mount with no source cannot be translated through.
    """
    mounts: list[dict[str, str]] = [
        {'Destination': '/data/repo'},
        {'Source': '', 'Destination': '/data'},
    ]

    assert mount_covering(mounts, '/data/repo/src') is None


def test_mount_covering_returns_nothing_for_an_empty_list() -> None:
    """
    A container with no mounts covers nothing.
    """
    assert mount_covering([], '/data/repo') is None


def test_host_path_within_returns_the_source_for_the_mount_point() -> None:
    """
    The mount point maps to the source itself, with nothing appended.
    """
    assert host_path_within('/host/repo', '/data/repo', '/data/repo') == '/host/repo'


def test_host_path_within_appends_the_relative_remainder() -> None:
    """
    A path beneath the mount keeps its position relative to the source.
    """
    assert (
        host_path_within('/host/repo', '/data/repo', '/data/repo/src/app.py')
        == '/host/repo/src/app.py'
    )


async def test_an_undownloadable_image_is_named_as_a_pull_failure(
    sandbox: DockerSandboxManager,
) -> None:
    """
    A missing image that cannot be fetched reads as a download failure.

    Requirement 7.7. Letting `run` pull implicitly would surface this as a
    container that would not start, sending the reader to the wrong problem.
    """
    _, client = attach_fake_container(sandbox)
    client.images.get.side_effect = NotFound('absent')
    client.images.pull.side_effect = RuntimeError('registry unreachable')

    with pytest.raises(SandboxError) as raised:
        await sandbox.spawn_sandbox('abc123')

    assert raised.value.stage is SandboxStage.image_pull
    assert 'registry unreachable' in str(raised.value)


async def test_a_container_that_will_not_start_is_named_as_such(
    sandbox: DockerSandboxManager,
) -> None:
    """
    A failure to start is distinct from a failure to download.
    """
    _, client = attach_fake_container(sandbox)
    client.containers.run.side_effect = RuntimeError('no such device')

    with pytest.raises(SandboxError) as raised:
        await sandbox.spawn_sandbox('abc123')

    assert raised.value.stage is SandboxStage.container_start


async def test_a_path_failure_is_not_relabelled_as_a_start_failure(
    sandbox: DockerSandboxManager,
) -> None:
    """
    Path discovery keeps its own stage through the spawn path.

    The mount source is resolved before the container is attempted precisely so
    that its staged error is not swallowed by the start failure handler.
    """
    _, client = attach_fake_container(sandbox)
    attach_own_container(
        sandbox, client, [{'Source': '/srv/elsewhere', 'Destination': '/not/the/repo'}]
    )

    with pytest.raises(SandboxError) as raised:
        await sandbox.spawn_sandbox('abc123')

    assert raised.value.stage is SandboxStage.path_discovery


async def test_a_present_image_is_not_pulled(sandbox: DockerSandboxManager) -> None:
    """
    An image already present locally is used without contacting a registry.
    """
    _, client = attach_fake_container(sandbox)

    await sandbox.spawn_sandbox('abc123')

    client.images.pull.assert_not_called()


def test_every_stage_can_actually_occur() -> None:
    """
    No stage is defined that nothing produces.

    A stage with no producer is worse than a missing one: it suggests the cause
    is distinguishable when in practice it is reported as something else.
    """
    source: str = Path(__file__).resolve().parent.parent.joinpath(
        'services', 'docker_sandbox.py'
    ).read_text()

    for stage in SandboxStage:
        assert f'SandboxStage.{stage.name}' in source, f'nothing raises {stage.name}'
