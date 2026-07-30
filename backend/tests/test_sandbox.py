from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from services.docker_sandbox import (
    DEFAULT_SANDBOX_IMAGE,
    TIMEOUT_EXIT_CODE,
    CommandResult,
    DockerSandboxManager,
    SandboxError,
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
    client.containers.get.return_value = container
    client.containers.run.return_value = container
    manager._client = client
    return captured, client


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


def test_host_path_is_used_when_containerised() -> None:
    """
    The sandbox is mounted from the host's path, not this process's view of it.

    Bind mount sources are resolved by the daemon on the host, so mounting
    `/tmp/repo` from inside a container mounts the host's `/tmp/repo` rather
    than the project, and the sandbox operates on the wrong tree.
    """
    manager: DockerSandboxManager = DockerSandboxManager(
        repo_path='/tmp/repo', host_repo_root='/srv/loom/data/repo'
    )
    assert manager.host_mount_source() == '/srv/loom/data/repo'


def test_subdirectories_are_translated() -> None:
    """
    A repository beneath the mount root maps onto the matching host path.
    """
    manager: DockerSandboxManager = DockerSandboxManager(
        repo_path='/tmp/repo', host_repo_root='/srv/loom/data/repo'
    )
    manager.repo_path = '/tmp/repo/project-a'

    assert manager.host_mount_source() == '/srv/loom/data/repo/project-a'


def test_path_outside_the_mount_is_refused() -> None:
    """
    A repository the container cannot see is reported rather than mounted wrongly.
    """
    manager: DockerSandboxManager = DockerSandboxManager(
        repo_path='/tmp/repo', host_repo_root='/srv/loom/data/repo'
    )
    manager.repo_path = '/home/user/elsewhere'

    with pytest.raises(SandboxError, match='only directory mounted'):
        manager.host_mount_source()


def test_relative_host_path_is_refused() -> None:
    """
    A relative source would become a named volume, leaving the sandbox empty.
    """
    manager: DockerSandboxManager = DockerSandboxManager(
        repo_path='/tmp/repo', host_repo_root='data/repo'
    )

    with pytest.raises(SandboxError, match='absolute path'):
        manager.host_mount_source()


def test_without_a_host_root_the_local_path_is_used() -> None:
    """
    Running outside a container needs no translation.
    """
    manager: DockerSandboxManager = DockerSandboxManager(repo_path='/srv/project')
    assert manager.host_mount_source() == '/srv/project'


async def test_spawn_mounts_the_host_path(sandbox: DockerSandboxManager) -> None:
    """
    The resolved host path is what actually reaches the container runtime.
    """
    _, client = attach_fake_container(sandbox)
    sandbox._host_repo_root = '/srv/loom/data/repo'
    sandbox._container_repo_root = sandbox.repo_path

    await sandbox.spawn_sandbox('abc123')

    volumes = client.containers.run.call_args.kwargs['volumes']
    assert list(volumes) == ['/srv/loom/data/repo']
    assert volumes['/srv/loom/data/repo']['bind'] == '/workspace'
