from __future__ import annotations

import asyncio
import os
import socket
import shutil
import tempfile
from collections.abc import Iterable, Mapping
from logging import Logger, getLogger
from typing import Annotated, Any

import docker
from docker.errors import NotFound
from pydantic import BaseModel, Field

from services.events import SandboxStage

logger: Logger = getLogger(__name__)

DEFAULT_SANDBOX_IMAGE: str = 'python:3.14-slim'
DEFAULT_COMMAND_TIMEOUT: float = 60.0
TIMEOUT_EXIT_CODE: int = 124
REDACTION: str = '***'
GRACE_SECONDS: float = 5.0


def mount_covering(
    mounts: Iterable[Mapping[str, Any]], container_path: str
) -> tuple[str, str] | None:
    """
    Find the bind mount that a path inside this container falls under.

    Where several mounts could apply, the most specific wins: a repository at
    `/data/repo` under mounts of both `/data` and `/data/repo` belongs to the
    latter, and using the former would translate to the wrong host path.

    :param mounts: The mount records as the container runtime reports them.
    :param container_path: An absolute path inside this container.
    :returns: The matching source and destination, or None if nothing covers it.
    """
    best: tuple[str, str] | None = None
    for mount in mounts:
        destination: str = str(mount.get('Destination') or '')
        source: str = str(mount.get('Source') or '')
        if not destination or not source:
            continue
        covers: bool = container_path == destination or container_path.startswith(
            destination.rstrip('/') + '/'
        )
        if covers and (best is None or len(destination) > len(best[1])):
            best = (source, destination)
    return best


def host_path_within(source: str, destination: str, container_path: str) -> str:
    """
    Translate a path inside this container to the host path for the same file.

    :param source: The mount's source, as the host sees it.
    :param destination: Where that source is mounted inside this container.
    :param container_path: An absolute path inside this container, under the mount.
    :returns: The host path for the same location.
    """
    if container_path == destination:
        return source
    return os.path.join(source, os.path.relpath(container_path, destination))


class SandboxError(RuntimeError):
    """
    Raised when the sandbox cannot be prepared or is not usable.

    Carries the step that failed, so that callers can tell the causes apart
    rather than presenting every one of them as the same sentence.
    """

    stage: SandboxStage

    def __init__(self, stage: SandboxStage, message: str) -> None:
        """
        Describe a sandbox failure.

        :param stage: Which step of preparing the sandbox failed.
        :param message: What went wrong, in terms the reader can act on.
        """
        super().__init__(message)
        self.stage = stage


class CommandResult(BaseModel):
    """
    Outcome of a command executed inside the sandbox.
    """

    exit_code: Annotated[int, Field(description='Process exit status. Zero means the command succeeded.')]
    output: Annotated[str, Field(description='Combined stdout and stderr, with any secret values redacted.')]
    timed_out: Annotated[bool, Field(description='Whether the command was killed for exceeding its time limit.')] = False

    @property
    def success(self) -> bool:
        """
        Whether the command completed successfully.

        :returns: True if the command exited with status zero.
        """
        return self.exit_code == 0


class DockerSandboxManager:
    """
    Manager for Docker sandbox environments.

    One container is built and started per session and reused for every command
    in that session; it is never rebuilt per tool call.
    """

    _client: Any | None
    _repo_path: str
    _default_image: str
    _command_timeout: float
    _container_name: str
    _built_image: str | None
    _secrets: dict[str, str]
    _supports_shell_timeout: bool

    def __init__(
        self,
        repo_path: str,
        default_image: str = DEFAULT_SANDBOX_IMAGE,
        command_timeout: float = DEFAULT_COMMAND_TIMEOUT,
    ) -> None:
        """
        Create a sandbox manager.

        :param repo_path: Path to the repository as this process sees it.
        :param default_image: Image used when the user supplies no Dockerfile.
        :param command_timeout: How long a single command may run, in seconds.
        """
        self._client = None
        self._repo_path = repo_path
        self._default_image = default_image
        self._command_timeout = command_timeout
        self._container_name = 'loom-sandbox'
        self._built_image = None
        self._secrets = {}
        self._supports_shell_timeout = False

    @property
    def repo_path(self) -> str:
        """
        The host path bind mounted into the sandbox.

        :returns: The repository path.
        """
        return self._repo_path

    @repo_path.setter
    def repo_path(self, value: str) -> None:
        """
        Change the repository that will be mounted.

        :param value: The new repository path.
        """
        self._repo_path = value

    @property
    def container_name(self) -> str:
        """
        The name of this session's container.

        :returns: The container name.
        """
        return self._container_name

    @property
    def secret_names(self) -> list[str]:
        """
        The names of the secrets available inside the sandbox.

        Names are safe to show the agent so it can reference them; the values
        never leave this object.

        :returns: The secret names, sorted.
        """
        return sorted(self._secrets)

    @property
    def client(self) -> Any:
        """
        Lazy-loaded Docker client.

        :returns: The Docker client.
        :raises RuntimeError: If the Docker socket cannot be reached.
        """
        if self._client is None:
            try:
                self._client = docker.from_env()
                self._client.ping()
            except Exception as e:
                if 'Permission denied' in str(e) or getattr(e, 'errno', None) == 13:
                    raise RuntimeError(
                        'Permission denied when connecting to Docker socket. '
                        'Please ensure the user running the application has read/write access to /var/run/docker.sock. '
                        "If using Docker, you might need 'sudo usermod -aG docker $USER'. "
                        'If using Podman, ensure the podman.socket is running and the path is correct.'
                    ) from e
                raise
        return self._client

    def _own_container(self) -> Any | None:
        """
        Find this process's own container, if it is running inside one.

        The runtime sets a container's hostname to its identifier, so asking the
        runtime about our own hostname either returns us or tells us there is no
        such container. That answer is the detection: nothing has to be
        configured to say whether Loom is containerised.

        :returns: This process's container, or None if it is not in one.
        :raises SandboxError: If the container runtime cannot be reached.
        """
        try:
            client: Any = self.client
        except Exception as error:
            raise SandboxError(
                SandboxStage.runtime_connectivity,
                f'The container runtime could not be reached: {error}',
            ) from error

        try:
            return client.containers.get(socket.gethostname())
        except NotFound:
            # Not containerised: this process and the runtime share a filesystem.
            return None
        except Exception as error:
            raise SandboxError(
                SandboxStage.runtime_connectivity,
                f'The container runtime could not be asked about this container: {error}',
            ) from error

    def host_mount_source(self) -> str:
        """
        Resolve the path to bind mount into the sandbox, as the host sees it.

        Bind mount sources are resolved by the container runtime's daemon, which
        runs on the host. When this process is itself in a container, its own
        view of the repository is meaningless to that daemon: asking for
        `/tmp/repo` would mount the host's `/tmp/repo`, not the project. The host
        path is therefore discovered by inspecting this container's own mounts,
        rather than being supplied as configuration nobody can verify.

        :returns: An absolute host path to mount.
        :raises SandboxError: If the host path cannot be discovered.
        """
        container_path: str = os.path.abspath(self._repo_path)

        own: Any | None = self._own_container()
        if own is None:
            # Not containerised, so the path is already a host path.
            return container_path

        mounts: list[Mapping[str, Any]] = list(own.attrs.get('Mounts') or [])
        covering: tuple[str, str] | None = mount_covering(mounts, container_path)
        if covering is None:
            visible: str = ', '.join(
                sorted(str(mount.get('Destination') or '') for mount in mounts if mount.get('Destination'))
            )
            raise SandboxError(
                SandboxStage.path_discovery,
                f'The repository at {container_path} inside this container is not under any '
                'directory mounted from the host, so the sandbox cannot be given the same files. '
                f'Mounted here: {visible or "nothing"}. Mount the repository into this container '
                'and choose a path beneath it.',
            )

        source, destination = covering
        resolved: str = host_path_within(source, destination, container_path)
        if not os.path.isabs(resolved):
            # A relative source is silently treated as a named volume rather than
            # a bind mount, so the sandbox would start with an empty tree.
            raise SandboxError(
                SandboxStage.path_discovery,
                f'The host path discovered for {container_path} is {resolved!r}, which is not '
                f'absolute. It came from the mount of {source!r} at {destination!r}. A relative '
                'source is treated as a named volume and the sandbox would see nothing.',
            )
        return resolved

    def redact(self, text: str) -> str:
        """
        Replace any secret value appearing in text with a placeholder.

        Command output is echoed back to the agent and to the browser, so a
        script that prints its environment would otherwise leak every secret.

        :param text: The text to sanitise.
        :returns: The text with secret values replaced.
        """
        sanitised: str = text
        # Longest first, so a secret containing another is not partly revealed.
        for value in sorted(self._secrets.values(), key=len, reverse=True):
            if value:
                sanitised = sanitised.replace(value, REDACTION)
        return sanitised

    def _build_image(self, session_id: str, dockerfile: str) -> str:
        """
        Build a sandbox image from user-supplied Dockerfile content.

        :param session_id: Identifier used to tag the image.
        :param dockerfile: The Dockerfile content to build.
        :returns: The tag of the built image.
        :raises SandboxError: If the image cannot be built.
        """
        tag: str = f'loom-sandbox-{session_id}'
        context: str = tempfile.mkdtemp(prefix='loom-build-')
        try:
            with open(os.path.join(context, 'Dockerfile'), 'w') as handle:
                handle.write(dockerfile)
            self.client.images.build(path=context, tag=tag, rm=True)
        except Exception as error:
            raise SandboxError(
                SandboxStage.image_build,
                f'The sandbox image could not be built from the supplied Dockerfile: {error}',
            ) from error
        finally:
            shutil.rmtree(context, ignore_errors=True)

        self._built_image = tag
        return tag

    async def check_reachable(self) -> str | None:
        """
        Ask whether the container runtime can be reached at all.

        Cheap enough to run before indexing: it opens the client and pings, doing
        no image build and starting no container. Knowing early means the
        interface can say straight away that nothing will be verifiable, rather
        than letting the user wait through indexing to find out.

        :returns: None if the runtime is reachable, otherwise why it is not.
        """

        def _probe() -> str | None:
            try:
                # Constructing the client pings the daemon.
                _ = self.client
            except Exception as error:
                return str(error)
            return None

        return await asyncio.to_thread(_probe)

    async def spawn_sandbox(
        self,
        session_id: str,
        dockerfile: str = '',
        environment: Mapping[str, str] | None = None,
    ) -> Any:
        """
        Build if necessary, then start this session's sandbox container.

        :param session_id: Identifier used to name the container and any built image.
        :param dockerfile: Optional Dockerfile content to build the image from.
        :param environment: Secrets to inject as environment variables.
        :returns: The created container instance.
        :raises SandboxError: If the image cannot be built or the container is not usable.
        """
        self._container_name = f'loom-sandbox-{session_id}'
        self._secrets = dict(environment or {})

        def _spawn() -> Any:
            image: str = (
                self._build_image(session_id, dockerfile)
                if dockerfile.strip()
                else self._default_image
            )

            # Pulled explicitly rather than letting `run` do it implicitly, so a
            # download failure is reported as a download failure. Implicitly, it
            # surfaces from `run` and reads as a container that would not start,
            # which sends the reader looking in the wrong place.
            try:
                self.client.images.get(image)
            except NotFound:
                try:
                    self.client.images.pull(image)
                except Exception as error:
                    raise SandboxError(
                        SandboxStage.image_pull,
                        f'The sandbox image {image!r} is not present locally and could not be '
                        f'downloaded: {error}',
                    ) from error

            try:
                existing: Any = self.client.containers.get(self._container_name)
                existing.remove(force=True)
            except NotFound:
                pass

            # `host_mount_source` raises its own staged error, which must not be
            # relabelled as a start failure, so it is resolved before the attempt.
            mount_source: str = self.host_mount_source()

            try:
                return self.client.containers.run(
                    image,
                    command='tail -f /dev/null',  # Keep it running for the session.
                    detach=True,
                    name=self._container_name,
                    environment=dict(self._secrets),
                    volumes={mount_source: {'bind': '/workspace', 'mode': 'rw'}},
                    working_dir='/workspace',
                )
            except Exception as error:
                raise SandboxError(
                    SandboxStage.container_start,
                    f'The sandbox container could not be started from image {image!r}: {error}',
                ) from error

        container: Any = await asyncio.to_thread(_spawn)
        await self._verify_ready()
        return container

    async def _verify_ready(self) -> None:
        """
        Confirm the container can actually run commands.

        A container that starts and immediately dies would otherwise only be
        discovered when the agent's first test command fails for no clear reason.

        :raises SandboxError: If the container does not respond.
        """
        def _probe() -> tuple[int, str, bool]:
            container: Any = self.client.containers.get(self._container_name)
            check: Any = container.exec_run(['/bin/sh', '-c', 'echo ok'])
            has_timeout: Any = container.exec_run(['/bin/sh', '-c', 'command -v timeout'])
            return (
                int(check.exit_code),
                check.output.decode('utf-8', errors='replace'),
                int(has_timeout.exit_code) == 0,
            )

        try:
            exit_code, output, has_timeout = await asyncio.to_thread(_probe)
        except Exception as error:
            raise SandboxError(
                SandboxStage.readiness_probe,
                f'The sandbox container started but could not be reached: {error}',
            ) from error

        if exit_code != 0 or 'ok' not in output:
            raise SandboxError(
                SandboxStage.readiness_probe,
                'The sandbox container started but did not respond to a test command. '
                f'It exited with status {exit_code}. Check that the image provides /bin/sh.',
            )

        self._supports_shell_timeout = has_timeout

    async def run_test_command(self, command: str) -> CommandResult:
        """
        Execute a command inside the sandbox container.

        The exit status is reported alongside the output so callers can tell a
        passing command from a failing one, and any secret values are stripped
        before the output leaves this method.

        :param command: The shell command to run.
        :returns: The exit status and sanitised output of the command.
        """
        wrapped: str = (
            f'timeout {int(self._command_timeout)} /bin/sh -c {_quote(command)}'
            if self._supports_shell_timeout
            else command
        )

        def _run() -> CommandResult:
            container: Any = self.client.containers.get(self._container_name)
            result: Any = container.exec_run(['/bin/sh', '-c', wrapped])
            exit_code: int = int(result.exit_code)
            return CommandResult(
                exit_code=exit_code,
                output=self.redact(result.output.decode('utf-8', errors='replace')),
                timed_out=exit_code == TIMEOUT_EXIT_CODE and self._supports_shell_timeout,
            )

        try:
            # The in-container timeout does the killing; this is a backstop for a
            # container that has stopped responding altogether.
            return await asyncio.wait_for(
                asyncio.to_thread(_run), timeout=self._command_timeout + GRACE_SECONDS
            )
        except TimeoutError:
            return CommandResult(
                exit_code=TIMEOUT_EXIT_CODE,
                output=(
                    f'The command did not finish within {self._command_timeout:.0f} seconds '
                    'and was abandoned. Try a narrower command, such as a single test file.'
                ),
                timed_out=True,
            )

    async def stop_sandbox(self) -> None:
        """
        Stop and remove this session's container, and any image built for it.
        """
        def _stop() -> None:
            try:
                container: Any = self.client.containers.get(self._container_name)
                container.remove(force=True)
            except NotFound:
                pass
            except Exception as error:
                logger.warning('Could not remove %s: %s', self._container_name, error)

            if self._built_image is not None:
                try:
                    self.client.images.remove(self._built_image, force=True)
                except Exception as error:
                    logger.warning('Could not remove image %s: %s', self._built_image, error)
                self._built_image = None

            # The client holds connections to the Docker socket. One per session
            # would accumulate now that the manager is no longer a singleton.
            if self._client is not None:
                try:
                    self._client.close()
                except Exception as error:
                    logger.warning('Could not close the Docker client: %s', error)
                self._client = None

        await asyncio.to_thread(_stop)


def _quote(command: str) -> str:
    """
    Quote a command so it survives being passed to an inner shell.

    :param command: The command to quote.
    :returns: The single-quoted command.
    """
    return "'" + command.replace("'", "'\\''") + "'"
