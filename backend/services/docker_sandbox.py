import docker
import os
from typing import Any
import asyncio

class DockerSandboxManager:
    """
    Manager for Docker sandbox environments.
    """
    _client: Any | None = None
    _repo_path: str
    container_name: str

    def __init__(self, repo_path: str):
        """
        :param repo_path: The host path to the repository to bind mount.
        """
        self._repo_path = repo_path
        self.container_name = 'loom-sandbox'

    @property
    def repo_path(self) -> str:
        return self._repo_path

    @repo_path.setter
    def repo_path(self, value: str) -> None:
        if self._repo_path != value:
            self._repo_path = value
            # We don't necessarily need to reset self._client here as it's the docker client,
            # but it doesn't hurt if we want to be super safe.
            # However, repo_path is used in spawn_sandbox.

    @property
    def client(self) -> Any:
        """
        Lazy-loaded Docker client.
        """
        if self._client is None:
            try:
                self._client = docker.from_env()
                # Test connection
                self._client.ping()
            except Exception as e:
                if "Permission denied" in str(e) or (hasattr(e, 'errno') and e.errno == 13):
                    raise RuntimeError(
                        "Permission denied when connecting to Docker socket. "
                        "Please ensure the user running the application has read/write access to /var/run/docker.sock. "
                        "If using Docker, you might need 'sudo usermod -aG docker $USER'. "
                        "If using Podman, ensure the podman.socket is running and the path is correct."
                    ) from e
                raise
        return self._client

    async def spawn_sandbox(self, image: str = 'python:3.13-slim') -> Any:
        """
        Spawn a background Docker container with a bind mount.

        :param image: The Docker image to use for the sandbox.
        :returns: The created container instance.
        """
        def _spawn() -> Any:
            # Kill existing container if any
            try:
                from docker.models.containers import Container
                container: Container = self.client.containers.get(self.container_name)
                container.remove(force=True)
            except docker.errors.NotFound: # type: ignore
                pass

            return self.client.containers.run(
                image,
                command='tail -f /dev/null', # Keep it running
                detach=True,
                name=self.container_name,
                volumes={
                    os.path.abspath(self.repo_path): {'bind': '/workspace', 'mode': 'rw'}
                },
                working_dir='/workspace'
            )
        
        return await asyncio.to_thread(_spawn)

    async def run_test_command(self, command: str) -> str:
        """
        Execute a command inside the sandbox container.

        :param command: The shell command to run.
        :returns: The stdout/stderr output of the command.
        """
        def _run() -> str:
            from docker.models.containers import Container
            container: Container = self.client.containers.get(self.container_name)
            result: Any = container.exec_run(command)
            output: str = result.output.decode('utf-8')
            return output
        
        return await asyncio.to_thread(_run)

    async def stop_sandbox(self) -> None:
        """
        Stop and remove the sandbox container.
        """
        def _stop() -> None:
            try:
                from docker.models.containers import Container
                container: Container = self.client.containers.get(self.container_name)
                container.remove(force=True)
            except docker.errors.NotFound: # type: ignore
                pass
        
        await asyncio.to_thread(_stop)
