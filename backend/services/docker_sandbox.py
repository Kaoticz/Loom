import docker
from typing import Any
import asyncio

class DockerSandboxManager:
    """
    Manager for Docker sandbox environments.
    """
    client: Any
    repo_path: str
    container_name: str

    def __init__(self, repo_path: str):
        """
        :param repo_path: The host path to the repository to bind mount.
        """
        self.client = docker.from_env()
        self.repo_path = repo_path
        self.container_name = 'loom-sandbox'

    async def spawn_sandbox(self, image: str = 'python:3.13-slim') -> Any:
        """
        Spawn a background Docker container with a bind mount.

        :param image: The Docker image to use for the sandbox.
        :returns: The created container instance.
        """
        def _spawn():
            # Kill existing container if any
            try:
                container = self.client.containers.get(self.container_name)
                container.remove(force=True)
            except docker.errors.NotFound: # type: ignore
                pass

            return self.client.containers.run(
                image,
                command='tail -f /dev/null', # Keep it running
                detach=True,
                name=self.container_name,
                volumes={
                    self.repo_path: {'bind': '/workspace', 'mode': 'rw'}
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
        def _run():
            container = self.client.containers.get(self.container_name)
            result = container.exec_run(command)
            return result.output.decode('utf-8')
        
        return await asyncio.to_thread(_run)

    async def stop_sandbox(self) -> None:
        """
        Stop and remove the sandbox container.
        """
        def _stop():
            try:
                container = self.client.containers.get(self.container_name)
                container.remove(force=True)
            except docker.errors.NotFound: # type: ignore
                pass
        
        await asyncio.to_thread(_stop)
