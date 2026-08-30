from __future__ import annotations

import os
from typing import Annotated

from aiofiles import open as aio_open
from pydantic import BaseModel, Field


class EditResult(BaseModel):
    """
    Outcome of a surgical file edit.
    """

    success: Annotated[bool, Field(description='Whether the search block was found and replaced.')]
    match_count: Annotated[int, Field(ge=0, description='How many times the search block occurred in the file. Only the first occurrence is replaced.')]
    message: Annotated[str, Field(description='Human readable description of the outcome, suitable for returning to the agent.')]


class FileEditorService:
    """
    Service for surgical file modifications.
    """
    root_path: str

    def __init__(self, root_path: str) -> None:
        """
        Create a file editor rooted at a repository.

        :param root_path: The directory that all edits are confined to.
        """
        self.root_path = root_path

    def _safe_path(self, path: str) -> str:
        """
        Validate and return an absolute path within the root.

        :param path: The relative path to resolve.
        :returns: The resolved absolute path.
        :raises ValueError: If the path escapes the root directory.
        """
        abs_root: str = os.path.abspath(self.root_path)
        safe_path: str = os.path.abspath(os.path.join(abs_root, path))
        if not safe_path.startswith(abs_root + os.sep) and safe_path != abs_root:
            raise ValueError(f'Path {path} is outside of root {abs_root}')
        return safe_path

    async def edit_file(self, path: str, search_block: str, replace_block: str) -> EditResult:
        """
        Apply a surgical edit to a file, replacing only the first occurrence.

        The number of occurrences is reported back so that the caller can tell
        when a search block was not unique and the edit may not have landed
        where it was intended.

        :param path: Relative path to the file.
        :param search_block: The block of code to find.
        :param replace_block: The block of code to replace it with.
        :returns: The outcome of the edit, including the number of matches found.
        :raises ValueError: If the path escapes the repository root.
        """
        safe_path: str = self._safe_path(path)
        if not os.path.exists(safe_path):
            return EditResult(
                success=False,
                match_count=0,
                message=f'File {path} does not exist.'
            )

        content: str
        async with aio_open(safe_path, 'r') as file:
            content = await file.read()

        match_count: int = content.count(search_block)
        if match_count == 0:
            return EditResult(
                success=False,
                match_count=0,
                message=(
                    f'The search block was not found in {path}. '
                    'Read the file again and copy the block to replace verbatim.'
                )
            )

        new_content: str = content.replace(search_block, replace_block, 1)

        async with aio_open(safe_path, 'w') as file:
            await file.write(new_content)

        if match_count > 1:
            return EditResult(
                success=True,
                match_count=match_count,
                message=(
                    f'Edited {path}, but the search block occurred {match_count} times and only '
                    'the first occurrence was replaced. Verify this was the intended location and '
                    'use a longer, more specific search block if it was not.'
                )
            )

        return EditResult(
            success=True,
            match_count=1,
            message=f'Edited {path} successfully.'
        )

    async def read_file(
        self,
        path: str,
        start_line: int | None = None,
        end_line: int | None = None
    ) -> str:
        """
        Read a file, optionally restricted to a range of lines.

        When a range is given the returned lines are prefixed with their 1-based
        line numbers, so the agent can locate them. The prefixes are presentation
        only and must not be copied into an `edit_file` search block.

        :param path: Relative path to the file.
        :param start_line: 1-based first line to return. Defaults to the start of the file.
        :param end_line: 1-based last line to return, inclusive. Defaults to the end of the file.
        :returns: The file contents, or the requested numbered slice of them.
        :raises ValueError: If the path escapes the repository root, or the range is invalid.
        :raises FileNotFoundError: If the file does not exist.
        """
        safe_path: str = self._safe_path(path)
        if not os.path.isfile(safe_path):
            raise FileNotFoundError(f'File {path} does not exist.')

        content: str
        async with aio_open(safe_path, 'r') as file:
            content = await file.read()

        if start_line is None and end_line is None:
            return content

        lines: list[str] = content.splitlines()
        first: int = 1 if start_line is None else start_line
        last: int = len(lines) if end_line is None else end_line

        if first < 1:
            raise ValueError('start_line must be 1 or greater.')
        if last < first:
            raise ValueError('end_line must be greater than or equal to start_line.')
        if first > len(lines):
            raise ValueError(f'start_line {first} is past the end of {path}, which has {len(lines)} lines.')

        last = min(last, len(lines))
        return '\n'.join(f'{number}: {lines[number - 1]}' for number in range(first, last + 1))

    async def create_file(self, path: str, content: str) -> None:
        """
        Create a new file.

        :param path: Relative path to the file.
        :param content: The content of the file.
        :raises ValueError: If the path escapes the repository root.
        """
        safe_path: str = self._safe_path(path)
        parent: str = os.path.dirname(safe_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        async with aio_open(safe_path, 'w') as file:
            await file.write(content)

    async def delete_file(self, path: str) -> str:
        """
        Permanently remove one file from the repository.

        :param path: Relative path to the file.
        :returns: The path that was deleted.
        :raises ValueError: If the path escapes the root, names the root, or is not a file.
        :raises FileNotFoundError: If the file does not exist.
        """
        safe_path: str = self._safe_path(path)
        root: str = os.path.abspath(self.root_path)
        if safe_path == root:
            raise ValueError('Refusing to delete the repository root.')
        if not os.path.exists(safe_path):
            raise FileNotFoundError(f'File {path} does not exist.')
        if not os.path.isfile(safe_path):
            raise ValueError(f'Path {path} is not a file.')

        os.remove(safe_path)
        return path
