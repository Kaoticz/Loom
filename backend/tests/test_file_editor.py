from __future__ import annotations

from pathlib import Path

import pytest

from services.edit_result import EditResult
from services.file_editor import FileEditorService


async def test_single_match_is_replaced(editor: FileEditorService, repo: str) -> None:
    """
    A unique search block is replaced and reported as one match.
    """
    target: Path = Path(repo, 'sample.py')
    target.write_text('alpha\nbeta\ngamma\n')

    result: EditResult = await editor.edit_file('sample.py', 'beta', 'BETA')

    assert isinstance(result, EditResult)
    assert result.success is True
    assert result.match_count == 1
    assert target.read_text() == 'alpha\nBETA\ngamma\n'


async def test_only_the_first_of_many_matches_is_replaced(editor: FileEditorService, repo: str) -> None:
    """
    A repeated search block replaces only its first occurrence and warns.
    """
    target: Path = Path(repo, 'sample.py')
    target.write_text('dup\ndup\ndup\n')

    result: EditResult = await editor.edit_file('sample.py', 'dup', 'FIXED')

    assert result.success is True
    assert result.match_count == 3
    assert target.read_text() == 'FIXED\ndup\ndup\n'
    assert 'only' in result.message.lower()


async def test_missing_search_block_fails(editor: FileEditorService, repo: str) -> None:
    """
    A search block that does not occur is reported as a failure.
    """
    Path(repo, 'sample.py').write_text('alpha\n')
    result: EditResult = await editor.edit_file('sample.py', 'nonexistent', 'x')

    assert result.success is False
    assert result.match_count == 0


async def test_missing_file_fails_cleanly(editor: FileEditorService) -> None:
    """
    Editing a file that does not exist reports the problem rather than raising.
    """
    result: EditResult = await editor.edit_file('nope.py', 'a', 'b')
    assert result.success is False
    assert 'does not exist' in result.message


@pytest.mark.parametrize('attempt', ['../escape.py', '../../etc/passwd', '../../../root/.ssh/id_rsa'])
async def test_path_traversal_is_blocked(editor: FileEditorService, attempt: str) -> None:
    """
    Paths outside the repository root are refused.
    """
    with pytest.raises(ValueError):
        await editor.edit_file(attempt, 'a', 'b')


async def test_sibling_prefix_traversal_is_blocked(tmp_path: Path) -> None:
    """
    A sibling directory sharing the root's name prefix cannot be reached.

    A naive `startswith` check would treat `/tmp/repo-evil` as inside `/tmp/repo`.
    """
    root: Path = tmp_path / 'repo'
    root.mkdir()
    sibling: Path = tmp_path / 'repo-evil'
    sibling.mkdir()
    (sibling / 'secret.py').write_text('secret')

    editor: FileEditorService = FileEditorService(root_path=str(root))
    with pytest.raises(ValueError):
        await editor.edit_file('../repo-evil/secret.py', 'secret', 'pwned')


async def test_create_file_makes_parent_directories(editor: FileEditorService, repo: str) -> None:
    """
    Creating a nested file creates the directories leading to it.
    """
    await editor.create_file('nested/deep/new.py', 'content here')
    assert Path(repo, 'nested/deep/new.py').read_text() == 'content here'


async def test_create_file_blocks_traversal(editor: FileEditorService) -> None:
    """
    Files cannot be created outside the repository root.
    """
    with pytest.raises(ValueError):
        await editor.create_file('../evil.py', 'x')


async def test_delete_file_removes_an_existing_file(editor: FileEditorService, repo: str) -> None:
    """
    Deleting a file removes it and reports the relative path.
    """
    target: Path = Path(repo, 'nested', 'obsolete.py')
    target.parent.mkdir()
    target.write_text('obsolete\n')

    assert await editor.delete_file('nested/obsolete.py') == 'nested/obsolete.py'
    assert not target.exists()


@pytest.mark.parametrize('attempt', ['../escape.py', '../../etc/passwd'])
async def test_delete_file_blocks_traversal(editor: FileEditorService, attempt: str) -> None:
    """
    Deletion cannot reach outside the repository root.
    """
    with pytest.raises(ValueError):
        await editor.delete_file(attempt)


async def test_delete_file_refuses_the_root_and_missing_files(editor: FileEditorService) -> None:
    """
    The repository root and nonexistent paths cannot be deleted.
    """
    with pytest.raises(ValueError, match='repository root'):
        await editor.delete_file('.')
    with pytest.raises(FileNotFoundError, match='does not exist'):
        await editor.delete_file('missing.py')


async def test_read_file_returns_full_content_unnumbered(editor: FileEditorService, repo: str) -> None:
    """
    Reading without a range returns the file exactly as stored.
    """
    Path(repo, 'a.py').write_text('one\ntwo\nthree\n')
    assert await editor.read_file('a.py') == 'one\ntwo\nthree\n'


async def test_read_file_range_is_numbered(editor: FileEditorService, repo: str) -> None:
    """
    A ranged read returns only those lines, each prefixed with its number.
    """
    Path(repo, 'a.py').write_text('one\ntwo\nthree\nfour\n')
    assert await editor.read_file('a.py', 2, 3) == '2: two\n3: three'


async def test_read_file_clamps_end_line(editor: FileEditorService, repo: str) -> None:
    """
    An end line past the file is clamped rather than rejected.
    """
    Path(repo, 'a.py').write_text('one\ntwo\n')
    assert await editor.read_file('a.py', 1, 99) == '1: one\n2: two'


async def test_read_file_rejects_bad_ranges(editor: FileEditorService, repo: str) -> None:
    """
    Invalid line ranges are rejected with an explanation.
    """
    Path(repo, 'a.py').write_text('one\ntwo\n')

    with pytest.raises(ValueError):
        await editor.read_file('a.py', 0)
    with pytest.raises(ValueError):
        await editor.read_file('a.py', 2, 1)
    with pytest.raises(ValueError):
        await editor.read_file('a.py', 99)


async def test_read_file_missing_raises(editor: FileEditorService) -> None:
    """
    Reading a file that does not exist raises FileNotFoundError.
    """
    with pytest.raises(FileNotFoundError):
        await editor.read_file('nope.py')
