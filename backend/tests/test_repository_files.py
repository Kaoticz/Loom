from __future__ import annotations

import subprocess
from pathlib import Path

from services.repository_files import list_indexable_files


def repository(root: Path) -> None:
    """
    Turn a directory into a git repository.

    :param root: The directory to initialise.
    """
    subprocess.run(['git', 'init', '-q', str(root)], capture_output=True, check=True)
    subprocess.run(
        ['git', '-C', str(root), 'config', 'user.email', 'test@example.com'],
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ['git', '-C', str(root), 'config', 'user.name', 'Test'],
        capture_output=True,
        check=True,
    )


def track(root: Path, *paths: str) -> None:
    """
    Commit the given paths, so git considers them tracked.

    :param root: The repository root.
    :param paths: Repository-relative paths to add.
    """
    subprocess.run(
        ['git', '-C', str(root), 'add', '-f', *paths], capture_output=True, check=True
    )
    subprocess.run(
        ['git', '-C', str(root), 'commit', '-qm', 'add'], capture_output=True, check=True
    )


def found(root: Path) -> set[str]:
    """
    List what would be indexed, as repository-relative paths.

    :param root: The repository root.
    :returns: The relative paths, with forward slashes.
    """
    return {
        Path(path).relative_to(root).as_posix() for path in list_indexable_files(str(root))
    }


def test_a_file_the_repository_ignores_is_not_indexed(tmp_path: Path) -> None:
    """
    The point of the exercise: build output should not be embedded.
    """
    repository(tmp_path)
    (tmp_path / '.gitignore').write_text('artifacts/\n*.log\n')
    (tmp_path / 'main.py').write_text('x = 1\n')
    (tmp_path / 'run.log').write_text('noise\n')
    (tmp_path / 'artifacts').mkdir()
    (tmp_path / 'artifacts' / 'bundle.js').write_text('generated\n')

    indexed: set[str] = found(tmp_path)

    assert 'main.py' in indexed
    assert 'run.log' not in indexed
    assert 'artifacts/bundle.js' not in indexed


def test_a_nested_ignore_file_applies_to_its_own_subtree(tmp_path: Path) -> None:
    """
    Each `.gitignore` governs the directory it sits in and below.

    Reading only the root file would index everything the inner one excludes,
    and matching its patterns against the whole tree would exclude far too much.
    """
    repository(tmp_path)
    (tmp_path / 'keep.tmp').write_text('kept at the root\n')
    (tmp_path / 'inner').mkdir()
    (tmp_path / 'inner' / '.gitignore').write_text('*.tmp\n')
    (tmp_path / 'inner' / 'scratch.tmp').write_text('ignored below\n')
    (tmp_path / 'inner' / 'code.py').write_text('y = 2\n')

    indexed: set[str] = found(tmp_path)

    assert 'keep.tmp' in indexed
    assert 'inner/code.py' in indexed
    assert 'inner/scratch.tmp' not in indexed


def test_a_negated_pattern_reinstates_a_file(tmp_path: Path) -> None:
    """
    A later `!` rule wins over an earlier exclusion.

    Order matters, and a pattern matcher applied without regard to it would drop
    a file the repository deliberately kept.
    """
    repository(tmp_path)
    (tmp_path / '.gitignore').write_text('*.cfg\n!keep.cfg\n')
    (tmp_path / 'drop.cfg').write_text('a\n')
    (tmp_path / 'keep.cfg').write_text('b\n')

    indexed: set[str] = found(tmp_path)

    assert 'keep.cfg' in indexed
    assert 'drop.cfg' not in indexed


def test_a_tracked_file_is_indexed_even_if_a_pattern_matches_it(tmp_path: Path) -> None:
    """
    Git ignores nothing it is already tracking, and neither should indexing.

    A file someone committed deliberately is part of the project whatever a
    later pattern says, and the reader will expect to be able to ask about it.
    """
    repository(tmp_path)
    (tmp_path / 'committed.log').write_text('deliberately kept\n')
    track(tmp_path, 'committed.log')
    (tmp_path / '.gitignore').write_text('*.log\n')
    (tmp_path / 'fresh.log').write_text('not kept\n')

    indexed: set[str] = found(tmp_path)

    assert 'committed.log' in indexed
    assert 'fresh.log' not in indexed


def test_the_repositorys_own_metadata_is_never_indexed(tmp_path: Path) -> None:
    """
    `.git` holds every version of everything, and none of it is source.
    """
    repository(tmp_path)
    (tmp_path / 'main.py').write_text('x = 1\n')
    track(tmp_path, 'main.py')

    indexed: set[str] = found(tmp_path)

    assert indexed == {'main.py'}
    assert not any(path.startswith('.git/') for path in indexed)


def test_build_directories_are_skipped_even_when_not_ignored(tmp_path: Path) -> None:
    """
    The backstop: a repository that does not ignore its own dependencies would
    otherwise have tens of thousands of files embedded.
    """
    repository(tmp_path)
    (tmp_path / 'main.py').write_text('x = 1\n')
    (tmp_path / 'node_modules').mkdir()
    (tmp_path / 'node_modules' / 'left-pad.js').write_text('module.exports = 1\n')

    indexed: set[str] = found(tmp_path)

    assert 'main.py' in indexed
    assert 'node_modules/left-pad.js' not in indexed


def test_a_directory_that_is_not_a_repository_is_still_indexed(tmp_path: Path) -> None:
    """
    There are no ignore rules to obey, but there is still code to read.

    Refusing would leave the session unable to start over something that is not
    an error.
    """
    (tmp_path / 'main.py').write_text('x = 1\n')
    (tmp_path / '__pycache__').mkdir()
    (tmp_path / '__pycache__' / 'main.cpython-314.pyc').write_bytes(b'\x00')

    indexed: set[str] = found(tmp_path)

    assert 'main.py' in indexed
    assert '__pycache__/main.cpython-314.pyc' not in indexed


def test_a_tracked_file_that_has_been_deleted_is_not_offered(tmp_path: Path) -> None:
    """
    Git lists a tracked file whether or not it is still on disk, and a path that
    is gone cannot be read.
    """
    repository(tmp_path)
    (tmp_path / 'gone.py').write_text('x = 1\n')
    (tmp_path / 'here.py').write_text('y = 2\n')
    track(tmp_path, 'gone.py', 'here.py')
    (tmp_path / 'gone.py').unlink()

    indexed: set[str] = found(tmp_path)

    assert indexed == {'here.py'}
