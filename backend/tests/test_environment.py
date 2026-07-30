from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

BACKEND_ROOT: Path = Path(__file__).resolve().parent.parent
ENV_FILE: Path = BACKEND_ROOT / '.env'

PROBE: str = textwrap.dedent(
    f'''
    import os, sys
    sys.path.insert(0, {str(BACKEND_ROOT)!r})
    from environment import load_environment
    load_environment()
    print(os.getenv('LOOM_PROBE_VALUE', '<default>'))
    '''
)


def resolve(env_file_contents: str | None, shell_value: str | None) -> str:
    """
    Resolve a variable in a fresh process, under the given conditions.

    Run out of process because the loader is deliberately idempotent, so its
    effect cannot be repeated within one interpreter.

    :param env_file_contents: What to write to backend/.env, or None for no file.
    :param shell_value: The value to export, or None to leave it unset.
    :returns: The value the process resolved.
    """
    environment: dict[str, str] = {k: v for k, v in os.environ.items() if k != 'LOOM_PROBE_VALUE'}
    if shell_value is not None:
        environment['LOOM_PROBE_VALUE'] = shell_value

    existing: str | None = ENV_FILE.read_text() if ENV_FILE.is_file() else None
    try:
        if env_file_contents is None:
            ENV_FILE.unlink(missing_ok=True)
        else:
            ENV_FILE.write_text(env_file_contents)

        result = subprocess.run(
            [sys.executable, '-c', PROBE], capture_output=True, text=True, env=environment
        )
        assert result.returncode == 0, result.stderr
        return result.stdout.strip()
    finally:
        if existing is None:
            ENV_FILE.unlink(missing_ok=True)
        else:
            ENV_FILE.write_text(existing)


@pytest.mark.skipif(ENV_FILE.is_file(), reason='a real backend/.env is present')
def test_default_is_used_when_nothing_is_configured() -> None:
    """
    With no file and no shell variable, the code's default applies.
    """
    assert resolve(None, None) == '<default>'


@pytest.mark.skipif(ENV_FILE.is_file(), reason='a real backend/.env is present')
def test_the_env_file_is_read() -> None:
    """
    A value in backend/.env is picked up without any shell configuration.
    """
    assert resolve('LOOM_PROBE_VALUE=from-file\n', None) == 'from-file'


@pytest.mark.skipif(ENV_FILE.is_file(), reason='a real backend/.env is present')
def test_the_shell_wins_over_the_file() -> None:
    """
    A variable already in the environment beats the file.

    This is the precedence Docker Compose and Vite both use, so a value injected
    by the container runtime overrides a developer's local file rather than
    being silently replaced by it.
    """
    assert resolve('LOOM_PROBE_VALUE=from-file\n', 'from-shell') == 'from-shell'


@pytest.mark.skipif(ENV_FILE.is_file(), reason='a real backend/.env is present')
def test_a_missing_file_is_not_an_error() -> None:
    """
    A fresh clone with no .env starts normally.
    """
    assert resolve(None, 'from-shell') == 'from-shell'


def test_loading_is_idempotent() -> None:
    """
    Repeated loading does not re-read or re-apply the file.

    Both the application and aerich call it, and aerich imports the Tortoise
    configuration without going through main.
    """
    from environment import load_environment

    first: bool = load_environment()
    assert load_environment() == first


def test_the_example_file_documents_every_variable() -> None:
    """
    Every variable the backend reads appears in .env.example.

    An undocumented variable is one nobody knows they can set.
    """
    import re

    example: str = (BACKEND_ROOT / '.env.example').read_text()
    documented: set[str] = set(re.findall(r'^([A-Z_]+)=', example, flags=re.M))

    referenced: set[str] = set()
    for path in BACKEND_ROOT.rglob('*.py'):
        if '.venv' in str(path) or 'tests' in path.parts:
            continue
        source: str = path.read_text()
        # Direct reads, plus names passed to a helper that reads them, which a
        # getenv-only scan would miss.
        referenced |= set(re.findall(r"getenv\(\s*'([A-Z_]+)'", source))
        referenced |= set(re.findall(r"'(LOOM_[A-Z_]+)'", source))

    assert referenced, 'the scan found nothing, so it is not checking anything'
    assert referenced - documented == set(), (
        f'undocumented variables: {sorted(referenced - documented)}'
    )
