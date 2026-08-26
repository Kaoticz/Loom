import json
import os

import pytest

from services.browser_tooling import (
    AGENT_BROWSER_INSTALL_COMMAND,
    BROWSER_GUIDANCE,
    browser_automation_refusal,
)
from services.frontend_detection import repository_has_frontend


def write(root: str, relative: str, content: str) -> None:
    """
    Write a file into a scratch repository, creating its directories.

    :param root: The repository root.
    :param relative: Path to the file, relative to the root.
    :param content: What to write.
    """
    path: str = os.path.join(root, relative)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write(content)


def manifest(**sections: dict[str, str]) -> str:
    """
    Render a `package.json`.

    :param sections: The dependency sections to include.
    :returns: The manifest as JSON.
    """
    return json.dumps({'name': 'scratch', **sections})


BLOCKED_COMMANDS: list[str] = [
    'playwright install',
    'playwright install --with-deps chromium',
    'npx playwright test',
    'npx --yes playwright install chromium',
    'pip install playwright',
    'pip3 install pytest-playwright',
    'npm install -D @playwright/test',
    'npm i playwright',
    'pnpm add -D playwright-core',
    'yarn add selenium-webdriver',
    'pip install selenium',
    'poetry add selenium',
    'uv add playwright',
    'python -m playwright install',
    'cd frontend && npx playwright test',
    'npm run build && playwright install',
]

ALLOWED_COMMANDS: list[str] = [
    AGENT_BROWSER_INSTALL_COMMAND,
    'npx --yes agent-browser install && npm run dev',
    'npm install agent-browser',
    'npm install',
    'npm install express',
    'npm run test',
    'apt-get install -y nodejs',
    'curl -fsSL https://deb.nodesource.com/setup_22.x | bash -',
    'node --version',
    'pytest tests/',
    'ls node_modules',
    'cat package.json',
]


@pytest.mark.parametrize('command', BLOCKED_COMMANDS)
def test_playwright_and_selenium_commands_are_refused(command: str) -> None:
    """
    Installing or invoking either tool is caught, however it is spelled.
    """
    refusal: str | None = browser_automation_refusal(command)

    assert refusal is not None, f'{command!r} should have been refused'
    assert 'agent-browser' in refusal


@pytest.mark.parametrize('command', ALLOWED_COMMANDS)
def test_ordinary_commands_are_not_refused(command: str) -> None:
    """
    Nothing unrelated is caught, `agent-browser` and Node installs least of all.

    A false positive here is worse than a missed one: it blocks work the agent
    was told to do, with a message about a tool it never asked for.
    """
    assert browser_automation_refusal(command) is None, f'{command!r} should have run'


def test_merely_mentioning_the_tools_is_not_refused() -> None:
    """
    Searching for existing references is reading, not running.

    The refusal is about installing or invoking, so a command that names the
    tool without doing either has to survive.
    """
    assert browser_automation_refusal('grep -rn playwright .') is None
    assert browser_automation_refusal('rg selenium docs/') is None
    assert browser_automation_refusal("echo 'we do not use selenium'") is None


def test_the_refusal_explains_itself_and_names_the_alternative() -> None:
    """
    A legitimately different need must stay visible rather than be swallowed.

    The agent is told why nothing ran, what to use instead, and not to retry,
    so it adapts rather than looping on a command that can never succeed.
    """
    refusal: str | None = browser_automation_refusal('pip install playwright')

    assert refusal is not None
    assert 'was not run' in refusal
    assert AGENT_BROWSER_INSTALL_COMMAND in refusal
    assert 'rather than retrying' in refusal


def test_a_react_repository_has_a_frontend(tmp_path: object) -> None:
    """
    A declared dependency on a frontend framework is the signal.
    """
    root: str = str(tmp_path)
    write(root, 'package.json', manifest(dependencies={'react': '^18.0.0'}))

    assert repository_has_frontend(root) is True


def test_a_dev_dependency_on_a_bundler_counts(tmp_path: object) -> None:
    """
    A build tool is usually a dev dependency, so those sections are read too.
    """
    root: str = str(tmp_path)
    write(root, 'frontend/package.json', manifest(devDependencies={'vite': '^5.0.0'}))

    assert repository_has_frontend(root) is True


def test_a_backend_only_node_repository_has_no_frontend(tmp_path: object) -> None:
    """
    A `package.json` alone is not enough.

    An Express API is a Node project with no browser to drive, and telling its
    agent which browser tool to prefer would be noise on every task.
    """
    root: str = str(tmp_path)
    write(root, 'package.json', manifest(dependencies={'express': '^4.0.0'}))

    assert repository_has_frontend(root) is False


def test_a_python_repository_has_no_frontend(tmp_path: object) -> None:
    """
    A repository with no manifest at all is not a frontend.
    """
    root: str = str(tmp_path)
    write(root, 'app/main.py', 'print("hello")\n')
    write(root, 'pyproject.toml', '[project]\nname = "scratch"\n')

    assert repository_has_frontend(root) is False


def test_a_root_index_html_counts_as_a_frontend(tmp_path: object) -> None:
    """
    A Vite project keeps `index.html` at its own root.
    """
    root: str = str(tmp_path)
    write(root, 'index.html', '<!doctype html><html></html>')

    assert repository_has_frontend(root) is True


def test_a_buried_index_html_does_not_count(tmp_path: object) -> None:
    """
    Generated documentation is not an application.

    Counting an `index.html` at any depth would make every repository with
    built docs look like a frontend.
    """
    root: str = str(tmp_path)
    write(root, 'docs/build/html/index.html', '<!doctype html><html></html>')

    assert repository_has_frontend(root) is False


def test_a_malformed_manifest_is_not_a_frontend(tmp_path: object) -> None:
    """
    A manifest that will not parse is the repository's problem, not a crash.
    """
    root: str = str(tmp_path)
    write(root, 'package.json', '{ this is not json')

    assert repository_has_frontend(root) is False


def test_a_missing_repository_is_not_a_frontend() -> None:
    """
    Guidance is an optimisation: an unreadable path runs the task without it.
    """
    assert repository_has_frontend('/tmp/loom-does-not-exist-anywhere') is False


def test_the_guidance_names_the_tool_and_the_install_command() -> None:
    """
    The instruction has to be actionable on its own.
    """
    assert 'agent-browser' in BROWSER_GUIDANCE
    assert AGENT_BROWSER_INSTALL_COMMAND in BROWSER_GUIDANCE
    assert 'Playwright' in BROWSER_GUIDANCE
    assert 'Selenium' in BROWSER_GUIDANCE
