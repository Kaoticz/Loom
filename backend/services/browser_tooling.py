from __future__ import annotations

import re

AGENT_BROWSER_INSTALL_COMMAND: str = 'npx --yes agent-browser install'
"""
How `agent-browser` is installed. Quoted verbatim wherever it is suggested, so
there is one spelling of it rather than several drifting copies.
"""

BROWSER_GUIDANCE: str = (
    'This repository contains a web frontend. If you need to test or verify it by driving a\n'
    'browser, use `agent-browser`, installed with `' + AGENT_BROWSER_INSTALL_COMMAND + '`.\n'
    'Do not use Playwright or Selenium; commands that install or invoke either are refused.\n'
    'You may run whatever commands are needed to install prerequisites first, including Node\n'
    'itself, inside the sandbox.'
)
"""
The supplementary instruction added to a task whose repository has a frontend.

Kept out of `CODER_SYSTEM_PROMPT` because it is conditional: a repository with
no browser to drive should not be told which browser tool to prefer.
"""

_PACKAGE: str = r'(?:@playwright/test|[\w.-]*(?:playwright|selenium)[\w.-]*)'
"""
The package names being watched for.

The surrounding character classes cover the family around each name —
`pytest-playwright`, `playwright-core`, `selenium-webdriver` — since matching
the bare word alone would miss every prefixed distribution. Installation
context is required separately, so a broad name pattern cannot fire on prose.
"""

_NODE_INSTALLER: str = r'(?:npm|pnpm|yarn|bun)\s+(?:install|add|i)\b'
_PYTHON_INSTALLER: str = r'(?:pip3?|uv|poetry|pipx|conda)\s+(?:install|add)\b'

_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Installing it through a Node package manager.
    re.compile(rf'\b{_NODE_INSTALLER}[^;&|]*?(?<![\w.-]){_PACKAGE}(?![\w.-])', re.IGNORECASE),
    # Installing it through a Python package manager.
    re.compile(rf'\b{_PYTHON_INSTALLER}[^;&|]*?(?<![\w.-]){_PACKAGE}(?![\w.-])', re.IGNORECASE),
    # Invoking it directly, as a bare command, through npx, or as a Python module.
    re.compile(
        r'(?:^|[;&|]|\bnpx\s+(?:--yes\s+|-y\s+)?|\bpython3?\s+-m\s+|\buvx\s+)\s*'
        r'(?<![\w.-])(?:playwright|selenium)(?![\w.-])',
        re.IGNORECASE,
    ),
    # `playwright install`, wherever it appears in a chain.
    re.compile(r'(?<![\w.-])playwright(?![\w.-])\s+install\b', re.IGNORECASE),
)
"""
What counts as installing or invoking Playwright or Selenium.

Each pattern requires installation or invocation context rather than the bare
word, so a command that merely mentions one — searching the repository for
existing references, say — still runs. The `[^;&|]` bound keeps an installer at
the start of a chain from claiming a package named after the `&&`.
"""


def browser_automation_refusal(command: str) -> str | None:
    """
    Check whether a shell command reaches for Playwright or Selenium.

    :param command: The command the agent wants to run.
    :returns: The refusal to report back, or None if the command may run.
    """
    if not any(pattern.search(command) for pattern in _PATTERNS):
        return None

    return (
        'This command was not run. Loom drives browsers with `agent-browser`, not with '
        'Playwright or Selenium.\n\n'
        f'Install it with `{AGENT_BROWSER_INSTALL_COMMAND}` and use it instead. Installing '
        'prerequisites such as Node first is fine.\n\n'
        'If you wanted Playwright or Selenium for something other than driving a browser, '
        'say so in your next step: there is no way to run it here, so find another approach '
        'rather than retrying this command.'
    )
