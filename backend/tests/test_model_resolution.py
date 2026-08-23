from __future__ import annotations

import pytest
from pydantic_ai.models.openai import OpenAIChatModel

from services.model_resolution import resolve_model

"""
Turning a configured model name into something an agent can run.

This module had no tests at all, and not merely no test file: every suite that
would have touched it replaced it with a stub, so none of it had ever run. A
mistake in the URL it builds would break every real session while the whole suite
stayed green, which is the kind of gap worth closing first.
"""


def test_a_name_without_the_ollama_prefix_is_passed_through() -> None:
    """
    Other providers are Pydantic AI's business, not ours.
    """
    assert resolve_model('openai:gpt-4o') == 'openai:gpt-4o'
    assert resolve_model('anthropic:claude-3-5-sonnet') == 'anthropic:claude-3-5-sonnet'


def test_an_ollama_name_becomes_a_configured_model() -> None:
    """
    Ollama is reached through its OpenAI-compatible endpoint.
    """
    resolved = resolve_model('ollama:qwen2.5-coder')

    assert isinstance(resolved, OpenAIChatModel)
    # The prefix is stripped: Ollama is asked for the model, not for the name
    # this project happens to call it by.
    assert resolved.model_name == 'qwen2.5-coder'


def test_only_the_first_colon_separates_the_prefix() -> None:
    """
    Model names contain colons of their own, and a tag must survive.

    Splitting on every colon would ask Ollama for `qwen2.5-coder` when the
    configuration said `qwen2.5-coder:14b`, quietly running a different model
    from the one that was chosen.
    """
    resolved = resolve_model('ollama:qwen2.5-coder:14b')

    assert isinstance(resolved, OpenAIChatModel)
    assert resolved.model_name == 'qwen2.5-coder:14b'


def test_the_configured_host_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    A host set in the environment reaches the client.
    """
    monkeypatch.setenv('OLLAMA_BASE_URL', 'http://elsewhere:9999')

    resolved = resolve_model('ollama:qwen2.5-coder')

    assert isinstance(resolved, OpenAIChatModel)
    assert 'elsewhere:9999' in str(resolved.client.base_url)


def test_the_default_host_is_used_when_none_is_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    The default is the compose service name, which is what runs unconfigured.
    """
    monkeypatch.delenv('OLLAMA_BASE_URL', raising=False)

    resolved = resolve_model('ollama:qwen2.5-coder')

    assert isinstance(resolved, OpenAIChatModel)
    assert 'ollama:11434' in str(resolved.client.base_url)


def test_the_openai_compatible_suffix_is_appended(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Ollama serves its OpenAI-compatible API under `/v1`.

    Asserted on its own because it is the single character most likely to be
    wrong and the least likely to be noticed: everything else about the
    configuration would look right, and every request would fail.
    """
    monkeypatch.setenv('OLLAMA_BASE_URL', 'http://ollama:11434')

    resolved = resolve_model('ollama:qwen2.5-coder')

    assert isinstance(resolved, OpenAIChatModel)
    assert str(resolved.client.base_url).rstrip('/').endswith('/v1')


def test_a_name_that_merely_contains_ollama_is_not_treated_as_one() -> None:
    """
    The prefix is a prefix, not a substring.
    """
    assert resolve_model('openai:ollama-tuned') == 'openai:ollama-tuned'
