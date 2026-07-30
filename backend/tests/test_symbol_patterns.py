from __future__ import annotations

import json
from pathlib import Path

import pytest

from services.indexing import IndexingService
from services.symbol_patterns import (
    NODE_KINDS,
    build_rules,
    is_supported,
    kinds_for,
    language_for,
    supported_languages,
)
from tests.conftest import ast_grep_available


def extract(binary: str, path: Path) -> dict[str, str]:
    """
    Run the registered rules for a file and collect the symbols found.

    :param binary: The ast-grep executable.
    :param path: The file to scan.
    :returns: The symbol names found, mapped to their kind.
    """
    language: str | None = language_for(str(path))
    if language is None:
        return {}

    process = IndexingService._run_ast_grep(binary, build_rules(language), str(path))
    if process.returncode != 0 or not process.stdout:
        return {}

    found: dict[str, str] = {}
    for match in json.loads(process.stdout):
        name = IndexingService._extract_symbol_name(match)
        if name is not None:
            found[name] = str(match.get('ruleId'))
    return found


@pytest.mark.parametrize(
    'path,supported',
    [
        ('a.py', True), ('a.pyi', True),
        ('a.ts', True), ('a.tsx', True), ('a.js', True), ('a.jsx', True),
        ('a.svelte', False), ('a.md', False), ('a.json', False), ('Makefile', False),
    ],
)
def test_language_support_is_declared_by_extension(path: str, supported: bool) -> None:
    """
    Whether a file can yield symbols is decided by its extension alone.
    """
    assert is_supported(path) is supported


def test_extension_lookup_is_case_insensitive() -> None:
    """
    An uppercase extension resolves to the same language.
    """
    assert language_for('MODULE.PY') == language_for('module.py') == 'python'


def test_rules_match_node_kinds_not_syntax() -> None:
    """
    Rules select tree-sitter node kinds rather than describing source text.

    A syntactic pattern silently fails on any definition written differently
    from the pattern, which is what reduced Python extraction to 12% coverage.
    Matching the node kind cannot fail that way.
    """
    for language in supported_languages():
        document: str = build_rules(language)
        assert 'kind:' in document
        assert '$$$' not in document, f'{language} rules describe syntax'
        assert 'field: name' in document, f'{language} rules do not capture a name'


def test_every_language_declares_functions_and_classes() -> None:
    """
    Each supported language extracts at least functions and classes.
    """
    for language in supported_languages():
        categories = kinds_for(language)
        assert 'function' in categories
        assert 'class' in categories


def test_unsupported_language_produces_no_rules() -> None:
    """
    A language with no registered kinds yields nothing to run.
    """
    assert build_rules('cobol') == ''
    assert kinds_for('cobol') == {}


def test_python_symbols_including_annotated(ast_grep_bin: str, tmp_path: Path) -> None:
    """
    Python extraction covers annotated, async, decorated and nested definitions.
    """
    if not ast_grep_available(ast_grep_bin):
        pytest.skip('ast-grep is not installed')

    path: Path = tmp_path / 'sample.py'
    path.write_text(
        'def plain(a):\n    return a\n\n\n'
        'def annotated(a: int) -> str:\n    return str(a)\n\n\n'
        'async def coro(b: int) -> None:\n    return None\n\n\n'
        '@decorator\ndef decorated() -> int:\n    return 1\n\n\n'
        'class Derived(Base, metaclass=Meta):\n'
        '    async def method(self, x: int) -> bool:\n        return True\n'
    )

    assert extract(ast_grep_bin, path) == {
        'plain': 'function',
        'annotated': 'function',
        'coro': 'function',
        'decorated': 'function',
        'method': 'function',
        'Derived': 'class',
    }


def test_typescript_symbols_including_methods(ast_grep_bin: str, tmp_path: Path) -> None:
    """
    TypeScript extraction covers methods, which no syntactic pattern matched.
    """
    if not ast_grep_available(ast_grep_bin):
        pytest.skip('ast-grep is not installed')

    path: Path = tmp_path / 'sample.ts'
    path.write_text(
        "import { thing } from './thing';\n\n"
        'export function exported(a: number): string {\n    return String(a);\n}\n\n'
        'export class Widget extends Base {\n'
        '    public compute(x: number): number {\n        return x;\n    }\n\n'
        '    async load(): Promise<void> {\n        await go();\n    }\n}\n\n'
        'export const arrow = (c: number): number => c * 2;\n\n'
        'const plainConst = 42;\n\n'
        'export interface Shape {\n    area(): number;\n}\n\n'
        'export type Alias = string | number;\n'
    )

    found: dict[str, str] = extract(ast_grep_bin, path)
    assert found['exported'] == 'function'
    assert found['compute'] == 'function'
    assert found['load'] == 'function'
    assert found['arrow'] == 'function'
    assert found['Widget'] == 'class'
    assert found['Shape'] == 'interface'
    assert found['Alias'] == 'type'
    assert 'plainConst' not in found, 'ordinary constants are not symbols'


def test_javascript_symbols(ast_grep_bin: str, tmp_path: Path) -> None:
    """
    JavaScript extraction covers functions, classes and methods.
    """
    if not ast_grep_available(ast_grep_bin):
        pytest.skip('ast-grep is not installed')

    path: Path = tmp_path / 'sample.js'
    path.write_text(
        'export function exported(a) {\n    return a;\n}\n\n'
        'export class Widget extends Base {\n    compute(x) {\n        return x;\n    }\n}\n'
    )

    found: dict[str, str] = extract(ast_grep_bin, path)
    assert found['exported'] == 'function'
    assert found['Widget'] == 'class'
    assert found['compute'] == 'function'


def test_ranges_span_whole_definitions(ast_grep_bin: str, tmp_path: Path) -> None:
    """
    A symbol's range covers its body, in every supported language.

    Chunking on symbol boundaries depends on this being true.
    """
    if not ast_grep_available(ast_grep_bin):
        pytest.skip('ast-grep is not installed')

    # The TypeScript body closes with a brace on its own line, so it is a line longer.
    for name, source, symbol, expected in (
        ('a.py', 'def widget(a: int) -> int:\n    x = 1\n    return x + a\n', 'widget', (1, 3)),
        ('a.ts', 'function widget(a: number): number {\n    const x = 1;\n    return x + a;\n}\n', 'widget', (1, 4)),
    ):
        path: Path = tmp_path / name
        path.write_text(source)

        language: str | None = language_for(str(path))
        assert language is not None
        process = IndexingService._run_ast_grep(ast_grep_bin, build_rules(language), str(path))
        spans: list[tuple[int, int]] = [
            span
            for match in json.loads(process.stdout)
            if IndexingService._extract_symbol_name(match) == symbol
            for span in [IndexingService._extract_line_range(match)]
            if span is not None
        ]

        assert spans == [expected], f'{name} produced {spans}'


def test_all_categories_run_in_one_invocation(ast_grep_bin: str, tmp_path: Path) -> None:
    """
    A file is scanned once, not once per symbol category.

    Each match carries the id of the rule that produced it, so the kind is read
    from the output rather than inferred from which invocation returned it.
    """
    if not ast_grep_available(ast_grep_bin):
        pytest.skip('ast-grep is not installed')

    path: Path = tmp_path / 'sample.py'
    path.write_text('def alpha() -> None:\n    pass\n\n\nclass Beta:\n    pass\n')

    process = IndexingService._run_ast_grep(ast_grep_bin, build_rules('python'), str(path))
    kinds: set[str] = {str(match.get('ruleId')) for match in json.loads(process.stdout)}

    assert kinds == {'function', 'class'}


def test_node_kinds_are_declared_per_language() -> None:
    """
    Node kind names differ between grammars and are not shared.

    Python calls a function `function_definition`; TypeScript calls it
    `function_declaration`. Reusing one language's names for another silently
    matches nothing.
    """
    assert NODE_KINDS['python']['function'] == ('function_definition',)
    assert 'function_declaration' in NODE_KINDS['typescript']['function']
    assert 'method_definition' in NODE_KINDS['typescript']['function']
