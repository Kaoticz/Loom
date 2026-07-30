from __future__ import annotations

import os
from collections.abc import Mapping, Sequence

# ast-grep is a pattern matcher, not an AST dumper, but it can match on
# tree-sitter node kinds rather than on syntax. Matching kinds is what makes
# extraction reliable: a syntactic pattern such as 'def $NAME($$$): $$$' silently
# fails on any function carrying a return annotation, and returns an empty list
# that looks exactly like a file with no symbols. Matching the node kind cannot
# fail that way, because it does not care how the definition is written.
#
# Node kind names come from each language's tree-sitter grammar, so they differ
# between languages even where the concept is the same.
# Categories whose match is a variable declaration holding an arrow function.
# These need a second condition on the assigned value, since a bare
# `variable_declarator` would also match every ordinary constant and local.
ARROW_FUNCTION_LANGUAGES: frozenset[str] = frozenset({'typescript', 'javascript'})

NODE_KINDS: Mapping[str, Mapping[str, tuple[str, ...]]] = {
    'python': {
        'function': ('function_definition',),
        'class': ('class_definition',),
    },
    'typescript': {
        'function': ('function_declaration', 'method_definition'),
        'class': ('class_declaration',),
        'interface': ('interface_declaration',),
        'type': ('type_alias_declaration',),
    },
    'javascript': {
        'function': ('function_declaration', 'method_definition'),
        'class': ('class_declaration',),
    },
}

LANGUAGE_BY_EXTENSION: Mapping[str, str] = {
    '.py': 'python',
    '.pyi': 'python',
    '.ts': 'typescript',
    '.tsx': 'typescript',
    '.mts': 'typescript',
    '.cts': 'typescript',
    '.js': 'javascript',
    '.jsx': 'javascript',
    '.mjs': 'javascript',
    '.cjs': 'javascript',
}


def language_for(file_path: str) -> str | None:
    """
    Determine which language's rules apply to a file.

    :param file_path: Path to the file, used for its extension.
    :returns: The ast-grep language name, or None if the language is unsupported.
    """
    extension: str = os.path.splitext(file_path)[1].lower()
    return LANGUAGE_BY_EXTENSION.get(extension)


def is_supported(file_path: str) -> bool:
    """
    Check whether symbols can be extracted from a file.

    :param file_path: Path to the file, used for its extension.
    :returns: True if rules are registered for this file's language.
    """
    return language_for(file_path) is not None


def kinds_for(language: str) -> Mapping[str, tuple[str, ...]]:
    """
    Look up the node kinds that make up each symbol category in a language.

    :param language: The ast-grep language name.
    :returns: Symbol category to node kinds, empty when the language is unsupported.
    """
    return NODE_KINDS.get(language, {})


def build_rules(language: str) -> str:
    """
    Build the ast-grep rule document for a language.

    One document is produced per symbol category, separated by `---`, so a
    single invocation covers every category. Languages where functions are
    commonly written as arrow functions assigned to a constant gain an extra
    document for that shape. Each match is tagged in the output
    with the rule's id, which is used directly as the symbol's kind.

    The name is captured through the node's `name` field rather than parsed out
    of the matched text, so it stays correct however the definition is written.

    :param language: The ast-grep language name.
    :returns: The rule document, empty when the language is unsupported.
    """
    categories: Mapping[str, tuple[str, ...]] = kinds_for(language)
    if not categories:
        return ''

    documents: list[str] = []
    for category, node_kinds in categories.items():
        alternatives: str = '\n'.join(f'      - kind: {kind}' for kind in node_kinds)
        documents.append(
            f'id: {category}\n'
            f'language: {language}\n'
            'rule:\n'
            '  all:\n'
            '    - any:\n'
            f'{alternatives}\n'
            '    - has:\n'
            '        field: name\n'
            '        pattern: $NAME\n'
        )

    if language in ARROW_FUNCTION_LANGUAGES:
        documents.append(
            'id: function\n'
            f'language: {language}\n'
            'rule:\n'
            '  all:\n'
            '    - kind: variable_declarator\n'
            '    - has:\n'
            '        field: name\n'
            '        pattern: $NAME\n'
            '    - has:\n'
            '        field: value\n'
            '        kind: arrow_function\n'
        )

    return '---\n'.join(documents)


def supported_languages() -> Sequence[str]:
    """
    List the languages symbols can be extracted from.

    :returns: The ast-grep language names, sorted.
    """
    return sorted(NODE_KINDS)
