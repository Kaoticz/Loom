from __future__ import annotations

import pytest

from database.models.file import File
from database.models.symbol import Symbol
from services.code_search import CodeSearchService, SymbolDefinition, extract_lines, LexicalMatch

SAMPLE: str = '''import os
from typing import Any


class UserRepository:
    """Fetches users."""

    def find_user(self, user_id: int) -> Any:
        return self._query(user_id)
'''

OTHER: str = 'def unrelated_helper(x):\n    return x * 2\n'


@pytest.fixture
async def indexed_files(sqlite_db: None) -> None:
    """
    Insert two indexed files for lexical search to operate on.

    :param sqlite_db: The database fixture this depends on.
    """
    await File.create(file_path='repo/users.py', raw_content=SAMPLE, hash='h1')
    await File.create(file_path='repo/helpers.py', raw_content=OTHER, hash='h2')


@pytest.mark.usefixtures('indexed_files')
async def test_finds_a_keyword_with_context(search: CodeSearchService) -> None:
    """
    A matching line is returned with its file, 1-based number and context.
    """
    hits: list[LexicalMatch] = await search.search_code_lexical('find_user')

    assert len(hits) > 0
    assert isinstance(hits[0], LexicalMatch)
    assert hits[0].file_path == 'repo/users.py'
    assert SAMPLE.splitlines()[hits[0].line_number - 1].strip().startswith('def find_user')
    assert 'find_user' in hits[0].snippet
    assert hits[0].snippet.splitlines()[0].split(':')[0].isdigit()


@pytest.mark.usefixtures('indexed_files')
async def test_search_is_case_insensitive(search: CodeSearchService) -> None:
    """
    Keywords match regardless of case.
    """
    hits: list[LexicalMatch] = await search.search_code_lexical('FIND_USER')
    assert len(hits) > 0
    assert hits[0].file_path == 'repo/users.py'


@pytest.mark.usefixtures('indexed_files')
async def test_matches_on_file_path(search: CodeSearchService) -> None:
    """
    A file whose path matches is returned even when its contents do not.
    """
    hits: list[LexicalMatch] = await search.search_code_lexical('helpers')
    assert any(hit.file_path == 'repo/helpers.py' for hit in hits)


@pytest.mark.usefixtures('indexed_files')
async def test_respects_the_limit(search: CodeSearchService) -> None:
    """
    No more results are returned than the caller asked for.
    """
    assert len(await search.search_code_lexical('user_id', limit=2)) <= 2


@pytest.mark.usefixtures('indexed_files')
@pytest.mark.parametrize(
    'keyword,limit',
    [('   ', 10), ('user', 0), ('zzz_not_present', 10)],
    ids=['blank_keyword', 'zero_limit', 'no_match'],
)
async def test_returns_nothing_for_unusable_queries(
    search: CodeSearchService, keyword: str, limit: int
) -> None:
    """
    Blank keywords, zero limits and absent terms all yield no results.
    """
    assert await search.search_code_lexical(keyword, limit=limit) == []


def test_semantic_query_prefetches_the_file_relation() -> None:
    """
    The semantic query must prefetch its file relation.

    Without it, `chunk.file` is a QuerySet rather than a File, and every caller
    that reads `chunk.file.file_path` raises AttributeError.
    """
    import inspect

    source: str = inspect.getsource(CodeSearchService.search_code_semantic)
    assert "prefetch_related('file')" in source


def test_semantic_query_does_not_interpolate_sql() -> None:
    """
    The embedding is bound through the query builder rather than formatted in.
    """
    import inspect

    source: str = inspect.getsource(CodeSearchService.search_code_semantic)
    assert 'RawSQL' not in source
    assert 'VectorDistance' in source


@pytest.mark.usefixtures('sqlite_db')
async def test_symbol_lookup_returns_the_definition(search: CodeSearchService) -> None:
    """
    A known symbol is found with its file and line range.
    """
    from database.models.symbol import Symbol

    file_obj = await File.create(file_path='repo/users.py', raw_content=SAMPLE, hash='h1')
    await Symbol.create(
        file=file_obj, symbol_name='find_user', kind='function', line_start=8, line_end=9
    )

    found = await search.get_symbol_definition('find_user')

    assert len(found) == 1
    assert found[0].kind == 'function'
    assert (found[0].line_start, found[0].line_end) == (8, 9)
    # The declaring file travels with the definition, so the caller needs no
    # knowledge of the rows behind it.
    assert found[0].file_path == 'repo/users.py'


@pytest.mark.usefixtures('sqlite_db')
async def test_unknown_symbol_returns_nothing(search: CodeSearchService) -> None:
    """
    An absent symbol yields an empty result rather than raising.
    """
    assert await search.get_symbol_definition('does_not_exist') == []


@pytest.mark.usefixtures('sqlite_db')
async def test_same_name_in_several_files_returns_all(search: CodeSearchService) -> None:
    """
    A name defined in more than one place returns every definition.

    Picking one arbitrarily would send the agent to the wrong file.
    """
    from database.models.symbol import Symbol

    for path in ('repo/a.py', 'repo/b.py'):
        file_obj = await File.create(file_path=path, raw_content='x', hash=path)
        await Symbol.create(
            file=file_obj, symbol_name='handler', kind='function', line_start=1, line_end=2
        )

    found = await search.get_symbol_definition('handler')

    assert len(found) == 2
    assert {found_symbol.file_path for found_symbol in found} == {'repo/a.py', 'repo/b.py'}


@pytest.mark.usefixtures('sqlite_db')
async def test_symbol_lookup_is_exact_not_partial(search: CodeSearchService) -> None:
    """
    A lookup matches the whole name, not a fragment of it.
    """
    from database.models.symbol import Symbol

    file_obj = await File.create(file_path='repo/a.py', raw_content='x', hash='h')
    await Symbol.create(
        file=file_obj, symbol_name='find_user_by_email', kind='function', line_start=1, line_end=2
    )

    assert await search.get_symbol_definition('find_user') == []


def test_extract_lines_takes_an_inclusive_range() -> None:
    """
    Line numbers from the index are 1-based and inclusive at both ends.
    """
    content: str = 'one\ntwo\nthree\nfour\n'

    assert extract_lines(content, 2, 3) == 'two\nthree'


def test_extract_lines_takes_a_single_line() -> None:
    """
    A one-line definition is a range of one, not an empty range.
    """
    assert extract_lines('one\ntwo\n', 2, 2) == 'two'


def test_extract_lines_clamps_a_range_past_the_end() -> None:
    """
    The index records where a symbol was, and the file may have shrunk since.
    """
    assert extract_lines('one\ntwo\n', 1, 99) == 'one\ntwo'


def test_extract_lines_returns_nothing_for_a_range_past_the_file() -> None:
    """
    A range entirely beyond the file selects nothing, rather than raising.
    """
    assert extract_lines('one\n', 5, 9) == ''


def test_extract_lines_survives_an_empty_file() -> None:
    """
    An indexed file that has since been emptied must not raise.
    """
    assert extract_lines('', 1, 3) == ''


def test_a_definition_renders_its_location_and_source() -> None:
    """
    The agent asks for a definition in order to read it.
    """
    definition: SymbolDefinition = SymbolDefinition(
        symbol_name='find_user',
        kind='function',
        file_path='repo/users.py',
        line_start=8,
        line_end=9,
        source='def find_user(name):\n    return name',
    )

    rendered: str = definition.render()

    assert 'function find_user' in rendered
    assert 'repo/users.py:8-9' in rendered
    assert 'def find_user(name):' in rendered


def test_a_definition_whose_source_has_gone_says_so() -> None:
    """
    Silence would read as an empty function body.
    """
    definition: SymbolDefinition = SymbolDefinition(
        symbol_name='gone',
        kind='function',
        file_path='repo/a.py',
        line_start=40,
        line_end=41,
        source='',
    )

    assert 'file may have changed' in definition.render()


@pytest.mark.usefixtures('sqlite_db')
async def test_a_looked_up_definition_carries_its_source(search: CodeSearchService) -> None:
    """
    The source is read from the indexed content, not from disk.
    """
    file_obj = await File.create(
        file_path='repo/users.py',
        raw_content='import os\n\n\ndef find_user(name):\n    return name\n',
        hash='h',
    )
    await Symbol.create(
        file=file_obj, symbol_name='find_user', kind='function', line_start=4, line_end=5
    )

    found = await search.get_symbol_definition('find_user')

    assert found[0].source == 'def find_user(name):\n    return name'


def test_extract_lines_floors_a_start_below_one() -> None:
    """
    A zero or negative start must not index from the end of the file.

    This is the one bound worth guarding: an over-long end simply yields what
    remains, but `lines[-1:]` silently returns the last line, so a bad start
    would produce confidently wrong source rather than none.
    """
    content: str = 'one\ntwo\nthree\n'

    assert extract_lines(content, 0, 2) == 'one\ntwo'
    assert extract_lines(content, -5, 1) == 'one'
