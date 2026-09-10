from __future__ import annotations

import pytest

from database.models.file import File
from database.models.symbol import Symbol
from services.code_search import (
    MAX_DISPLAYED_MATCH_PATHS,
    MAX_LEXICAL_KEYWORDS,
    CodeSearchService,
    describe_ignored_keywords,
    describe_matched_files,
    describe_no_lexical_matches,
    extract_lines,
    normalise_keywords,
    split_at_keyword_limit,
)
from services.lexical_match import LexicalMatch
from services.lexical_match_mode import LexicalMatchMode
from services.symbol_definition import SymbolDefinition
from services.projects import normalise_project_path
from tests.conftest import TEST_PROJECT, TEST_REPO_PATH

SAMPLE: str = '''import os
from typing import Any


class UserRepository:
    """Fetches users."""

    def find_user(self, user_id: int) -> Any:
        return self._query(user_id)
'''

OTHER: str = 'def unrelated_helper(x):\n    return x * 2\n'


@pytest.fixture
async def indexed_files(db: None) -> None:
    """
    Insert two indexed files for lexical search to operate on.

    :param db: The database fixture this depends on.
    """
    await File.create(project=TEST_PROJECT, file_path='repo/users.py', raw_content=SAMPLE, hash='h1')
    await File.create(project=TEST_PROJECT, file_path='repo/helpers.py', raw_content=OTHER, hash='h2')


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


@pytest.mark.usefixtures('indexed_files')
async def test_any_mode_merges_matches_from_several_keywords(
    search: CodeSearchService,
) -> None:
    """
    Keywords living in different files are all represented in one result.

    This is the reported failure: a single joined string found nothing, because
    no one line held every term.
    """
    hits: list[LexicalMatch] = await search.search_code_lexical(
        ['find_user', 'unrelated_helper']
    )

    paths: set[str] = {hit.file_path for hit in hits}
    assert paths == {'repo/users.py', 'repo/helpers.py'}


@pytest.mark.usefixtures('indexed_files')
async def test_all_mode_requires_every_keyword_on_the_same_line(
    search: CodeSearchService,
) -> None:
    """
    *All* excludes a line holding only some of the keywords.
    """
    together: list[LexicalMatch] = await search.search_code_lexical(
        ['def', 'find_user'], mode=LexicalMatchMode.all_of
    )
    assert [hit.line_number for hit in together] == [8]

    # Both keywords are in this one file, but never on the same line: 'class' is
    # on line 5 and 'find_user' on line 8. The file therefore survives the
    # candidate query, so this exercises the line-level test rather than the
    # database filter — without it, an *all* that quietly behaved like *any*
    # would still pass, because no candidate file would have been returned.
    same_file: list[LexicalMatch] = await search.search_code_lexical(
        ['class', 'find_user'], mode=LexicalMatchMode.all_of
    )
    assert same_file == []

    # Keywords in different files cannot share a line either, and here it is the
    # candidate query that rules them out.
    different_files: list[LexicalMatch] = await search.search_code_lexical(
        ['find_user', 'unrelated_helper'], mode=LexicalMatchMode.all_of
    )
    assert different_files == []


@pytest.mark.usefixtures('indexed_files')
async def test_a_line_matching_several_keywords_is_returned_once(
    search: CodeSearchService,
) -> None:
    """
    Overlapping keywords must not double the results.
    """
    hits: list[LexicalMatch] = await search.search_code_lexical(['find_user', 'user_id'])
    located: list[tuple[str, int]] = [(hit.file_path, hit.line_number) for hit in hits]

    assert len(located) == len(set(located))


@pytest.mark.parametrize(
    'supplied,expected',
    [
        (['find_user', 'find_user'], ['find_user']),
        (['find_user', 'FIND_USER'], ['find_user']),
        (['find_user', ' find_user '], ['find_user']),
        (['beta', 'alpha', 'beta'], ['beta', 'alpha']),
    ],
    ids=['exact', 'differing_case', 'surrounding_space', 'order_preserved'],
)
def test_duplicate_keywords_are_reduced_to_one(
    supplied: list[str], expected: list[str]
) -> None:
    """
    A repeated keyword, in any casing, is searched for once.

    Asserted on the normalisation directly rather than through the search,
    because the search happens to be insensitive to duplicates anyway: a line is
    tested with `any`, so it is returned once however many keywords it matched.
    A test that went through the search would therefore pass whether or not
    duplicates were removed, and protect nothing.

    What deduplicating does change is real: one database query per keyword in
    *any* mode rather than several identical ones, and a failure message that
    does not list the same keyword twice.
    """
    assert normalise_keywords(supplied) == expected


@pytest.mark.usefixtures('indexed_files')
async def test_duplicates_do_not_reach_the_failure_message() -> None:
    """
    The keywords named back to the agent are the ones actually searched for.
    """
    message: str = describe_no_lexical_matches(
        normalise_keywords(['absent', 'ABSENT']), LexicalMatchMode.any_of
    )

    assert message.count('absent') == 1


@pytest.mark.usefixtures('indexed_files')
async def test_a_single_entry_is_one_literal_phrase(search: CodeSearchService) -> None:
    """
    An entry containing spaces is a phrase, not several keywords.

    Splitting it would make searching for a phrase impossible, and the caller
    can always pass separate entries when it wants separate terms.
    """
    phrase: list[LexicalMatch] = await search.search_code_lexical(['def find_user'])
    assert [hit.line_number for hit in phrase] == [8]

    assert await search.search_code_lexical(['find_user def']) == []


@pytest.mark.usefixtures('indexed_files')
@pytest.mark.parametrize(
    'keywords',
    [[], ['   '], ['', '   ']],
    ids=['empty_list', 'one_blank', 'all_blank'],
)
async def test_nothing_usable_returns_nothing(
    search: CodeSearchService, keywords: list[str]
) -> None:
    """
    No usable keyword must never degrade into matching everything.
    """
    assert await search.search_code_lexical(keywords) == []


@pytest.mark.usefixtures('indexed_files')
async def test_the_limit_applies_across_the_merged_result(
    search: CodeSearchService,
) -> None:
    """
    The cap counts total lines returned, not lines per keyword.

    A per-keyword limit would let a two-keyword call return twice what the
    caller asked for, so the number would not mean what it appears to.
    """
    for mode in (LexicalMatchMode.any_of, LexicalMatchMode.all_of):
        hits: list[LexicalMatch] = await search.search_code_lexical(
            ['def', 'self'], limit=1, mode=mode
        )
        assert len(hits) <= 1


def test_normalise_keywords_rejects_a_bare_string_being_split() -> None:
    """
    A single string is one keyword, not one keyword per character.

    `Sequence[str]` is satisfied by `str`, so without this the service would
    quietly search for each letter and match virtually every line indexed.
    """
    assert normalise_keywords('find_user') == ['find_user']


@pytest.mark.parametrize(
    'mode,expected',
    [
        (LexicalMatchMode.any_of, 'No occurrences of any of'),
        (LexicalMatchMode.all_of, 'No single line contains all of'),
    ],
    ids=['any', 'all'],
)
def test_a_fruitless_search_names_the_keywords(
    mode: LexicalMatchMode, expected: str
) -> None:
    """
    The message says what was tried, so a typo is distinguishable from absence.
    """
    message: str = describe_no_lexical_matches(['alpha', 'beta'], mode)

    assert expected in message
    assert "'alpha'" in message
    assert "'beta'" in message


def test_all_mode_failure_suggests_the_other_mode() -> None:
    """
    *All* finding nothing is common enough to be worth explaining unprompted.
    """
    message: str = describe_no_lexical_matches(['alpha', 'beta'], LexicalMatchMode.all_of)

    assert 'mode="any"' in message


@pytest.mark.usefixtures('indexed_files')
async def test_only_the_first_ten_keywords_are_searched_for(
    search: CodeSearchService,
) -> None:
    """
    A call asking for more keywords than the cap searches for the first ten.

    Each keyword costs a scan of every indexed file, so an unbounded list makes
    a single call arbitrarily expensive.
    """
    absent: list[str] = [f'absent_{index}' for index in range(MAX_LEXICAL_KEYWORDS)]

    # The real term sits beyond the cap, so it must not be found. Were the cap
    # absent it would match, which is what makes this assertion able to fail.
    beyond: list[LexicalMatch] = await search.search_code_lexical([*absent, 'find_user'])
    assert beyond == []

    # And the first ten are genuinely searched, not merely truncated away.
    within: list[LexicalMatch] = await search.search_code_lexical(
        ['find_user', *absent[:-1]]
    )
    assert [hit.line_number for hit in within] == [8]


@pytest.mark.parametrize(
    'count,expected_searched,expected_ignored',
    [
        (1, 1, 0),
        (MAX_LEXICAL_KEYWORDS, MAX_LEXICAL_KEYWORDS, 0),
        (MAX_LEXICAL_KEYWORDS + 3, MAX_LEXICAL_KEYWORDS, 3),
    ],
    ids=['one', 'exactly_at_the_cap', 'over_the_cap'],
)
def test_keywords_are_partitioned_at_the_cap(
    count: int, expected_searched: int, expected_ignored: int
) -> None:
    """
    The split reports what was left out rather than discarding it silently.
    """
    keywords: list[str] = [f'kw_{index}' for index in range(count)]
    searched, ignored = split_at_keyword_limit(keywords)

    assert len(searched) == expected_searched
    assert len(ignored) == expected_ignored
    # Nothing is lost or reordered between the two halves.
    assert [*searched, *ignored] == keywords


def test_ignored_keywords_are_named_and_explained() -> None:
    """
    The agent is told which terms went unsearched, and why there is a cap.
    """
    message: str = describe_ignored_keywords(['alpha', 'beta'])

    assert "'alpha'" in message
    assert "'beta'" in message
    assert str(MAX_LEXICAL_KEYWORDS) in message


def test_nothing_is_said_when_no_keywords_were_dropped() -> None:
    """
    A call within the cap carries no notice at all.
    """
    assert describe_ignored_keywords([]) == ''


def test_a_large_number_of_matched_files_is_summarised() -> None:
    """
    Replacing a wall of code with a wall of paths would miss the point.

    The count is stated in full even though the listing is shortened, so the
    reader can tell a broad search from a narrow one.
    """
    paths: list[str] = [f'repo/file_{index}.py' for index in range(25)]

    display: str = describe_matched_files('Searched for x', paths)

    assert 'Matched 25 files' in display
    assert display.count('repo/file_') == MAX_DISPLAYED_MATCH_PATHS
    assert f'...and {25 - MAX_DISPLAYED_MATCH_PATHS} more.' in display


def test_a_short_result_is_listed_in_full() -> None:
    """
    Below the cap there is no truncation and nothing said about it.
    """
    display: str = describe_matched_files('Searched for x', ['a.py', 'b.py'])

    assert 'Matched 2 files' in display
    assert 'more.' not in display
    assert 'showing the first' not in display


def test_one_match_reads_as_one_file() -> None:
    """
    'Matched 1 files' would be sloppy in something a person reads constantly.
    """
    assert 'Matched 1 file:' in describe_matched_files('Searched for x', ['a.py'])


def test_no_matches_reads_as_a_sentence() -> None:
    """
    An empty result must say so rather than render an empty list.
    """
    display: str = describe_matched_files('Searched for x', [])

    assert display == 'Searched for x. No files matched.'


def test_notices_come_before_the_count() -> None:
    """
    A caveat after the count reads as a footnote to a complete search.
    """
    display: str = describe_matched_files(
        'Searched for x', ['a.py'], ['Semantic search is unavailable.']
    )

    assert display.index('unavailable') < display.index('Matched')


def test_a_failed_search_with_no_matches_still_carries_its_notice() -> None:
    """
    The half that failed is the likeliest reason the other half found nothing.
    """
    display: str = describe_matched_files('Searched for x', [], ['Semantic search is unavailable.'])

    assert 'unavailable' in display
    assert 'No files matched' in display


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


@pytest.mark.usefixtures('db')
async def test_symbol_lookup_returns_the_definition(search: CodeSearchService) -> None:
    """
    A known symbol is found with its file and line range.
    """
    from database.models.symbol import Symbol

    file_obj = await File.create(project=TEST_PROJECT, file_path='repo/users.py', raw_content=SAMPLE, hash='h1')
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


@pytest.mark.usefixtures('db')
async def test_unknown_symbol_returns_nothing(search: CodeSearchService) -> None:
    """
    An absent symbol yields an empty result rather than raising.
    """
    assert await search.get_symbol_definition('does_not_exist') == []


@pytest.mark.usefixtures('db')
async def test_same_name_in_several_files_returns_all(search: CodeSearchService) -> None:
    """
    A name defined in more than one place returns every definition.

    Picking one arbitrarily would send the agent to the wrong file.
    """
    from database.models.symbol import Symbol

    for path in ('repo/a.py', 'repo/b.py'):
        file_obj = await File.create(project=TEST_PROJECT, file_path=path, raw_content='x', hash=path)
        await Symbol.create(
            file=file_obj, symbol_name='handler', kind='function', line_start=1, line_end=2
        )

    found = await search.get_symbol_definition('handler')

    assert len(found) == 2
    assert {found_symbol.file_path for found_symbol in found} == {'repo/a.py', 'repo/b.py'}


@pytest.mark.usefixtures('db')
async def test_symbol_lookup_is_exact_not_partial(search: CodeSearchService) -> None:
    """
    A lookup matches the whole name, not a fragment of it.
    """
    from database.models.symbol import Symbol

    file_obj = await File.create(project=TEST_PROJECT, file_path='repo/a.py', raw_content='x', hash='h')
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


@pytest.mark.usefixtures('db')
async def test_a_looked_up_definition_carries_its_source(search: CodeSearchService) -> None:
    """
    The source is read from the indexed content, not from disk.
    """
    file_obj = await File.create(
        project=TEST_PROJECT,
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


OTHER_REPO_PATH: str = '/tmp/loom-other-project'
"""A second project, to prove reads never cross between them."""

OTHER_PROJECT: str = normalise_project_path(OTHER_REPO_PATH)


@pytest.fixture
async def two_projects(db: None) -> None:
    """
    Index the same relative path under two projects, with distinct contents.

    :param db: The database fixture this depends on.
    """
    await File.create(
        project=TEST_PROJECT,
        file_path='src/main.py',
        raw_content='def shared_name():\n    return "MINE"\n',
        hash='h1',
    )
    await File.create(
        project=OTHER_PROJECT,
        file_path='src/main.py',
        raw_content='def shared_name():\n    return "THEIRS"\n',
        hash='h2',
    )


@pytest.mark.usefixtures('two_projects')
async def test_lexical_any_mode_never_returns_another_project(
    search: CodeSearchService,
) -> None:
    """
    Scoped per read path, because one unscoped path leaks in silence.
    """
    hits: list[LexicalMatch] = await search.search_code_lexical(['shared_name'])

    assert len(hits) == 1
    assert 'MINE' in await content_of(hits[0])


@pytest.mark.usefixtures('two_projects')
async def test_lexical_all_mode_never_returns_another_project(
    search: CodeSearchService,
) -> None:
    """
    The two modes take different query paths, so both are checked.

    Scoping only the `any` branch would leave `all` leaking, and no test of the
    other mode would notice.
    """
    hits: list[LexicalMatch] = await search.search_code_lexical(
        ['def', 'shared_name'], mode=LexicalMatchMode.all_of
    )

    assert len(hits) == 1
    assert 'MINE' in await content_of(hits[0])


@pytest.mark.usefixtures('two_projects')
async def test_symbol_lookup_never_returns_another_project(
    search: CodeSearchService,
) -> None:
    """
    Symbols reach their project through their file, so the join must be scoped.
    """
    await Symbol.create(
        file=await File.get(project=TEST_PROJECT, file_path='src/main.py'),
        symbol_name='shared_name', kind='function', line_start=1, line_end=2,
    )
    await Symbol.create(
        file=await File.get(project=OTHER_PROJECT, file_path='src/main.py'),
        symbol_name='shared_name', kind='function', line_start=1, line_end=2,
    )

    found: list[SymbolDefinition] = await search.get_symbol_definition('shared_name')

    assert len(found) == 1
    assert 'MINE' in found[0].source


async def content_of(hit: LexicalMatch) -> str:
    """
    Read back the content of the file a hit came from.

    A hit carries a repository-relative path, which both projects share, so the
    row has to be fetched by project as well to tell them apart.

    :param hit: The match to resolve.
    :returns: The raw content of the matching file in the project under test.
    """
    row: File = await File.get(project=TEST_PROJECT, file_path=hit.file_path)
    return row.raw_content


@pytest.mark.usefixtures('two_projects')
async def test_a_path_spelled_differently_is_the_same_project(
    db: None,
) -> None:
    """
    Trailing separators and relative spellings must not fork a project in two.

    Indexing under one spelling and searching under another would return
    nothing, which reads exactly like an empty index rather than a bug.
    """
    for spelling in (TEST_REPO_PATH + '/', TEST_REPO_PATH + '/.', TEST_REPO_PATH):
        service: CodeSearchService = CodeSearchService(repo_path=spelling)
        hits: list[LexicalMatch] = await service.search_code_lexical(['shared_name'])

        assert len(hits) == 1, f'{spelling!r} should be the same project'


async def test_two_projects_sharing_a_path_are_two_candidate_files(
    two_projects: None,
) -> None:
    """
    Candidate files are deduplicated by row, not by path.

    `file_path` stopped being unique the moment it became unique only within a
    project, so keying on it collapses two different files into one. That is
    wrong on its own terms, and it is quietly dangerous: while it held, an
    unscoped search returned one result instead of two and so looked correctly
    scoped. The mutation that removed the scoping went undetected because of it.
    """
    service: CodeSearchService = CodeSearchService(repo_path=TEST_REPO_PATH)
    candidates: list[File] = await service._candidate_files(
        ['shared_name'], LexicalMatchMode.any_of, limit=10
    )

    # Scoped, so only this project's row comes back.
    assert [row.project for row in candidates] == [TEST_PROJECT]

    unscoped: list[File] = await File.filter(file_path='src/main.py')
    assert len({row.id for row in unscoped}) == 2, 'both rows exist to be confused'
    assert len({row.file_path for row in unscoped}) == 1, 'and they share a path'
