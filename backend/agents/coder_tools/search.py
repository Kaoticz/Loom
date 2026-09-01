from __future__ import annotations

import os
from logging import Logger, getLogger
from typing import Any

from httpx import AsyncClient, HTTPError, Response
from pydantic_ai import RunContext

from services.code_search import (
    DEFAULT_LEXICAL_MODE,
    describe_ignored_keywords,
    describe_matched_files,
    describe_no_lexical_matches,
    normalise_keywords,
    split_at_keyword_limit,
)
from services.code_search_outcome import CodeSearchOutcome
from services.lexical_match import LexicalMatch
from services.lexical_match_mode import LexicalMatchMode
from services.symbol_definition import SymbolDefinition
from services.tool_reporting import describe_symbol_lookup, report_tool

from .deps import AgentDeps

logger: Logger = getLogger('agents.coder')

WEB_SEARCH_RESULT_LIMIT: int = 5
WEB_SEARCH_SNIPPET_LENGTH: int = 200


async def search_code(ctx: RunContext[AgentDeps], query: str) -> str:
    """
    Search the codebase for code related to a natural language query.

    Runs a semantic search over the indexed embeddings and a literal text search
    in parallel, then merges the two, so that both conceptual matches and exact
    keyword matches are covered.
    :param query: What you are looking for, in plain words, for example "where retries are configured".
    """
    async with report_tool(ctx.deps.events, 'search_code', {'query': query}) as report:
        outcome: CodeSearchOutcome = await _search_code(ctx, query)

        # The agent gets the code; the reader gets the files it came from. The
        # snippets are what the agent asked for and what floods the feed.
        return report.succeed(
            outcome.rendered,
            observation=describe_matched_files(
                f'Searched for {query!r}', outcome.file_paths, outcome.notices
            ),
        )


async def _search_code(ctx: RunContext[AgentDeps], query: str) -> CodeSearchOutcome:
    """
    Run the merged semantic and lexical search.

    :param ctx: The agent run context.
    :param query: The natural language query.
    :returns: The rendered results, the files they came from, and any warnings.
    """
    if not query.strip():
        # Embedding an empty string raises, and there is nothing meaningful to
        # search for anyway. Refuse with an explanation rather than let the
        # exception abort the whole session.
        return CodeSearchOutcome(
            rendered='No query was given. Pass what you are looking for in plain '
            'words, for example "where retries are configured".',
            file_paths=[],
            notices=[],
        )

    embedding: list[float] = await ctx.deps.embedding.embed(query)

    sections: list[str] = []
    seen: set[str] = set()
    notices: list[str] = []
    ordered_paths: list[str] = []

    semantic: list[Any] = []
    try:
        semantic = await ctx.deps.search.search_code_semantic(
            embedding, ctx.deps.embedding.embedding_model
        )
    except Exception as error:
        # A vector backend failure must not cost us the lexical results, but it
        # is reported rather than swallowed so a broken index cannot go unnoticed.
        logger.warning('Semantic search failed for query %r: %s', query, error)
        notices.append(f'Semantic search is unavailable ({type(error).__name__}); showing text matches only.')

    for chunk in semantic:
        path: str = str(chunk.file.file_path)
        if path in seen:
            continue
        seen.add(path)
        ordered_paths.append(path)
        sections.append(f'{path}:{chunk.start_line} (semantic)\n{chunk.chunk_text}')

    lexical: list[LexicalMatch] = await ctx.deps.search.search_code_lexical(query)
    for match in lexical:
        if match.file_path in seen:
            continue
        seen.add(match.file_path)
        ordered_paths.append(match.file_path)
        sections.append(f'{match.render()} (lexical)')

    if not sections:
        # Kept out of `notices`: the observation says as much in its own words,
        # and repeating it there would read as two separate complaints.
        empty: str = f'No code found for {query!r}. The repository may not be indexed yet.'
        return CodeSearchOutcome(
            rendered='\n'.join([*notices, empty]),
            file_paths=[],
            notices=notices,
        )

    body: str = '\n\n'.join(sections)
    return CodeSearchOutcome(
        rendered='\n\n'.join([*notices, body]) if notices else body,
        file_paths=ordered_paths,
        notices=notices,
    )


async def search_code_lexical(
    ctx: RunContext[AgentDeps],
    keywords: list[str],
    limit: int = 10,
    mode: LexicalMatchMode = DEFAULT_LEXICAL_MODE
) -> str:
    """
    Search the codebase for exact, case-insensitive text.

    Use this when you know the literal strings you are looking for, such as
    function names, an error message, or a configuration key.

    Pass as many keywords as you like in one call. Each entry is one literal,
    spaces included, so use a single entry to search for a phrase such as
    "connection refused", and separate entries to search for separate terms.

    `mode` decides how they combine, and defaults to "any":

    - "any" returns lines containing at least one of the keywords. Use it to
      look for several unrelated things at once, such as three function names.
    - "all" returns only lines containing every keyword. Use it to narrow down,
      remembering that related symbols usually sit on different lines, so this
      often finds nothing even when both keywords are present in the file.

    `limit` caps the total number of lines returned across all keywords, and at
    most 10 keywords are searched for in one call.
    :param keywords: The exact identifiers or literals to look for, as a list of separate strings.
    :param limit: How many results to return at most.
    :param mode: Whether a result must contain all of the keywords or any of them.
    """
    searched: list[str]
    ignored: list[str]
    searched, ignored = split_at_keyword_limit(normalise_keywords(keywords))
    arguments: dict[str, object] = {'keywords': searched, 'limit': limit, 'mode': mode.value}

    async with report_tool(ctx.deps.events, 'search_code_lexical', arguments) as report:
        if not searched:
            return report.succeed(
                'No keywords were given. Pass one or more non-empty strings, '
                'for example ["parse_plan", "render_plan"].'
            )

        matches: list[LexicalMatch] = await ctx.deps.search.search_code_lexical(
            searched, limit=limit, mode=mode
        )
        body: str = (
            '\n\n'.join(match.render() for match in matches)
            if matches
            else describe_no_lexical_matches(searched, mode)
        )

        # Prepended rather than appended: the agent needs to know part of its
        # request went unserved before it draws conclusions from the results.
        notice: str = describe_ignored_keywords(ignored)
        result: str = f'{notice}\n\n{body}' if notice else body

        # Unique, in order: a file with nine matching lines is one file to the
        # reader, however many times it appears in what the agent receives.
        matched_paths: list[str] = list(dict.fromkeys(match.file_path for match in matches))
        subject: str = 'Searched for ' + ', '.join(repr(keyword) for keyword in searched)

        return report.succeed(
            result,
            observation=describe_matched_files(
                subject, matched_paths, [notice] if notice else []
            ),
        )


async def get_symbol(ctx: RunContext[AgentDeps], symbol: str) -> str:
    """
    Get the definition of a symbol (class, function).
    :param symbol: The exact name of the function, class or method, without brackets or a module prefix.
    """
    async with report_tool(ctx.deps.events, 'get_symbol', {'symbol': symbol}) as report:
        if not symbol.strip():
            return report.fail('No symbol was given. Pass the exact name of the function, '
                               'class or method to look up, for example "find_user".')
        found: list[SymbolDefinition] = await ctx.deps.search.get_symbol_definition(symbol)

        if not found:
            return report.succeed(
                f'No definition of {symbol!r} is in the index. It may be spelled differently, '
                'declared in a file that was not indexed, or not exist.'
            )

        # The agent gets the definitions; the reader gets a line saying how many.
        return report.succeed(
            '\n\n'.join(definition.render() for definition in found),
            observation=describe_symbol_lookup(symbol, len(found)),
        )


async def web_search(ctx: RunContext[AgentDeps], query: str) -> str:
    """
    Search the web for information using SearXNG.

    Returns the top results as title, URL and a short snippet each.
    :param query: What to search the web for, in plain words.
    """
    async with report_tool(ctx.deps.events, 'web_search', {'query': query}) as report:
        if not query.strip():
            return report.fail('No query was given. Pass what you want to search the web '
                               'for, in plain words.')
        return report.succeed(await _web_search(query))


async def _web_search(query: str) -> str:
    """
    Query SearXNG and render the top results.

    :param query: The search query.
    :returns: The rendered results, or an explanation of why there are none.
    """
    searxng_url: str = os.getenv('SEARXNG_URL', 'http://searxng:8080')

    payload: dict[str, Any]
    try:
        async with AsyncClient(timeout=20.0) as client:
            response: Response = await client.get(
                f'{searxng_url}/search', params={'q': query, 'format': 'json'}
            )
            if response.status_code != 200:
                return f'Web search failed: SearXNG returned HTTP {response.status_code}.'
            payload = response.json()
    except HTTPError as error:
        return f'Web search failed: could not reach SearXNG at {searxng_url}. {error}'
    except ValueError:
        return 'Web search failed: SearXNG returned a response that is not valid JSON.'

    results: Any = payload.get('results')
    if not isinstance(results, list) or not results:
        return f'No web results were found for {query!r}.'

    sections: list[str] = []
    for entry in results[:WEB_SEARCH_RESULT_LIMIT]:
        if not isinstance(entry, dict):
            continue
        title: str = str(entry.get('title', 'Untitled')).strip()
        url: str = str(entry.get('url', '')).strip()
        snippet: str = ' '.join(str(entry.get('content', '')).split())[:WEB_SEARCH_SNIPPET_LENGTH]
        sections.append(f'{title}\n{url}\n{snippet}' if snippet else f'{title}\n{url}')

    if not sections:
        return f'No usable web results were found for {query!r}.'
    return '\n\n'.join(sections)
