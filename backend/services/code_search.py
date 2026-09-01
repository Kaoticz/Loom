from __future__ import annotations

from collections.abc import Sequence

from database.models.code_embedding import CodeEmbedding
from database.models.embedding_model import EmbeddingModel
from database.models.file import File
from database.models.symbol import Symbol
from services.code_search_outcome import CodeSearchOutcome
from services.lexical_match import LexicalMatch
from services.lexical_match_mode import LexicalMatchMode
from services.projects import normalise_project_path
from services.symbol_definition import SymbolDefinition
from tortoise.expressions import Q
from tortoise_embeddings.vector_distance import VectorDistance

DEFAULT_LEXICAL_LIMIT: int = 10
DEFAULT_CONTEXT_LINES: int = 2

MAX_LEXICAL_KEYWORDS: int = 10
"""
How many keywords one lexical search may look for.

`raw_content` carries no index, so a keyword search is a `LIKE '%...%'` over
every indexed file. In *any* mode the keywords are queried separately, which
means the cost of a call scales with how many are given: ten keywords is ten
sequential scans of the table. The cap keeps a single call from turning into an
arbitrarily expensive one, and ten is well past what a search worth making needs.

Keywords beyond the cap are reported rather than dropped in silence, so the
agent can tell that part of what it asked for was not searched for.
"""


DEFAULT_LEXICAL_MODE: LexicalMatchMode = LexicalMatchMode.any_of


def normalise_keywords(keywords: str | Sequence[str]) -> list[str]:
    """
    Reduce what a caller supplied to the keywords actually worth searching for.

    A bare string counts as one keyword rather than a sequence of characters.
    That is not only a convenience: `Sequence[str]` is satisfied by `str`, so
    without this a single keyword would silently be searched for one letter at a
    time, matching almost every line in the repository.

    A single entry is always one literal, spaces included, so a phrase remains
    searchable. Splitting on whitespace would make that impossible, and the
    caller can pass several entries when it wants several keywords.

    Blank entries are dropped and duplicates removed, compared without regard to
    case because the search itself ignores case: searching for both spellings
    would return every line twice.

    :param keywords: One keyword, or several.
    :returns: The usable keywords, in the order given, without duplicates.
    """
    supplied: list[str] = [keywords] if isinstance(keywords, str) else list(keywords)
    usable: list[str] = []
    seen: set[str] = set()

    for keyword in supplied:
        stripped: str = keyword.strip()

        if not stripped:
            continue

        folded: str = stripped.lower()

        if folded in seen:
            continue

        seen.add(folded)
        usable.append(stripped)

    return usable


def split_at_keyword_limit(keywords: Sequence[str]) -> tuple[list[str], list[str]]:
    """
    Divide normalised keywords into those that will be searched for and the rest.

    Returned as a pair rather than simply truncated so that the caller can say
    which keywords went unsearched. Quietly dropping them would leave the agent
    believing a term is absent from the repository when it was never looked for.

    :param keywords: The normalised keywords, in the order given.
    :returns: The keywords to search for, and those beyond the cap.
    """
    return list(keywords[:MAX_LEXICAL_KEYWORDS]), list(keywords[MAX_LEXICAL_KEYWORDS:])


def describe_ignored_keywords(ignored: Sequence[str]) -> str:
    """
    Explain that a call asked for more keywords than one search will take.

    :param ignored: The keywords that were not searched for.
    :returns: A sentence naming them, or an empty string when none were dropped.
    """
    if not ignored:
        return ''

    quoted: str = ', '.join(repr(keyword) for keyword in ignored)
    return (
        f'Only the first {MAX_LEXICAL_KEYWORDS} keywords were searched for; '
        f'{quoted} were not. Each keyword is scanned across every indexed file, '
        f'so search in batches of {MAX_LEXICAL_KEYWORDS} or fewer.'
    )


MAX_DISPLAYED_MATCH_PATHS: int = 10
"""
How many file paths a search observation lists before summarising the rest.

The point of showing paths rather than contents is to keep the feed readable, so
a search matching fifty files must not replace a wall of code with a wall of
paths. The count is always stated in full; only the listing is shortened.
"""


def describe_matched_files(
    subject: str,
    file_paths: Sequence[str],
    notices: Sequence[str] = ()
) -> str:
    """
    Describe a search by the files it matched, without any of their contents.

    The reader wants to know what a search turned up and whether it is worth
    looking at; the code itself is the agent's business and floods the feed.

    Notices come first. A search that half failed must not read as a complete
    one that simply found little, so the caveat has to arrive before the count
    rather than trailing after it.

    :param subject: What was searched for, already quoted for display.
    :param file_paths: The matched files, unique and in order.
    :param notices: Warnings about the search, such as a backend being down.
    :returns: A short block naming the files.
    """
    lines: list[str] = list(notices)
    total: int = len(file_paths)

    if total == 0:
        lines.append(f'{subject}. No files matched.')
        return '\n'.join(lines)

    shown: list[str] = list(file_paths[:MAX_DISPLAYED_MATCH_PATHS])
    remaining: int = total - len(shown)
    counted: str = '1 file' if total == 1 else f'{total} files'
    heading: str = (
        f'{subject}. Matched {counted}:'
        if remaining == 0
        else f'{subject}. Matched {counted}, showing the first {len(shown)}:'
    )

    lines.append(heading)
    lines.extend(shown)

    if remaining > 0:
        lines.append(f'...and {remaining} more.')

    return '\n'.join(lines)


def describe_no_lexical_matches(keywords: Sequence[str], mode: LexicalMatchMode) -> str:
    """
    Explain that a lexical search found nothing, naming what was tried.

    Naming the keywords matters because the agent may have supplied several, and
    a bare "nothing found" leaves it unable to tell a misspelling from a term
    that genuinely is not there.

    In *all* mode the suggestion to try *any* is worth making unprompted: related
    symbols usually sit on different lines, so requiring every keyword on one
    line finds nothing far more often than an agent expects.

    :param keywords: The keywords that were searched for, already normalised.
    :param mode: The mode the search ran in.
    :returns: A sentence, or two in the case worth explaining.
    """
    quoted: str = ', '.join(repr(keyword) for keyword in keywords)

    if mode is LexicalMatchMode.all_of:
        return (
            f'No single line contains all of {quoted}. '
            'Related code often sits on separate lines, so try mode="any" to find '
            'lines containing at least one of them.'
        )

    if len(keywords) == 1:
        return f'No occurrences of {quoted} were found.'

    return f'No occurrences of any of {quoted} were found.'


def extract_lines(content: str, line_start: int, line_end: int) -> str:
    """
    Take an inclusive, 1-based range of lines out of a file's text.

    The index records where a symbol was when the file was last read, and a file
    edited since may be shorter, so a range running past the end is ordinary
    rather than a fault; slicing yields what remains, or nothing.

    The start is floored at 1 because that case is not benign: a zero or negative
    start would index from the end of the list and silently return the wrong
    lines rather than none.

    :param content: The full text of the file.
    :param line_start: The 1-based first line to take.
    :param line_end: The 1-based last line to take, inclusive.
    :returns: The selected lines, or an empty string if the range selects none.
    """
    lines: list[str] = content.splitlines()
    first: int = max(line_start, 1)
    return '\n'.join(lines[first - 1 : line_end])


class CodeSearchService:
    """
    Service for hybrid code retrieval (semantic + AST + lexical).

    Bound to one project. Every read scopes itself to it, because an unscoped
    read does not fail loudly — it quietly returns another project's code, which
    is worse than the collision this scoping exists to prevent. Holding the
    project on the service rather than passing it per call means a new read path
    cannot forget to ask for it.
    """

    repo_path: str

    def __init__(self, repo_path: str) -> None:
        """
        Bind the service to the project it may read.

        :param repo_path: Path to the repository, in any spelling.
        """
        self.repo_path = repo_path

    @property
    def project(self) -> str:
        """
        The project this service reads from.

        Derived on each use rather than settled in the constructor, because a
        session may repoint the service at another repository partway through.
        Caching it left the service reading one project while the indexer wrote
        to another, and a mismatch of that kind does not raise — searches simply
        come back empty, which reads as an unindexed repository.

        :returns: The normalised path identifying the current project.
        """
        return normalise_project_path(self.repo_path)

    async def search_code_semantic(
        self,
        query_embedding: list[float],
        model_name: str,
        limit: int = 5
    ) -> list[CodeEmbedding]:
        """
        Perform a semantic search using pgvector, scoped to one embedding model.

        Restricting the search to a single model guarantees every candidate
        vector has the same width as the query. Embeddings left behind by a
        previously configured model are simply not considered, rather than
        breaking the comparison.

        :param query_embedding: The vector embedding of the search query.
        :param model_name: Name of the embedding model that produced the query vector.
        :param limit: Maximum number of results to return.
        :returns: List of CodeEmbedding instances, nearest first.
        """
        model: EmbeddingModel | None = await EmbeddingModel.get_or_none(model_name=model_name)
        if model is None:
            return []

        # VectorDistance builds the operator through the query builder, so the
        # embedding is bound as a parameter rather than interpolated into SQL.
        results: list[CodeEmbedding] = await CodeEmbedding.filter(
            model=model, file__project=self.project
        ).annotate(
            distance=VectorDistance('embedding', '<->', query_embedding)
        ).order_by('distance').limit(limit).prefetch_related('file')
        return results

    async def _candidate_files(
        self,
        keywords: list[str],
        mode: LexicalMatchMode,
        limit: int
    ) -> list[File]:
        """
        Fetch the files worth scanning line by line.

        In *all* mode a line can only contain every keyword if its file does, so
        one query requiring all of them is both correct and cheap.

        In *any* mode the keywords are fetched separately and merged. A single
        query with an `OR` would be fewer round trips, but the row limit would
        then be spent on whichever keyword happened to match most, and the other
        keywords could go unrepresented entirely — which is the failure this
        whole capability exists to fix.

        :param keywords: The keywords to search for, already normalised.
        :param mode: Whether a line must contain any keyword or all of them.
        :param limit: Maximum number of matching lines the caller will keep.
        :returns: Candidate files, without duplicates, in a stable order.
        """
        def matches(keyword: str) -> Q:
            return Q(file_path__icontains=keyword) | Q(raw_content__icontains=keyword)

        if mode is LexicalMatchMode.all_of:
            required: Q = Q(*[matches(keyword) for keyword in keywords], join_type='AND')
            return await File.filter(required, project=self.project).limit(limit)

        merged: list[File] = []
        seen: set[int] = set()

        for keyword in keywords:
            for file_obj in await File.filter(
                matches(keyword), project=self.project
            ).limit(limit):
                # Keyed on the row, not the path. `file_path` is only unique
                # within a project, so two projects both holding `src/main.py`
                # are two different files that share a key — deduplicating on
                # the path silently discards one of them.
                if file_obj.id in seen:
                    continue
                seen.add(file_obj.id)
                merged.append(file_obj)

        return merged

    async def search_code_lexical(
        self,
        keywords: str | Sequence[str],
        limit: int = DEFAULT_LEXICAL_LIMIT,
        context_lines: int = DEFAULT_CONTEXT_LINES,
        mode: LexicalMatchMode = DEFAULT_LEXICAL_MODE
    ) -> list[LexicalMatch]:
        """
        Perform a case-insensitive text search across indexed file paths and contents.

        Files are matched in the database, then scanned line by line so that each
        result carries the line number and surrounding context the agent needs to
        locate the code.

        Several keywords may be given. In *any* mode a line is returned when it
        contains at least one of them; in *all* mode only when it contains every
        one. A line matching more than one keyword is returned once, not once per
        keyword.

        The limit applies to the merged result rather than to each keyword, so it
        caps what the caller actually receives.

        :param keywords: One keyword, or several. A single entry is one literal.
        :param limit: Maximum number of matching lines to return in total.
        :param context_lines: How many lines of context to include either side of a match.
        :param mode: Whether a line must contain any keyword or all of them.
        :returns: The matching lines, capped at the requested limit.
        """
        # Capped here as well as at the tool, because this is the public entry
        # point and each keyword costs a scan of every indexed file. The tool
        # tells the agent which keywords went unsearched; a caller reaching this
        # directly at least cannot make the call arbitrarily expensive.
        usable: list[str]
        usable, _ = split_at_keyword_limit(normalise_keywords(keywords))

        if not usable:
            return []
        if limit <= 0:
            return []

        needles: list[str] = [keyword.lower() for keyword in usable]
        required_in_line = all if mode is LexicalMatchMode.all_of else any
        candidates: list[File] = await self._candidate_files(usable, mode, limit)
        matches: list[LexicalMatch] = []

        for file_obj in candidates:
            lines: list[str] = file_obj.raw_content.splitlines()

            for index, line in enumerate(lines):
                lowered: str = line.lower()

                if not required_in_line(needle in lowered for needle in needles):
                    continue

                start: int = max(0, index - context_lines)
                end: int = min(len(lines), index + context_lines + 1)
                snippet: str = '\n'.join(
                    f'{number + 1}: {lines[number]}' for number in range(start, end)
                )
                matches.append(
                    LexicalMatch(
                        file_path=file_obj.file_path,
                        line_number=index + 1,
                        snippet=snippet
                    )
                )

                if len(matches) >= limit:
                    return matches

            # A file can match on its path alone, with nothing matching inside it.
            if not any(match.file_path == file_obj.file_path for match in matches):
                path: str = file_obj.file_path.lower()

                if required_in_line(needle in path for needle in needles):
                    matches.append(
                        LexicalMatch(
                            file_path=file_obj.file_path,
                            line_number=1,
                            snippet='(matched on file path)'
                        )
                    )
                    if len(matches) >= limit:
                        return matches

        return matches

    async def get_symbol_definition(self, symbol_name: str) -> list[SymbolDefinition]:
        """
        Look up a symbol's declarations, with the source that defines each.

        Returns rendered definitions rather than rows so the database models stay
        inside this service. Previously the rows themselves were handed to the
        agent, which received their default representation and so learned nothing
        but their primary keys.

        :param symbol_name: The name of the symbol (function, class) to find.
        :returns: One definition per declaration found, in file order.
        """
        rows: list[Symbol] = await Symbol.filter(
            symbol_name=symbol_name, file__project=self.project
        ).prefetch_related('file')

        return [
            SymbolDefinition(
                symbol_name=row.symbol_name,
                kind=row.kind,
                file_path=row.file.file_path,
                line_start=row.line_start,
                line_end=row.line_end,
                source=extract_lines(row.file.raw_content, row.line_start, row.line_end),
            )
            for row in rows
        ]
