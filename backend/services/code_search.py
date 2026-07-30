from __future__ import annotations

from typing import Annotated

from database.models.code_embedding import CodeEmbedding
from database.models.embedding_model import EmbeddingModel
from database.models.file import File
from database.models.symbol import Symbol
from pydantic import BaseModel, Field
from tortoise.expressions import Q
from tortoise_embeddings.vector_distance import VectorDistance

DEFAULT_LEXICAL_LIMIT: int = 10
DEFAULT_CONTEXT_LINES: int = 2


class LexicalMatch(BaseModel):
    """
    A single line matching a lexical search, with its surrounding context.
    """

    file_path: Annotated[str, Field(description='Repository-relative path of the matching file.')]
    line_number: Annotated[int, Field(ge=1, description='1-based line number of the matching line.')]
    snippet: Annotated[str, Field(description='The matching line with a few lines of surrounding context.')]

    def render(self) -> str:
        """
        Render this match as a readable block for the agent.

        :returns: The rendered match.
        """
        return f'{self.file_path}:{self.line_number}\n{self.snippet}'


class CodeSearchService:
    """
    Service for hybrid code retrieval (semantic + AST + lexical).
    """

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
        results: list[CodeEmbedding] = await CodeEmbedding.filter(model=model).annotate(
            distance=VectorDistance('embedding', '<->', query_embedding)
        ).order_by('distance').limit(limit).prefetch_related('file')
        return results

    async def search_code_lexical(
        self,
        keyword: str,
        limit: int = DEFAULT_LEXICAL_LIMIT,
        context_lines: int = DEFAULT_CONTEXT_LINES
    ) -> list[LexicalMatch]:
        """
        Perform a case-insensitive text search across indexed file paths and contents.

        Files are matched in the database, then scanned line by line so that each
        result carries the line number and surrounding context the agent needs to
        locate the code.

        :param keyword: The text to search for.
        :param limit: Maximum number of matching lines to return.
        :param context_lines: How many lines of context to include either side of a match.
        :returns: The matching lines, capped at the requested limit.
        """
        if not keyword.strip():
            return []
        if limit <= 0:
            return []

        candidates: list[File] = await File.filter(
            Q(file_path__icontains=keyword) | Q(raw_content__icontains=keyword)
        ).limit(limit)

        needle: str = keyword.lower()
        matches: list[LexicalMatch] = []

        for file_obj in candidates:
            lines: list[str] = file_obj.raw_content.splitlines()

            for index, line in enumerate(lines):
                if needle not in line.lower():
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
                if needle in file_obj.file_path.lower():
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

    async def get_symbol_definition(self, symbol_name: str) -> list[Symbol]:
        """
        Lookup a symbol definition in the symbols table.

        :param symbol_name: The name of the symbol (function, class) to find.
        :returns: List of Symbol instances.
        """
        return await Symbol.filter(symbol_name=symbol_name).prefetch_related('file')
