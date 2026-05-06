from typing import Any
from database.models.symbol import Symbol
from database.models.code_embedding import CodeEmbedding
from tortoise.expressions import RawSQL

class CodeSearchService:
    """
    Service for hybrid code retrieval (semantic + AST).
    """

    async def search_code_semantic(self, query_embedding: list[float], limit: int = 5) -> list[CodeEmbedding]:
        """
        Perform a semantic search using pgvector.

        :param query_embedding: The vector embedding of the search query.
        :param limit: Maximum number of results to return.
        :returns: List of CodeEmbedding instances.
        """
        # Format the embedding list as a pgvector string '[v1, v2, ...]'
        vector_str: str = '[' + ','.join(map(str, query_embedding)) + ']'
        
        # Using tortoise-vector's capabilities or raw SQL for distance
        results: list[CodeEmbedding] = await CodeEmbedding.annotate(
            distance=RawSQL(f"embedding <-> '{vector_str}'")
        ).order_by('distance').limit(limit)
        return results

    async def get_symbol_definition(self, symbol_name: str) -> list[Symbol]:
        """
        Lookup a symbol definition in the symbols table.

        :param symbol_name: The name of the symbol (function, class) to find.
        :returns: List of Symbol instances.
        """
        return await Symbol.filter(symbol_name=symbol_name).prefetch_related('file')
