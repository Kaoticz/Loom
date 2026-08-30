import os
import subprocess
import json
import hashlib
from typing import Any
from database.models.file import File
from database.models.symbol import Symbol
from database.models.code_embedding import CodeEmbedding
from database.models.embedding_model import EmbeddingModel
from services.embedding import (
    ChunkTooLargeForModel,
    EmbeddingError,
    EmbeddingInputTooLong,
    EmbeddingService,
)
from asyncio import Semaphore
from logging import Logger, getLogger
from collections.abc import Sequence
from services.symbol_patterns import build_rules, language_for
from services.chunking import Chunk, SymbolSpan, build_chunks
from services.indexing_progress import ProgressCallback, ProgressReporter
from services.repository_files import list_indexable_files
from services.projects import normalise_project_path
from subprocess import CompletedProcess
import asyncio

logger: Logger = getLogger(__name__)

MIN_CHUNK_LINES: int = 10
MAX_CHUNK_LINES: int = 1000
"""
Bounds on a hand-set chunk size.

Below the minimum a chunk holds too little to mean anything; above the maximum
no embedding model will take it. Enforced here as well as in the interface,
because the value arrives from the client.
"""

DEFAULT_MAX_FILE_SIZE_BYTES: int = 1024 * 1024


def max_file_size_bytes() -> int:
    """
    Read the largest file size that will be indexed.

    `LOOM_MAX_FILE_SIZE_BYTES` sets the limit. Zero or a negative value means no
    limit at all, for a user who would rather index a large generated file than
    have it skipped. An unusable value falls back to the default rather than
    stopping the session.

    :returns: The limit in bytes, or zero for unlimited.
    """
    raw: str | None = os.getenv('LOOM_MAX_FILE_SIZE_BYTES')
    if raw is None:
        return DEFAULT_MAX_FILE_SIZE_BYTES
    try:
        return int(raw)
    except ValueError:
        return DEFAULT_MAX_FILE_SIZE_BYTES


def clamp_chunk_lines(chunk_lines: int | None) -> int | None:
    """
    Hold a hand-set chunk size within usable bounds.

    A value of zero or less means automatic, matching how the opening message
    reader treats one: two places interpreting the same number differently is a
    bug waiting for someone to find it.

    :param chunk_lines: The requested size, or None for automatic sizing.
    :returns: The size to use, or None for automatic sizing.
    """
    if chunk_lines is None or chunk_lines <= 0:
        return None
    return max(MIN_CHUNK_LINES, min(chunk_lines, MAX_CHUNK_LINES))


class IndexingService:
    """
    Service for scanning the codebase and populating the database.
    """
    repo_path: str
    embedding: EmbeddingService
    chunk_lines: int | None

    def __init__(
        self,
        repo_path: str,
        embedding: EmbeddingService,
        chunk_lines: int | None = None,
    ) -> None:
        """
        Create an indexing service.

        :param repo_path: Path to the repository to index.
        :param embedding: Service used to generate chunk embeddings.
        :param chunk_lines: Lines per chunk, or None to size chunks automatically.
        """
        self.repo_path = repo_path
        self.embedding = embedding
        self.chunk_lines = clamp_chunk_lines(chunk_lines)

    @property
    def project(self) -> str:
        """
        The project this service writes to.

        Derived on each use rather than settled in the constructor: a session
        repoints `repo_path` when a start message names a repository, and a
        project cached before that would have every row written under the
        previous repository's name.

        :returns: The normalised path identifying the current project.
        """
        return normalise_project_path(self.repo_path)

    async def index_repository(self, progress_callback: ProgressCallback | None = None) -> None:
        """
        Scan the repository and update 'files', 'symbols', and 'embeddings' tables.

        :param progress_callback: Where to report progress, if anywhere.
        """
        abs_repo_path: str = os.path.abspath(self.repo_path)
        # What the repository ignores, it ignores here too. Embedding build
        # output and generated artefacts costs time and money and then buries
        # the code someone asked about underneath the things they never wrote.
        all_files: list[str] = list_indexable_files(abs_repo_path)

        reporter: ProgressReporter = ProgressReporter(len(all_files), progress_callback)
        await reporter.start(f'Indexing {len(all_files)} files...')

        await reporter.phase('Preparing the embedding model...')
        embedding_model: EmbeddingModel = await self.resolve_embedding_model()

        semaphore: Semaphore = Semaphore(10) # Process 10 files at a time

        tasks: list[Any] = []
        for file_path in all_files:
            rel_path: str = os.path.relpath(file_path, abs_repo_path)
            tasks.append(
                self.index_file(
                    rel_path,
                    embedding_model=embedding_model,
                    reporter=reporter,
                    semaphore=semaphore,
                )
            )

        await asyncio.gather(*tasks)
        await reporter.finish(f'Indexing complete: {reporter.completed} files processed.')

    async def index_file(
        self,
        rel_path: str,
        *,
        embedding_model: EmbeddingModel | None = None,
        reporter: ProgressReporter | None = None,
        semaphore: Semaphore | None = None,
    ) -> None:
        """
        Update the index for one file in this repository.

        A direct caller supplies only ``rel_path``. The optional arguments let a
        full repository scan reuse one already-resolved model and reporter rather
        than repeating that work for every file.

        :param rel_path: Path to index, relative to the repository root.
        :param embedding_model: An embedding model already resolved for this scan.
        :param reporter: Progress reporter shared by a full scan, if any.
        :param semaphore: Concurrency limiter shared by a full scan, if any.
        :returns: None.
        """
        abs_repo_path: str = os.path.abspath(self.repo_path)
        file_path: str = os.path.abspath(os.path.join(abs_repo_path, rel_path))
        if not file_path.startswith(abs_repo_path + os.sep):
            return
        if not os.path.isfile(file_path):
            if reporter is not None:
                await reporter.file_completed(rel_path)
            return

        active_reporter: ProgressReporter = reporter or ProgressReporter(1, None)
        active_semaphore: Semaphore = semaphore or Semaphore(1)
        await self._process_file(
            file_path,
            abs_repo_path,
            active_semaphore,
            embedding_model,
            active_reporter,
        )

    async def drop_file(self, rel_path: str) -> None:
        """
        Remove one repository file and its derived index rows.

        The database foreign keys cascade from ``File`` to both symbols and
        embeddings. Missing rows are intentionally harmless, because a newly
        created file may be removed before it is ever indexed.

        :param rel_path: Path to remove from the index, relative to the repository root.
        :returns: None.
        """
        abs_repo_path: str = os.path.abspath(self.repo_path)
        file_path: str = os.path.abspath(os.path.join(abs_repo_path, rel_path))
        if not file_path.startswith(abs_repo_path + os.sep):
            return

        safe_rel_path: str = os.path.relpath(file_path, abs_repo_path)
        await File.filter(project=self.project, file_path=safe_rel_path).delete()

    async def _process_file(
        self,
        file_path: str,
        abs_repo_path: str,
        semaphore: Semaphore,
        embedding_model: EmbeddingModel | None,
        reporter: ProgressReporter,
    ) -> None:
        """
        Index a single file, reporting progress as it goes.

        :param file_path: Absolute path to the file.
        :param abs_repo_path: Absolute path to the repository root.
        :param semaphore: Limits how many files are processed at once.
        :param embedding_model: The registry entry for the embedding model in use, if resolved.
        :param reporter: Where progress is reported.
        """
        async with semaphore:
            rel_path: str = os.path.relpath(file_path, abs_repo_path)
            
            content: str
            try:
                # Check if it is a binary file first
                # The whole file is held in memory and stored in a column, so an
                # oversized one is skipped rather than risking the process.
                limit: int = max_file_size_bytes()
                if limit > 0:
                    size: int = os.path.getsize(file_path)
                    if size > limit:
                        await reporter.phase(
                            f'Skipping {rel_path}: {size // 1024} KB exceeds the '
                            f'{limit // 1024} KB limit.'
                        )
                        await self.drop_file(rel_path)
                        await reporter.file_completed(rel_path)
                        return

                with open(file_path, 'rb') as f:
                    chunk: bytes = f.read(1024)
                    if b'\x00' in chunk:
                        await self.drop_file(rel_path)
                        await reporter.file_completed(rel_path)
                        return # Skip binary files

                with open(file_path, 'r', errors='ignore') as f:
                    content = f.read()
            except Exception:
                await reporter.file_completed(rel_path)
                return

            # Strip null bytes to be safe for PostgreSQL
            content = content.replace('\x00', '')

            if not content.strip():
                await self.drop_file(rel_path)
                await reporter.file_completed(rel_path)
                return

            file_hash: str = hashlib.sha256(content.encode()).hexdigest()

            existing: File | None = await File.get_or_none(
                project=self.project, file_path=rel_path
            )
            if existing is not None and existing.hash == file_hash:
                await reporter.file_completed(rel_path)
                return

            active_model: EmbeddingModel = (
                embedding_model
                if embedding_model is not None
                else await self.resolve_embedding_model()
            )
            
            file_obj: File
            created: bool
            file_obj, created = await File.get_or_create(
                project=self.project,
                file_path=rel_path,
                defaults={'raw_content': content, 'hash': file_hash}
            )
            
            needs_indexing: bool = created
            if not created and file_obj.hash != file_hash:
                file_obj.raw_content = content
                file_obj.hash = file_hash
                await file_obj.save()
                
                # Clean and re-index
                await Symbol.filter(file=file_obj).delete()
                await CodeEmbedding.filter(file=file_obj).delete()
                needs_indexing = True
            
            if needs_indexing:
                await reporter.phase(f'Extracting symbols from {rel_path}...')
                await self._index_symbols(file_obj, file_path)

                await reporter.phase(f'Generating embeddings for {rel_path}...')
                await self._index_embeddings(file_obj, active_model)

            await reporter.file_completed(rel_path)

    async def resolve_embedding_model(self) -> EmbeddingModel:
        """
        Look up the configured embedding model, registering it if it is new.

        The model's vector width is probed once and stored. If a previously
        registered model now reports a different width, its existing embeddings
        are discarded, because they can no longer be compared against anything
        the model produces.

        A change to the chunk size discards them for a weaker but sufficient
        reason: the vectors are still comparable, but they describe a chunking
        nobody asked for any more, so leaving them would make the setting appear
        to have no effect on anything already indexed.

        :returns: The registry entry for the configured embedding model.
        :raises EmbeddingError: If the embedding backend is unreachable or misconfigured.
        """
        name: str = self.embedding.embedding_model
        dimensions: int = await self.embedding.detect_dimensions()

        recorded_chunking: int = self.chunk_lines or 0

        model: EmbeddingModel | None = await EmbeddingModel.get_or_none(model_name=name)
        if model is None:
            return await EmbeddingModel.create(
                model_name=name, dimensions=dimensions, chunk_lines=recorded_chunking
            )

        if model.dimensions != dimensions or model.chunk_lines != recorded_chunking:
            # Both are deliberately unscoped. The model's embeddings are
            # dropped for every project, so every project's files have to be
            # re-embedded; scoping the hash reset to the current project would
            # leave the others with no embeddings and unchanged hashes, so their
            # next index would skip them and they would stay silently empty.
            await CodeEmbedding.filter(model=model).delete()
            await File.all().update(hash='')
            model.dimensions = dimensions
            model.chunk_lines = recorded_chunking
            await model.save()

        return model

    async def _index_embeddings(self, file_obj: File, model: EmbeddingModel) -> None:
        """
        Generate chunks and embeddings for a file.

        Symbols must already have been extracted for this file, since the chunk
        boundaries are taken from them.

        :param file_obj: The file record whose contents should be embedded.
        :param model: The registry entry for the model producing the embeddings.
        :raises EmbeddingError: If the backend is unreachable or returns a wrongly sized vector.
        """
        content: str = file_obj.raw_content
        lines: list[str] = content.splitlines()

        # Chunks follow symbol boundaries where the language allows, so an
        # embedding covers a whole definition rather than an arbitrary window
        # holding the tail of one function and the head of the next.
        spans: list[SymbolSpan] = [
            SymbolSpan(start_line=symbol.line_start, end_line=symbol.line_end)
            for symbol in await Symbol.filter(file=file_obj)
            if symbol.line_end >= symbol.line_start
        ]
        chunks: list[Chunk] = (
            build_chunks(lines, spans, max_lines=self.chunk_lines)
            if self.chunk_lines is not None
            else build_chunks(lines, spans)
        )

        pending: list[Chunk] = list(chunks)
        while pending:
            chunk: Chunk = pending.pop(0)
            chunk_text: str = chunk.text(lines)

            if not chunk_text.strip():
                continue

            embedding: list[float]
            try:
                embedding = await self.embedding.embed(chunk_text)
            except EmbeddingInputTooLong as error:
                if self.chunk_lines is not None:
                    raise ChunkTooLargeForModel(
                        f'Chunks of {self.chunk_lines} lines from {file_obj.file_path} are too '
                        f'large for the embedding model {model.model_name!r}. Lower the chunk '
                        'size in Settings, or set it to Auto to have Loom size chunks itself.'
                    ) from error

                halves: list[Chunk] = chunk.halve()
                if len(halves) > 1:
                    # How many tokens a chunk costs depends on the model's
                    # tokenizer, which this side cannot measure, so the chunk is
                    # divided until the model accepts it.
                    pending[:0] = halves
                    continue

                # A single line the model will not take. Skipping it costs this
                # one line its semantic search; failing here would cost the
                # whole repository its index.
                logger.warning(
                    'Skipping line %d of %s: %r will not accept it at any length.',
                    chunk.start_line,
                    file_obj.file_path,
                    model.model_name,
                )
                continue

            if len(embedding) != model.dimensions:
                raise EmbeddingError(
                    f'Model {model.model_name!r} returned a {len(embedding)}-dimension vector '
                    f'but is registered as producing {model.dimensions} dimensions. '
                    'Storing it would make this chunk incomparable with the rest of the index.'
                )

            await CodeEmbedding.create(
                model=model,
                file=file_obj,
                chunk_text=chunk_text,
                embedding=embedding,
                start_line=chunk.start_line
            )

    async def _index_symbols(self, file_obj: File, file_path: str) -> None:
        """
        Use ast-grep to extract symbols from a file.

        The language is inferred by ast-grep from the file extension, so only the
        patterns are language specific. Files whose language has no registered
        patterns are skipped.

        :param file_obj: The file record the extracted symbols are attached to.
        :param file_path: Absolute path to the file on disk, which ast-grep reads directly.
        """
        language: str | None = language_for(file_obj.file_path)
        if language is None:
            return # No rules registered for this language.

        rules: str = build_rules(language)
        ast_grep_bin: str = os.getenv('AST_GREP_BIN', 'ast-grep')

        try:
            # Every category runs in one invocation, and each match carries the
            # id of the rule that produced it.
            result: CompletedProcess[str] = await asyncio.to_thread(
                self._run_ast_grep, ast_grep_bin, rules, file_path
            )
        except FileNotFoundError:
            return # ast-grep is not installed.
        except Exception:
            return

        if result.returncode != 0 or not result.stdout:
            return

        try:
            matches: list[Any] = json.loads(result.stdout)
        except json.JSONDecodeError:
            return

        for match in matches:
            symbol_name: str | None = self._extract_symbol_name(match)
            if symbol_name is None:
                continue

            line_range: tuple[int, int] | None = self._extract_line_range(match)
            if line_range is None:
                continue

            kind: Any = match.get('ruleId')
            if not isinstance(kind, str) or not kind:
                continue

            await Symbol.create(
                file=file_obj,
                symbol_name=symbol_name,
                kind=kind,
                line_start=line_range[0],
                line_end=line_range[1]
            )

    @staticmethod
    def _extract_symbol_name(match: dict[str, Any]) -> str | None:
        """
        Read the captured `$NAME` metavariable out of an ast-grep match.

        The match's own `text` is the entire matched construct, body included,
        so it cannot be used as the symbol name: a later lookup by name would
        never match it.

        :param match: A single match object from ast-grep's JSON output.
        :returns: The captured symbol name, or None if the match has no `$NAME` capture.
        """
        meta_variables: Any = match.get('metaVariables')
        if not isinstance(meta_variables, dict):
            return None

        single: Any = meta_variables.get('single')
        if not isinstance(single, dict):
            return None

        name_capture: Any = single.get('NAME')
        if not isinstance(name_capture, dict):
            return None

        name: Any = name_capture.get('text')
        if not isinstance(name, str) or not name.strip():
            return None

        return name.strip()

    @staticmethod
    def _extract_line_range(match: dict[str, Any]) -> tuple[int, int] | None:
        """
        Read the 1-based, inclusive line range of an ast-grep match.

        ast-grep counts lines from zero, while editors, Git and the agent's own
        file reading tools all count from one. The conversion happens here, at
        the boundary, so that every consumer of the `symbols` table can assume a
        single convention.

        The end line is inclusive: it is the line holding the match's final
        character, and does not roll onto the following line when a match ends
        at a newline. A match with no usable range is rejected rather than
        defaulted, because a fabricated line number is indistinguishable from a
        real one downstream.

        :param match: A single match object from ast-grep's JSON output.
        :returns: The 1-based inclusive (start, end) line numbers, or None if the range is missing or malformed.
        """
        range_data: Any = match.get('range')
        if not isinstance(range_data, dict):
            return None

        start: int | None = IndexingService._read_line_number(range_data, 'start')
        end: int | None = IndexingService._read_line_number(range_data, 'end')
        if start is None or end is None:
            return None
        if end < start:
            return None

        return start + 1, end + 1

    @staticmethod
    def _read_line_number(range_data: dict[str, Any], boundary: str) -> int | None:
        """
        Read one 0-based boundary line number out of an ast-grep range object.

        :param range_data: The `range` object from an ast-grep match.
        :param boundary: Which boundary to read, either 'start' or 'end'.
        :returns: The 0-based line number, or None if it is absent or not a non-negative integer.
        """
        position: Any = range_data.get(boundary)
        if not isinstance(position, dict):
            return None

        line: Any = position.get('line')
        if isinstance(line, bool) or not isinstance(line, int):
            return None
        if line < 0:
            return None

        return line

    @staticmethod
    def _run_ast_grep(ast_grep_bin: str, rules: str, file_path: str) -> CompletedProcess[str]:
        """
        Run a set of ast-grep rules against a file on disk.

        The file is passed as a path rather than piped through stdin so that
        ast-grep reads the real content and reports the real file name.
        Explicitly named paths are read regardless of any ignore rules, so no
        file is silently skipped.

        :param ast_grep_bin: Path to the ast-grep executable.
        :param rules: The rule document, one document per symbol category.
        :param file_path: Absolute path to the file to search.
        :returns: The completed ast-grep process.
        :raises FileNotFoundError: If the ast-grep executable is not installed.
        """
        return subprocess.run(
            [ast_grep_bin, 'scan', '--inline-rules', rules, '--json', file_path],
            capture_output=True,
            text=True
        )
