import os
import subprocess
import json
import hashlib
from database.models.file import File
from database.models.symbol import Symbol
from database.models.code_embedding import CodeEmbedding
import asyncio
import httpx

class IndexingService:
    """
    Service for scanning the codebase and populating the database.
    """
    repo_path: str

    def __init__(self, repo_path: str):
        self.repo_path = repo_path

    async def index_repository(self, progress_callback=None) -> None:
        """
        Scan the repository and update 'files', 'symbols', and 'embeddings' tables.
        """
        abs_repo_path = os.path.abspath(self.repo_path)
        exclude_dirs = {'.git', 'node_modules', '__pycache__', 'dist', 'build', '.venv', 'venv'}
        
        all_files = []
        for root, dirs, files in os.walk(abs_repo_path):
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            for file_name in files:
                all_files.append(os.path.join(root, file_name))
        
        total_files = len(all_files)
        if progress_callback:
            await progress_callback(0, total_files, "Starting indexing...")

        semaphore = asyncio.Semaphore(10) # Process 10 files at a time
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            tasks = []
            for idx, file_path in enumerate(all_files):
                tasks.append(self._process_file(file_path, abs_repo_path, semaphore, client, idx, total_files, progress_callback))
            
            await asyncio.gather(*tasks)

    async def _process_file(self, file_path, abs_repo_path, semaphore, client, idx, total_files, progress_callback):
        async with semaphore:
            rel_path = os.path.relpath(file_path, abs_repo_path)
            
            try:
                # Check if it is a binary file first
                with open(file_path, 'rb') as f:
                    chunk = f.read(1024)
                    if b'\x00' in chunk:
                        return # Skip binary files

                with open(file_path, 'r', errors='ignore') as f:
                    content = f.read()
            except Exception:
                return

            # Strip null bytes to be safe for PostgreSQL
            content = content.replace('\x00', '')
            
            if not content.strip():
                return

            file_hash = hashlib.sha256(content.encode()).hexdigest()
            
            file_obj, created = await File.get_or_create(
                file_path=rel_path,
                defaults={'raw_content': content, 'hash': file_hash}
            )
            
            needs_indexing = created
            if not created and file_obj.hash != file_hash:
                file_obj.raw_content = content
                file_obj.hash = file_hash
                await file_obj.save()
                
                # Clean and re-index
                await Symbol.filter(file=file_obj).delete()
                await CodeEmbedding.filter(file=file_obj).delete()
                needs_indexing = True
            
            if needs_indexing:
                await self._index_symbols(file_obj)
                await self._index_embeddings(file_obj, client)

            if progress_callback and ((idx + 1) % 5 == 0 or (idx + 1) == total_files):
                await progress_callback(idx + 1, total_files, f"Indexed {rel_path}")

    async def _index_embeddings(self, file_obj: File, client: httpx.AsyncClient) -> None:
        """
        Generate chunks and embeddings for a file.
        """
        content = file_obj.raw_content
        lines = content.splitlines()
        chunk_size = 50 # lines per chunk
        
        ollama_url = os.getenv('OLLAMA_BASE_URL', 'http://ollama:11434')
        embedding_model = os.getenv('EMBEDDING_MODEL', 'llama3')
        
        for i in range(0, len(lines), chunk_size):
            chunk_lines = lines[i:i + chunk_size]
            chunk_text = '\n'.join(chunk_lines)
            
            if not chunk_text.strip():
                continue
            
            try:
                resp = await client.post(
                    f'{ollama_url}/api/embeddings',
                    json={'model': embedding_model, 'prompt': chunk_text}
                )
                if resp.status_code == 200:
                    embedding = resp.json().get('embedding')
                    
                    if embedding:
                        await CodeEmbedding.create(
                            file=file_obj,
                            chunk_text=chunk_text,
                            embedding=embedding,
                            start_line=i + 1
                        )
            except Exception:
                continue

    async def _index_symbols(self, file_obj: File) -> None:
        """
        Use ast-grep to extract symbols from a file.
        """
        if not file_obj.file_path.endswith('.py'):
            return # Currently only support Python
            
        rules = [
            {'id': 'function', 'pattern': 'def $NAME($$$): $$$'},
            {'id': 'class', 'pattern': 'class $NAME: $$$'}
        ]
        
        ast_grep_bin = os.getenv('AST_GREP_BIN', 'ast-grep')
        for rule in rules:
            try:
                # Use subprocess to run ast-grep
                result = await asyncio.to_thread(
                    lambda: subprocess.run(
                        [ast_grep_bin, 'run', '--pattern', rule['pattern'], '--json'],
                        input=file_obj.raw_content,
                        capture_output=True,
                        text=True
                    )
                )
                
                if result.returncode == 0 and result.stdout:
                    try:
                        matches = json.loads(result.stdout)
                        for match in matches:
                            await Symbol.create(
                                file=file_obj,
                                symbol_name=match.get('text', 'unknown'),
                                kind=rule['id'],
                                line_start=match.get('range', {}).get('start', {}).get('line', 0),
                                line_end=match.get('range', {}).get('end', {}).get('line', 0)
                            )
                    except json.JSONDecodeError:
                        continue
            except FileNotFoundError:
                # ast-grep not installed
                break
            except Exception:
                continue
