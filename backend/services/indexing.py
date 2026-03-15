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
        self.repo_path = os.path.abspath(repo_path)

    async def index_repository(self) -> None:
        """
        Scan the repository and update 'files', 'symbols', and 'embeddings' tables.
        """
        for root, _, files in os.walk(self.repo_path):
            if '.git' in root:
                continue
            
            for file_name in files:
                file_path = os.path.join(root, file_name)
                rel_path = os.path.relpath(file_path, self.repo_path)
                
                with open(file_path, 'r', errors='ignore') as f:
                    content = f.read()
                
                file_hash = hashlib.sha256(content.encode()).hexdigest()
                
                file_obj, created = await File.get_or_create(
                    file_path=rel_path,
                    defaults={'raw_content': content, 'hash': file_hash}
                )
                
                if not created and file_obj.hash != file_hash:
                    file_obj.raw_content = content
                    file_obj.hash = file_hash
                    await file_obj.save()
                    
                    # Clean and re-index
                    await Symbol.filter(file=file_obj).delete()
                    await CodeEmbedding.filter(file=file_obj).delete()
                    
                    await self._index_symbols(file_obj)
                    await self._index_embeddings(file_obj)
                elif created:
                    await self._index_symbols(file_obj)
                    await self._index_embeddings(file_obj)

    async def _index_embeddings(self, file_obj: File) -> None:
        """
        Generate chunks and embeddings for a file.
        """
        content = file_obj.raw_content
        lines = content.splitlines()
        chunk_size = 50 # lines per chunk
        
        ollama_url = os.getenv('OLLAMA_BASE_URL', 'http://ollama:11434')
        
        for i in range(0, len(lines), chunk_size):
            chunk_lines = lines[i:i + chunk_size]
            chunk_text = '\n'.join(chunk_lines)
            
            if not chunk_text.strip():
                continue
            
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.post(
                        f'{ollama_url}/api/embeddings',
                        json={'model': 'llama3', 'prompt': chunk_text}
                    )
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
        abs_path = os.path.join(self.repo_path, file_obj.file_path)
        
        # Predefined rule to find functions and classes in Python
        # In production, we would support multiple languages
        rules = [
            {'id': 'function', 'pattern': 'def $NAME($$$): $$$'},
            {'id': 'class', 'pattern': 'class $NAME: $$$'}
        ]
        
        for rule in rules:
            try:
                # Use subprocess to run ast-grep
                result = await asyncio.to_thread(
                    lambda: subprocess.run(
                        ['ast-grep', 'run', '--pattern', rule['pattern'], '--json'],
                        input=file_obj.raw_content,
                        capture_output=True,
                        text=True
                    )
                )
                
                if result.returncode == 0 and result.stdout:
                    matches = json.loads(result.stdout)
                    for match in matches:
                        await Symbol.create(
                            file=file_obj,
                            symbol_name=match.get('text', 'unknown'),
                            kind=rule['id'],
                            line_start=match.get('range', {}).get('start', {}).get('line', 0),
                            line_end=match.get('range', {}).get('end', {}).get('line', 0)
                        )
            except Exception:
                continue
