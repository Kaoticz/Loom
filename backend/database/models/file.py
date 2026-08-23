from datetime import datetime
from tortoise.fields import CharField, DatetimeField, Field, IntField, TextField
from tortoise.models import Model

class File(Model):
    """
    Stores file information from the repository.
    """
    id: Field[int] = IntField(primary_key=True)
    project: Field[str] = CharField(max_length=1024, db_index=True)
    file_path: Field[str] = CharField(max_length=1024)
    raw_content: Field[str] = TextField()
    hash: Field[str] = CharField(max_length=64)
    last_indexed: Field[datetime] = DatetimeField(auto_now=True)

    class Meta:  # pyright: ignore[reportIncompatibleVariableOverride]
        table: str = 'files'
        # `file_path` is repository-relative, so `src/main.py` names a different
        # file in every project. Unique globally, indexing a second project
        # overwrote the first project's row and cascaded away its symbols and
        # embeddings. Uniqueness belongs to the pair.
        unique_together: tuple[tuple[str, str], ...] = (('project', 'file_path'),)
