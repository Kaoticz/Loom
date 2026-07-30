from datetime import datetime
from tortoise.fields import CharField, DatetimeField, Field, IntField, TextField
from tortoise.models import Model

class File(Model):
    """
    Stores file information from the repository.
    """
    id: Field[int] = IntField(primary_key=True)
    file_path: Field[str] = CharField(max_length=1024, unique=True)
    raw_content: Field[str] = TextField()
    hash: Field[str] = CharField(max_length=64)
    last_indexed: Field[datetime] = DatetimeField(auto_now=True)

    class Meta:  # pyright: ignore[reportIncompatibleVariableOverride]
        table: str = 'files'
