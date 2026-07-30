from database.models.file import File
from tortoise.fields import CharField, Field, ForeignKeyField, ForeignKeyRelation, IntField, OnDelete
from tortoise.models import Model

class Symbol(Model):
    """
    Stores AST symbols (functions, classes) found in files.
    """
    id: Field[int] = IntField(primary_key=True)
    """
    The unique identifier for the symbol.
    """

    file: ForeignKeyRelation[File] = ForeignKeyField(
        'models.File',
        related_name='symbols',
        on_delete=OnDelete.CASCADE
    )
    """
    The file the symbol was found in.
    """

    symbol_name: Field[str] = CharField(max_length=255, index=True)
    """
    The declared name of the symbol.
    """

    kind: Field[str] = CharField(max_length=50)
    """
    What kind of symbol this is, for example 'function' or 'class'.
    """

    line_start: Field[int] = IntField()
    """
    The 1-based line where the symbol's definition begins.
    """

    line_end: Field[int] = IntField()
    """
    The 1-based line where the symbol's definition ends, inclusive.
    """

    class Meta:  # pyright: ignore[reportIncompatibleVariableOverride]
        table: str = 'symbols'
        indexes: list[tuple[str, ...]] = [('file',)]
