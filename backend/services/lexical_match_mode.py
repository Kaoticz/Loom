from enum import Enum


class LexicalMatchMode(str, Enum):
    """
    How several keywords combine when searching for lines.

    The values are what the model sees in the tool schema, so they are the plain
    words rather than the member names.
    """

    any_of = 'any'
    all_of = 'all'
