from pydantic import BaseModel, Field


class ModelDescription(BaseModel):
    """
    What a model reports about itself, for showing to the user.

    Both values are optional because a model need not describe itself, and
    neither is required for indexing to work. They exist so that a person
    choosing between models can see what they are choosing.
    """

    dimensions: int | None = Field(default=None, description='Vector width the model reports, or None if it does not say.')
    context_length: int | None = Field(default=None, description='Tokens the model accepts, or None if it does not say.')
