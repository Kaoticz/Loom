from __future__ import annotations
from pydantic_ai import Agent
from pydantic import BaseModel, Field
from typing import Annotated

class ReviewResult(BaseModel):
    """
    Structured result of a code review.
    """
    approved: Annotated[bool, Field()]
    comments: Annotated[str, Field()]

reviewer_agent: Agent[None, ReviewResult] = Agent(
    'ollama:llama3',
    result_type=ReviewResult, # type: ignore
    system_prompt=(
        'You are a senior code reviewer. Analyze the provided git diff and '
        'determine if it meets the quality standards. Provide detailed '
        'comments if you reject the changes.'
    )
)
