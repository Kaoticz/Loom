from __future__ import annotations

from pydantic_ai import Agent

COMMIT_MESSAGE_SYSTEM_PROMPT: str = '''You write Git commit subjects for completed coding tasks.

Read the user's original task, any later user feedback, and the task branch's commit subjects.
Derive the task's overall outcome and output exactly one concise imperative-mood commit subject.
Do not include a prefix, quotation marks, markdown, explanation, or trailing period. For example:
Add a retry policy to the session inbox'''


commit_message_agent: Agent[None, str] = Agent(
    deps_type=type(None),
    output_type=str,
    system_prompt=COMMIT_MESSAGE_SYSTEM_PROMPT,
    retries=2,
)
