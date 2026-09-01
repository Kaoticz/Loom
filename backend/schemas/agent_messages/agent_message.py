from typing import Annotated

from pydantic import Field

from .action_message import ActionMessage
from .commit_message import CommitMessage
from .compaction_message import CompactionMessage
from .context_usage_message import ContextUsageMessage
from .diff_message import DiffMessage
from .done_message import DoneMessage
from .error_message import ErrorMessage
from .observation_message import ObservationMessage
from .plan_message import PlanMessage
from .question_message import QuestionMessage
from .review_message import ReviewMessage
from .sandbox_unavailable_message import SandboxUnavailableMessage
from .session_busy_message import SessionBusyMessage
from .system_message import SystemMessage
from .thought_message import ThoughtMessage
from .warning_message import WarningMessage


AgentMessage = Annotated[
    ThoughtMessage
    | PlanMessage
    | ActionMessage
    | ObservationMessage
    | CommitMessage
    | DiffMessage
    | QuestionMessage
    | ReviewMessage
    | DoneMessage
    | ErrorMessage
    | WarningMessage
    | ContextUsageMessage
    | CompactionMessage
    | SessionBusyMessage
    | SystemMessage
    | SandboxUnavailableMessage,
    Field(discriminator='type'),
]
