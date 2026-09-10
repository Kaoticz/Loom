from __future__ import annotations

from services.events import SandboxStage


class SandboxError(RuntimeError):
    """
    Raised when the sandbox cannot be prepared or is not usable.

    Carries the step that failed, so that callers can tell the causes apart
    rather than presenting every one of them as the same sentence.
    """

    stage: SandboxStage

    def __init__(self, stage: SandboxStage, message: str) -> None:
        """
        Describe a sandbox failure.

        :param stage: Which step of preparing the sandbox failed.
        :param message: What went wrong, in terms the reader can act on.
        """
        super().__init__(message)
        self.stage = stage
