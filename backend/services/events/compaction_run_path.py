from enum import Enum


class CompactionRunPath(str, Enum):
    """
    The long-running agent path whose request history was shortened.
    """

    coder_streaming = 'coder_streaming'
    structured_planning = 'structured_planning'
    text_planning = 'text_planning'
