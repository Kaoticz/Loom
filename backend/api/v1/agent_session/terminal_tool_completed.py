class TerminalToolCompleted(Exception):
    """
    End a coder run after a terminal tool completed successfully.

    The tool name is carried as the exception message so the run wrapper can
    preserve the summary recorded by the matching completion path.
    """
