from __future__ import annotations

from asyncio import Lock


class SessionRegistry:
    """
    Tracks the one agent session allowed to run at a time.

    Two sessions on one repository both create branches and commit to it, so the
    second sees changes it did not make and reviews work it did not do. Refusing
    the second connection makes that impossible rather than merely unlikely.

    Holding the slot and working on a task are deliberately separate facts. The
    slot is taken the instant a socket connects, before any prompt has been sent,
    because that is when a second connection has to be refused. Whether an agent
    is actually working is a narrower question, and the one that matters to
    anything asking whether it is safe to disturb the repository's index.
    """

    _active: str | None
    _working: str | None
    _lock: Lock

    def __init__(self) -> None:
        """
        Create an empty registry.
        """
        self._active = None
        self._working = None
        self._lock = Lock()

    @property
    def active_session_id(self) -> str | None:
        """
        The session currently holding the slot, if any.

        :returns: The active session identifier, or None.
        """
        return self._active

    @property
    def working_session_id(self) -> str | None:
        """
        The session currently working on a task, if any.

        :returns: The working session identifier, or None.
        """
        return self._working

    @property
    def is_busy(self) -> bool:
        """
        Whether a session is currently running.

        :returns: True if the slot is taken.
        """
        return self._active is not None

    @property
    def has_active_task(self) -> bool:
        """
        Whether a session is part way through a task.

        Narrower than `is_busy`: a connected session that has not been given a
        prompt yet, or one whose task has since reached a terminal state, holds
        the slot without working. Callers deciding whether the repository can be
        disturbed want this rather than `is_busy`.

        :returns: True if some session has begun a task and not yet finished it.
        """
        return self._working is not None

    async def acquire(self, session_id: str) -> bool:
        """
        Claim the slot for a session.

        :param session_id: The session asking for the slot.
        :returns: True if the slot was granted, False if another session holds it.
        """
        async with self._lock:
            if self._active is not None and self._active != session_id:
                return False
            self._active = session_id
            return True

    async def release(self, session_id: str) -> bool:
        """
        Give up the slot.

        A session that never held the slot releases nothing, so a refused
        connection cannot free the running session's claim on its way out.

        :param session_id: The session giving up the slot.
        :returns: True if this session held the slot and has released it.
        """
        async with self._lock:
            if self._active != session_id:
                return False
            self._active = None
            # A session torn down mid-task never reaches a terminal state, so
            # without this its claim to be working would outlive it and block
            # index deletion until the process restarted.
            if self._working == session_id:
                self._working = None
            return True

    async def begin_task(self, session_id: str) -> bool:
        """
        Record that a session has started working on a task.

        Only the session holding the slot may do this: a refused connection has
        no work to report, and letting it mark itself as working would block the
        repository on behalf of a session that is already closing.

        :param session_id: The session that has received its first prompt.
        :returns: True if the session holds the slot and is now marked working.
        """
        async with self._lock:
            if self._active != session_id:
                return False
            self._working = session_id
            return True

    async def end_task(self, session_id: str) -> bool:
        """
        Record that a session's task has finished.

        Called when the task reaches a terminal state, which may be well before
        the connection closes: the user is still free to read a finished review,
        and the index is no longer at risk once nothing is working against it.

        :param session_id: The session whose task has ended.
        :returns: True if this session was the one working and is no longer.
        """
        async with self._lock:
            if self._working != session_id:
                return False
            self._working = None
            return True
