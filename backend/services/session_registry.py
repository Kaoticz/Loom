from __future__ import annotations

from asyncio import Lock


class SessionRegistry:
    """
    Tracks the one agent session allowed to run at a time.

    Two sessions on one repository both create branches and commit to it, so the
    second sees changes it did not make and reviews work it did not do. Refusing
    the second connection makes that impossible rather than merely unlikely.
    """

    _active: str | None
    _lock: Lock

    def __init__(self) -> None:
        """
        Create an empty registry.
        """
        self._active = None
        self._lock = Lock()

    @property
    def active_session_id(self) -> str | None:
        """
        The session currently holding the slot, if any.

        :returns: The active session identifier, or None.
        """
        return self._active

    @property
    def is_busy(self) -> bool:
        """
        Whether a session is currently running.

        :returns: True if the slot is taken.
        """
        return self._active is not None

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
            return True
