"""
Session timer using monotonic clock.

Provides consistent, drift-free timestamps for emotion recording sessions.
Uses time.monotonic() which cannot jump backwards due to NTP adjustments,
making it the correct foundation for the video sync controller in Phase 2.
"""

import time


class SessionTimer:
    """Tracks elapsed time from a defined start point using monotonic clock."""

    def __init__(self):
        self._start_time: float | None = None
        self._stop_time: float | None = None

    def start(self) -> None:
        """Mark the session start. Resets any previous state."""
        self._start_time = time.monotonic()
        self._stop_time = None

    def stop(self) -> None:
        """Mark the session end."""
        if self._start_time is None:
            raise RuntimeError("Timer was never started.")
        self._stop_time = time.monotonic()

    def elapsed_ms(self) -> int:
        """
        Return milliseconds elapsed since start.

        If the timer is running, returns time up to now.
        If the timer is stopped, returns time up to stop.
        Raises RuntimeError if the timer was never started.
        """
        if self._start_time is None:
            raise RuntimeError("Timer was never started.")

        end = self._stop_time if self._stop_time is not None else time.monotonic()
        return int((end - self._start_time) * 1000)

    @property
    def is_running(self) -> bool:
        return self._start_time is not None and self._stop_time is None

    @property
    def is_stopped(self) -> bool:
        return self._start_time is not None and self._stop_time is not None
