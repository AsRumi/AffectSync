"""
Session timer using monotonic clock.

Provides consistent, drift-free timestamps for emotion recording sessions
and video sync. Uses time.monotonic() which cannot jump backwards due to
NTP adjustments, making it the single source of truth for all timestamp
coordination in the pipeline.

pause() and resume() allow the sync controller to
freeze both video playback and emotion recording simultaneously.

elapsed_ms returns time subtracting paused time.
"""

import time


class SessionTimer:
    """
    Tracks elapsed time from a defined start point using monotonic clock.

    Elapsed time excludes all paused intervals. This means a timer that
    runs for 5s, pauses for 10s, then runs for 5s reports elapsed = 10s,
    not 20s. Both the video player and emotion recorder ask this timer
    "what time is it?", so pausing the timer pauses everything.
    """

    def __init__(self):
        self._start_time: float | None = None
        self._stop_time: float | None = None
        self._pause_time: float | None = None
        self._paused_accumulated: float = 0.0

    def start(self) -> None:
        """Mark the session start. Resets any previous state."""
        self._start_time = time.monotonic()
        self._stop_time = None
        self._pause_time = None
        self._paused_accumulated = 0.0

    def stop(self) -> None:
        """
        Mark the session end. Can be called from running or paused state.

        If called while paused, the stop time is set to the moment the
        timer was paused (not now), so elapsed doesn't include the final
        pause duration.
        """
        if self._start_time is None:
            raise RuntimeError("Timer was never started.")

        if self._pause_time is not None:
            # Stopped while paused — freeze at the pause moment
            self._stop_time = self._pause_time
            self._pause_time = None
        else:
            self._stop_time = time.monotonic()

    def pause(self) -> None:
        """
        Pause the timer. Elapsed time freezes at the current value.

        Raises RuntimeError if the timer is not currently running.
        """
        if not self.is_running:
            raise RuntimeError(
                "Cannot pause: timer is not running. "
                "Current state: "
                + ("paused" if self.is_paused else
                   "stopped" if self.is_stopped else "idle")
            )
        self._pause_time = time.monotonic()

    def resume(self) -> None:
        """
        Resume the timer from a paused state.

        Adds the duration of this pause to the accumulated pause time,
        so elapsed_ms() continues from where it left off with no jump.

        Raises RuntimeError if the timer is not currently paused.
        """
        if not self.is_paused:
            raise RuntimeError(
                "Cannot resume: timer is not paused. "
                "Current state: "
                + ("running" if self.is_running else
                   "stopped" if self.is_stopped else "idle")
            )
        # Accumulate the duration of this pause interval
        self._paused_accumulated += time.monotonic() - self._pause_time
        self._pause_time = None

    def elapsed_ms(self) -> int:
        """
        Return milliseconds elapsed since start, excluding paused time.

        - Running: time from start to now, minus all paused intervals.
        - Paused: time from start to pause moment, minus prior pauses.
        - Stopped: time from start to stop, minus all paused intervals.
        - Idle: raises RuntimeError.
        """
        if self._start_time is None:
            raise RuntimeError("Timer was never started.")

        if self._stop_time is not None:
            # Stopped — frozen value
            raw = self._stop_time - self._start_time
        elif self._pause_time is not None:
            # Paused — frozen at pause moment
            raw = self._pause_time - self._start_time
        else:
            # Running — live value
            raw = time.monotonic() - self._start_time

        return int((raw - self._paused_accumulated) * 1000)

    @property
    def is_running(self) -> bool:
        """True when started, not paused, and not stopped."""
        return (
            self._start_time is not None
            and self._stop_time is None
            and self._pause_time is None
        )

    @property
    def is_paused(self) -> bool:
        """True when the timer is in a paused state."""
        return (
            self._start_time is not None
            and self._stop_time is None
            and self._pause_time is not None
        )

    @property
    def is_stopped(self) -> bool:
        """True when the timer has been stopped (terminal state)."""
        return self._start_time is not None and self._stop_time is not None
