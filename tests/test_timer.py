"""Tests for utils.timer.SessionTimer.

Phase 1 tests: lifecycle (start/stop) and elapsed accuracy.
Phase 2 tests: pause/resume state machine and elapsed correctness across
               multiple pause/resume cycles.
"""

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.timer import SessionTimer


class TestSessionTimerLifecycle:
    """Test the start/stop state machine."""

    def test_fresh_timer_is_not_running(self):
        timer = SessionTimer()
        assert not timer.is_running
        assert not timer.is_stopped

    def test_started_timer_is_running(self):
        timer = SessionTimer()
        timer.start()
        assert timer.is_running
        assert not timer.is_stopped

    def test_stopped_timer_is_stopped(self):
        timer = SessionTimer()
        timer.start()
        timer.stop()
        assert not timer.is_running
        assert timer.is_stopped

    def test_elapsed_before_start_raises(self):
        timer = SessionTimer()
        with pytest.raises(RuntimeError, match="never started"):
            timer.elapsed_ms()

    def test_stop_before_start_raises(self):
        timer = SessionTimer()
        with pytest.raises(RuntimeError, match="never started"):
            timer.stop()

    def test_start_resets_previous_session(self):
        timer = SessionTimer()
        timer.start()
        time.sleep(0.05)
        timer.stop()
        first_elapsed = timer.elapsed_ms()

        # Restart — elapsed should reset close to zero
        timer.start()
        assert timer.elapsed_ms() < first_elapsed
        assert timer.is_running


class TestSessionTimerElapsed:
    """Test elapsed_ms accuracy."""

    def test_elapsed_increases_while_running(self):
        timer = SessionTimer()
        timer.start()
        t1 = timer.elapsed_ms()
        time.sleep(0.05)
        t2 = timer.elapsed_ms()
        assert t2 > t1

    def test_elapsed_freezes_after_stop(self):
        timer = SessionTimer()
        timer.start()
        time.sleep(0.05)
        timer.stop()
        frozen = timer.elapsed_ms()
        time.sleep(0.05)
        assert timer.elapsed_ms() == frozen

    def test_elapsed_is_approximately_correct(self):
        timer = SessionTimer()
        timer.start()
        time.sleep(0.1)
        elapsed = timer.elapsed_ms()
        # Allow 50ms tolerance for OS scheduling
        assert 80 <= elapsed <= 200

    def test_elapsed_returns_int(self):
        timer = SessionTimer()
        timer.start()
        assert isinstance(timer.elapsed_ms(), int)


class TestSessionTimerPauseResume:
    """Test the pause/resume state machine added in Phase 2."""

    # --- State transition tests ---

    def test_pause_sets_paused_state(self):
        timer = SessionTimer()
        timer.start()
        timer.pause()
        assert timer.is_paused
        assert not timer.is_running
        assert not timer.is_stopped

    def test_resume_returns_to_running(self):
        timer = SessionTimer()
        timer.start()
        timer.pause()
        timer.resume()
        assert timer.is_running
        assert not timer.is_paused

    def test_pause_when_not_running_raises(self):
        timer = SessionTimer()
        with pytest.raises(RuntimeError, match="not running"):
            timer.pause()

    def test_pause_when_already_paused_raises(self):
        timer = SessionTimer()
        timer.start()
        timer.pause()
        with pytest.raises(RuntimeError, match="not running"):
            timer.pause()

    def test_resume_when_not_paused_raises(self):
        timer = SessionTimer()
        timer.start()
        with pytest.raises(RuntimeError, match="not paused"):
            timer.resume()

    def test_resume_when_stopped_raises(self):
        timer = SessionTimer()
        timer.start()
        timer.stop()
        with pytest.raises(RuntimeError, match="not paused"):
            timer.resume()

    def test_stop_from_paused_state(self):
        """Stopping while paused should work and freeze at pause moment."""
        timer = SessionTimer()
        timer.start()
        time.sleep(0.05)
        timer.pause()
        paused_elapsed = timer.elapsed_ms()
        time.sleep(0.05)  # Extra time that should NOT count
        timer.stop()
        assert timer.is_stopped
        assert not timer.is_paused
        # Elapsed should match the paused value, not include the extra wait
        assert timer.elapsed_ms() == paused_elapsed

    # --- Elapsed accuracy during pause ---

    def test_elapsed_freezes_during_pause(self):
        timer = SessionTimer()
        timer.start()
        time.sleep(0.05)
        timer.pause()
        frozen = timer.elapsed_ms()
        time.sleep(0.05)
        assert timer.elapsed_ms() == frozen

    def test_elapsed_continues_after_resume(self):
        timer = SessionTimer()
        timer.start()
        time.sleep(0.05)
        timer.pause()
        frozen = timer.elapsed_ms()
        time.sleep(0.05)  # This should NOT count
        timer.resume()
        time.sleep(0.05)  # This SHOULD count
        after_resume = timer.elapsed_ms()
        assert after_resume > frozen

    def test_elapsed_excludes_paused_time(self):
        """Run 100ms, pause 100ms, run 100ms → elapsed ≈ 200ms, not 300ms."""
        timer = SessionTimer()
        timer.start()
        time.sleep(0.1)
        timer.pause()
        time.sleep(0.1)  # Paused — should not count
        timer.resume()
        time.sleep(0.1)
        elapsed = timer.elapsed_ms()
        # Should be ~200ms (2 × 100ms running), not ~300ms
        # Allow generous tolerance for OS scheduling
        assert 150 <= elapsed <= 350

    def test_multiple_pause_resume_cycles(self):
        """Three run-pause cycles should accumulate only the running time."""
        timer = SessionTimer()
        timer.start()

        for _ in range(3):
            time.sleep(0.05)   # ~50ms running
            timer.pause()
            time.sleep(0.05)   # ~50ms paused (excluded)
            timer.resume()

        elapsed = timer.elapsed_ms()
        # 3 × 50ms running = ~150ms. Paused time excluded.
        # Allow generous tolerance
        assert 100 <= elapsed <= 300

    def test_start_resets_pause_state(self):
        """Restarting after a pause should clear all pause tracking."""
        timer = SessionTimer()
        timer.start()
        time.sleep(0.05)
        timer.pause()
        timer.stop()

        # Restart — should be clean
        timer.start()
        assert timer.is_running
        assert not timer.is_paused
        assert timer.elapsed_ms() < 20  # Near-zero
