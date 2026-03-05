"""Tests for utils.timer.SessionTimer."""

import time
import pytest
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
