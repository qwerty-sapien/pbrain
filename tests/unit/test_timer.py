"""Unit tests for timer module.

Tests TimerManager, CaffeinateManager, BreakReminder, and send_notification.
macOS subprocess calls are mocked for CI compatibility.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from pb.core.timer import (
    BreakReminder,
    CaffeinateManager,
    TimerManager,
    TimerState,
    send_notification,
)


class TestSendNotification:
    """Tests for send_notification function."""

    @patch("pb.core.timer._has_terminal_notifier", return_value=False)
    @patch("pb.core.timer.subprocess.run")
    def test_send_notification_success(self, mock_run, _mock_tn):
        """Notification returns True when osascript succeeds. Silent by default."""
        mock_run.return_value = MagicMock(returncode=0)

        result = send_notification("Test Title", "Test Message")

        assert result is True
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert call_args == ["osascript"]
        script = mock_run.call_args[1]["input"]
        assert "display notification" in script
        assert "Test Message" in script
        assert "Test Title" in script
        assert "sound name" not in script

    @patch("pb.core.timer._has_terminal_notifier", return_value=False)
    @patch("pb.core.timer.subprocess.run")
    def test_send_notification_with_sound(self, mock_run, _mock_tn):
        """Notification with sound=True includes sound clause."""
        mock_run.return_value = MagicMock(returncode=0)

        result = send_notification("Title", "Message", sound=True)

        assert result is True
        script = mock_run.call_args[1]["input"]
        assert 'sound name "default"' in script

    @patch("pb.core.timer._has_terminal_notifier", return_value=False)
    @patch("pb.core.timer.subprocess.run")
    def test_send_notification_failure(self, mock_run, _mock_tn):
        """Notification returns False when osascript fails."""
        mock_run.return_value = MagicMock(returncode=1)

        result = send_notification("Title", "Message")

        assert result is False

    @patch("pb.core.timer._has_terminal_notifier", return_value=False)
    @patch("pb.core.timer.subprocess.run")
    def test_send_notification_exception(self, mock_run, _mock_tn):
        """Notification returns False when exception occurs."""
        mock_run.side_effect = Exception("OSError")

        result = send_notification("Title", "Message")

        assert result is False

    @patch("pb.core.timer._has_terminal_notifier", return_value=False)
    @patch("pb.core.timer.subprocess.run")
    def test_send_notification_escapes_quotes(self, mock_run, _mock_tn):
        """Double quotes in title/message are escaped."""
        mock_run.return_value = MagicMock(returncode=0)

        send_notification('Title with "quotes"', 'Message with "quotes"')

        script = mock_run.call_args[1]["input"]
        assert '\\"' in script

    @patch("pb.core.timer._has_terminal_notifier", return_value=True)
    @patch("pb.core.timer.subprocess.run")
    def test_send_notification_terminal_notifier(self, mock_run, _mock_tn):
        """Uses terminal-notifier when available. Click activates Terminal."""
        mock_run.return_value = MagicMock(returncode=0)

        result = send_notification("Title", "Message")

        assert result is True
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "terminal-notifier"
        assert "-activate" in cmd
        assert "com.apple.Terminal" in cmd
        assert "-execute" in cmd
        assert "pb" in cmd
        assert "-sound" not in cmd


class TestCaffeinateManager:
    """Tests for CaffeinateManager class."""

    @patch("pb.core.timer.CAFFEINATE_PID_FILE")
    @patch("pb.core.timer.subprocess.Popen")
    def test_start_creates_process(self, mock_popen, mock_pid_file):
        """start() spawns caffeinate subprocess with -i (not -w), writes PID file."""
        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_popen.return_value = mock_process
        mock_pid_file.parent = MagicMock()

        manager = CaffeinateManager()
        result = manager.start()

        assert result is True
        mock_popen.assert_called_once()
        call_args = mock_popen.call_args[0][0]
        assert "caffeinate" in call_args
        assert "-i" in call_args
        assert "-w" not in call_args
        mock_pid_file.write_text.assert_called_once_with("12345")

    @patch("pb.core.timer.subprocess.Popen")
    def test_start_only_once(self, mock_popen):
        """start() is idempotent - doesn't create multiple processes."""
        mock_popen.return_value = MagicMock()

        manager = CaffeinateManager()
        result1 = manager.start()
        result2 = manager.start()

        assert result1 is True
        assert result2 is True
        assert mock_popen.call_count == 1

    @patch("pb.core.timer.subprocess.Popen")
    def test_start_handles_exception(self, mock_popen):
        """start() returns False on exception."""
        mock_popen.side_effect = Exception("Process error")

        manager = CaffeinateManager()
        result = manager.start()

        assert result is False

    @patch("pb.core.timer.subprocess.Popen")
    def test_stop_terminates_process(self, mock_popen):
        """stop() terminates the caffeinate process."""
        mock_process = MagicMock()
        mock_popen.return_value = mock_process

        manager = CaffeinateManager()
        manager.start()
        manager.stop()

        mock_process.terminate.assert_called_once()
        assert manager.process is None

    @patch("pb.core.timer.subprocess.Popen")
    def test_stop_when_not_started(self, mock_popen):
        """stop() does nothing if not started."""
        manager = CaffeinateManager()
        manager.stop()

    @patch("pb.core.timer.subprocess.Popen")
    def test_is_active_when_running(self, mock_popen):
        """is_active returns True when process is running."""
        mock_process = MagicMock()
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process

        manager = CaffeinateManager()
        assert manager.is_active is False

        manager.start()
        assert manager.is_active is True

    @patch("pb.core.timer.subprocess.Popen")
    def test_is_active_when_stopped(self, mock_popen):
        """is_active returns False after stop."""
        mock_popen.return_value = MagicMock()

        manager = CaffeinateManager()
        manager.start()
        manager.stop()
        assert manager.is_active is False

    @patch("pb.core.timer.subprocess.Popen")
    def test_is_active_when_process_exited(self, mock_popen):
        """is_active returns False when process has exited."""
        mock_process = MagicMock()
        mock_process.poll.return_value = 0
        mock_popen.return_value = mock_process

        manager = CaffeinateManager()
        manager.start()
        assert manager.is_active is False


class TestBreakReminder:
    """Tests for BreakReminder class."""

    def test_interval_is_30_minutes(self):
        """Break reminder interval is 30 minutes per D-06."""
        assert BreakReminder.INTERVAL_MINUTES == 30

    def test_start_sets_running_flag(self):
        """start() sets running flag."""
        reminder = BreakReminder()
        reminder.start()

        assert reminder.running is True
        reminder.stop()

    @patch("pb.core.timer.send_notification")
    def test_start_creates_timer(self, mock_notify):
        """start() sends a break-cadence notification (no background timer)."""
        reminder = BreakReminder()
        reminder.start()

        mock_notify.assert_called_once()
        reminder.stop()

    def test_stop_clears_state(self):
        """stop() clears running flag."""
        with patch("pb.core.timer.send_notification"):
            reminder = BreakReminder()
            reminder.start()
            reminder.stop()

        assert reminder.running is False

    def test_stop_when_not_started(self):
        """stop() does nothing if not started."""
        reminder = BreakReminder()
        reminder.stop()
        assert reminder.running is False

    @patch("pb.core.timer.send_notification")
    def test_start_sends_single_notification(self, mock_notify):
        """start() sends one upfront notification with break cadence message."""
        reminder = BreakReminder()
        reminder.start()
        mock_notify.assert_called_once()
        args = mock_notify.call_args[0]
        assert "break" in args[1].lower() or "30" in args[1]
        reminder.stop()


class TestTimerState:
    """Tests for TimerState dataclass."""

    def test_timer_state_creation(self):
        """TimerState can be created with required fields."""
        from datetime import datetime

        state = TimerState(
            session_id="session-1",
            start_time=datetime.now(),
            duration_minutes=30,
            task_title="Test Task",
        )

        assert state.session_id == "session-1"
        assert state.duration_minutes == 30
        assert state.task_title == "Test Task"


class TestTimerManager:
    """Tests for TimerManager class."""

    @patch("pb.core.timer.CaffeinateManager")
    @patch("pb.core.timer.BreakReminder")
    def test_start_session_timers_with_duration(self, mock_break_class, mock_caff_class):
        """start_session_timers() records state without OS-side effects."""
        mock_break = MagicMock()
        mock_caff = MagicMock()
        mock_break_class.return_value = mock_break
        mock_caff_class.return_value = mock_caff

        manager = TimerManager()
        manager.start_session_timers("session-1", duration_minutes=30, task_title="Test Task")

        assert manager.state is not None
        assert manager.state.session_id == "session-1"
        assert manager.state.duration_minutes == 30
        mock_break.start.assert_not_called()
        mock_caff.start.assert_not_called()

    @patch("pb.core.timer.CaffeinateManager")
    @patch("pb.core.timer.BreakReminder")
    def test_start_session_timers_without_duration(self, mock_break_class, mock_caff_class):
        """start_session_timers() works without duration and stays prompt-first."""
        mock_break = MagicMock()
        mock_caff = MagicMock()
        mock_break_class.return_value = mock_break
        mock_caff_class.return_value = mock_caff

        manager = TimerManager()
        manager.start_session_timers("session-1", duration_minutes=None, task_title="Test Task")

        assert manager.state is not None
        mock_break.start.assert_not_called()
        mock_caff.start.assert_not_called()

    @patch("pb.core.timer.CaffeinateManager")
    @patch("pb.core.timer.BreakReminder")
    def test_start_session_timers_stops_previous(self, mock_break_class, mock_caff_class):
        """start_session_timers() stops any previous session timers."""
        mock_break = MagicMock()
        mock_caff = MagicMock()
        mock_break_class.return_value = mock_break
        mock_caff_class.return_value = mock_caff

        manager = TimerManager()
        manager.start_session_timers("session-1", duration_minutes=30, task_title="Task 1")
        manager.start_session_timers("session-2", duration_minutes=45, task_title="Task 2")

        assert manager.state.session_id == "session-2"

    @patch("pb.core.timer.CaffeinateManager")
    @patch("pb.core.timer.BreakReminder")
    def test_stop_session_timers(self, mock_break_class, mock_caff_class):
        """stop_session_timers() stops all timers and clears state."""
        mock_break = MagicMock()
        mock_caff = MagicMock()
        mock_break_class.return_value = mock_break
        mock_caff_class.return_value = mock_caff

        manager = TimerManager()
        manager.start_session_timers("session-1", duration_minutes=30, task_title="Test Task")
        manager.stop_session_timers()

        assert manager.state is None
        mock_break.stop.assert_called()
        mock_caff.stop.assert_called()

    @patch("pb.core.timer.CaffeinateManager")
    @patch("pb.core.timer.BreakReminder")
    def test_stop_session_timers_when_no_session(self, mock_break_class, mock_caff_class):
        """stop_session_timers() is safe when no session active."""
        mock_break_class.return_value = MagicMock()
        mock_caff_class.return_value = MagicMock()

        manager = TimerManager()
        manager.stop_session_timers()

    @patch("pb.core.timer.CaffeinateManager")
    @patch("pb.core.timer.BreakReminder")
    def test_get_elapsed_minutes_no_session(self, mock_break_class, mock_caff_class):
        """get_elapsed_minutes() returns None when no session."""
        mock_break_class.return_value = MagicMock()
        mock_caff_class.return_value = MagicMock()

        manager = TimerManager()
        assert manager.get_elapsed_minutes() is None

    @patch("pb.core.timer.CaffeinateManager")
    @patch("pb.core.timer.BreakReminder")
    def test_get_elapsed_minutes_with_session(self, mock_break_class, mock_caff_class):
        """get_elapsed_minutes() returns minutes since start."""
        mock_break_class.return_value = MagicMock()
        mock_caff_class.return_value = MagicMock()

        manager = TimerManager()
        manager.start_session_timers("session-1", duration_minutes=30, task_title="Test")

        elapsed = manager.get_elapsed_minutes()
        assert elapsed is not None
        assert elapsed >= 0
        assert elapsed < 1

    @patch("pb.core.timer.CaffeinateManager")
    @patch("pb.core.timer.BreakReminder")
    def test_get_remaining_minutes_with_duration(self, mock_break_class, mock_caff_class):
        """get_remaining_minutes() returns remaining time when duration set."""
        mock_break_class.return_value = MagicMock()
        mock_caff_class.return_value = MagicMock()

        manager = TimerManager()
        manager.start_session_timers("session-1", duration_minutes=30, task_title="Test")

        remaining = manager.get_remaining_minutes()
        assert remaining is not None
        assert remaining == 30 or remaining == 29

    @patch("pb.core.timer.CaffeinateManager")
    @patch("pb.core.timer.BreakReminder")
    def test_get_remaining_minutes_without_duration(self, mock_break_class, mock_caff_class):
        """get_remaining_minutes() returns None when no duration set."""
        mock_break_class.return_value = MagicMock()
        mock_caff_class.return_value = MagicMock()

        manager = TimerManager()
        manager.start_session_timers("session-1", duration_minutes=None, task_title="Test")

        assert manager.get_remaining_minutes() is None

    @patch("pb.core.timer.CaffeinateManager")
    @patch("pb.core.timer.BreakReminder")
    def test_get_remaining_minutes_no_session(self, mock_break_class, mock_caff_class):
        """get_remaining_minutes() returns None when no session."""
        mock_break_class.return_value = MagicMock()
        mock_caff_class.return_value = MagicMock()

        manager = TimerManager()
        assert manager.get_remaining_minutes() is None

    def test_default_init(self):
        """TimerManager creates default break_reminder and caffeinate."""
        manager = TimerManager()

        assert isinstance(manager.break_reminder, BreakReminder)
        assert isinstance(manager.caffeinate, CaffeinateManager)
        assert manager.state is None

    @patch("pb.core.timer.send_notification")
    @patch("pb.core.timer.CaffeinateManager")
    @patch("pb.core.timer.BreakReminder")
    def test_start_session_timers_does_not_send_block_notification(
        self, mock_break_class, mock_caff_class, mock_notify
    ):
        """start_session_timers() avoids default notifications for learning sessions."""
        mock_break_class.return_value = MagicMock()
        mock_caff_class.return_value = MagicMock()

        manager = TimerManager()
        manager.start_session_timers("session-1", duration_minutes=45, task_title="Deep Work")

        mock_notify.assert_not_called()

    @patch("pb.core.timer.send_notification")
    @patch("pb.core.timer.CaffeinateManager")
    @patch("pb.core.timer.BreakReminder")
    def test_start_session_timers_no_block_notification_without_duration(
        self, mock_break_class, mock_caff_class, mock_notify
    ):
        """start_session_timers() stays silent for untimed sessions too."""
        mock_break_class.return_value = MagicMock()
        mock_caff_class.return_value = MagicMock()

        manager = TimerManager()
        manager.start_session_timers("session-1", duration_minutes=None, task_title="Open Work")

        mock_notify.assert_not_called()
