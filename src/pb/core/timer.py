# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Timer management for session experience.

Handles macOS notifications via osascript display notification,
break reminders, block start notifications, and caffeinate for
preventing system sleep. Never uses AppleScript app automation
(tell application) to avoid macOS admin permission dialogs.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import structlog

logger = structlog.get_logger()

STATE_DIR = Path.home() / ".local" / "state" / "productivebrain"
LEGACY_STATE_DIR = Path.home() / ".local" / "state" / "pb"
CAFFEINATE_PID_FILE = STATE_DIR / "caffeinate.pid"
AT_JOB_ID_FILE = STATE_DIR / "at_job.id"
TIMER_EXPIRED_FLAG = STATE_DIR / "timer_expired.flag"
LEGACY_CAFFEINATE_PID_FILE = LEGACY_STATE_DIR / "caffeinate.pid"
LEGACY_AT_JOB_ID_FILE = LEGACY_STATE_DIR / "at_job.id"


def _has_terminal_notifier() -> bool:
    """Return True when terminal-notifier is available on PATH or Homebrew default."""
    return shutil.which("terminal-notifier") is not None or Path("/opt/homebrew/bin/terminal-notifier").exists()


def schedule_actionable_notification(
    *,
    title: str,
    message: str,
    execute: str,
    delay_minutes: int,
) -> bool:
    """Schedule a clickable reminder notification via at(1)."""
    if delay_minutes <= 0:
        return send_notification(title, message, execute=execute)

    try:
        if _has_terminal_notifier():
            safe_title = title.replace("'", "'\\''")
            safe_message = message.replace("'", "'\\''")
            safe_execute = execute.replace("'", "'\\''")
            notify_command = (
                "terminal-notifier "
                f"-title '{safe_title}' "
                f"-message '{safe_message}' "
                "-activate com.apple.Terminal "
                f"-execute '{safe_execute}'"
            )
        else:
            safe_title = title.replace('"', '\\"')
            safe_message = message.replace('"', '\\"')
            notify_command = (
                "osascript -e "
                f"'display notification \"{safe_message}\" with title \"{safe_title}\"'"
            )
        proc = subprocess.run(
            ["at", f"now + {delay_minutes} minutes"],
            input=f"{notify_command}\n",
            text=True,
            capture_output=True,
        )
        return proc.returncode == 0
    except Exception:
        return False


def _schedule_auto_finish(duration_minutes: int) -> None:
    """Schedule a notification at block expiry via 'at' command (D-11).

    Sends a clickable notification instead of forcibly opening Terminal
    (which requires Automation permissions and triggers admin prompts).
    Also writes timer_expired.flag so the next pb command intercepts the expiry (D-13/D-14).
    """
    if not isinstance(duration_minutes, int) or duration_minutes <= 0:
        return
    try:
        flag_path = str(TIMER_EXPIRED_FLAG)
        if _has_terminal_notifier():
            notify_command = (
                "terminal-notifier "
                "-title 'Block finished' "
                "-message 'Time to wrap up' "
                "-activate com.apple.Terminal "
                "-execute 'pb next' "
                "-sound default"
            )
        else:
            notify_command = (
                "osascript -e 'display notification \"Time to wrap up\" "
                "with title \"Block finished\" sound name \"default\"'"
            )
        at_input = f"{notify_command} && touch {flag_path}\n"
        proc = subprocess.run(
            ["at", f"now + {duration_minutes} minutes"],
            input=at_input,
            text=True,
            capture_output=True,
        )
        for line in proc.stderr.splitlines():
            if line.startswith("job "):
                job_id = line.split()[1]
                AT_JOB_ID_FILE.parent.mkdir(parents=True, exist_ok=True)
                AT_JOB_ID_FILE.write_text(job_id)
                break
    except Exception as e:
        logger.debug("timer.schedule_auto_finish_failed", error=str(e))


def _cancel_auto_finish() -> None:
    """Cancel scheduled at-job if active (idempotent)."""
    job_file = AT_JOB_ID_FILE if AT_JOB_ID_FILE.exists() else LEGACY_AT_JOB_ID_FILE
    if job_file.exists():
        try:
            job_id = job_file.read_text().strip()
            subprocess.run(["atrm", job_id], capture_output=True)
        except Exception:
            pass
        try:
            job_file.unlink(missing_ok=True)
        except Exception:
            pass


def send_notification(
    title: str,
    message: str,
    sound: bool = False,
    *,
    execute: Optional[str] = None,
    open_url: Optional[str] = None,
) -> bool:
    """Send macOS notification via osascript display notification.

    Args:
        title: Notification title
        message: Notification body
        sound: Whether to play sound (default False — silent)

    Returns:
        True if notification sent successfully, False otherwise
    """
    try:
        if _has_terminal_notifier():
            cmd = [
                "terminal-notifier",
                "-title", title,
                "-message", message,
                "-activate", "com.apple.Terminal",
                "-execute", execute or "pb",
            ]
            if open_url:
                cmd.extend(["-open", open_url])
            if sound:
                cmd.extend(["-sound", "default"])
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0

        escaped_title = title.replace('\\', '\\\\').replace('"', '\\"')
        escaped_message = message.replace('\\', '\\\\').replace('"', '\\"')

        if sound:
            script = f'display notification "{escaped_message}" with title "{escaped_title}" sound name "default"'
        else:
            script = f'display notification "{escaped_message}" with title "{escaped_title}"'

        # Pass script via stdin to avoid shell metacharacter injection
        result = subprocess.run(
            ["osascript"],
            input=script,
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except Exception:
        return False


class CaffeinateManager:
    """Prevent system sleep during active sessions.

    Per D-13: Caffeinate integration - prevent sleep during active session only
    (paused = sleep allowed)
    """

    def __init__(self) -> None:
        self.process: Optional[subprocess.Popen] = None

    def start(self) -> bool:
        """Start caffeinate to prevent idle sleep.

        Uses: caffeinate -i
        -i: prevent idle sleep (runs independently of this process)

        Writes the child PID to CAFFEINATE_PID_FILE so that a later
        invocation (`pb pause`, `pb finish`) can stop caffeinate even though
        the original `pb start` process has already exited.

        Returns:
            True if started successfully, False otherwise
        """
        if self.process is not None:
            return True

        try:
            self.process = subprocess.Popen(
                ["caffeinate", "-i"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            try:
                CAFFEINATE_PID_FILE.parent.mkdir(parents=True, exist_ok=True)
                CAFFEINATE_PID_FILE.write_text(str(self.process.pid))
            except Exception:
                pass
            return True
        except Exception:
            return False

    def stop(self) -> None:
        """Stop caffeinate and allow system to sleep again.

        If self.process is set (same-process stop): terminate directly.
        If self.process is None (cross-process stop): read the PID file,
        send SIGTERM, then delete the file.
        """
        if self.process is not None:
            try:
                self.process.terminate()
            except Exception:
                pass
            self.process = None
        elif CAFFEINATE_PID_FILE.exists() or LEGACY_CAFFEINATE_PID_FILE.exists():
            try:
                pid_file = CAFFEINATE_PID_FILE if CAFFEINATE_PID_FILE.exists() else LEGACY_CAFFEINATE_PID_FILE
                pid = int(pid_file.read_text().strip())
                os.kill(pid, signal.SIGTERM)
            except (ProcessLookupError, ValueError, OSError):
                pass
        try:
            CAFFEINATE_PID_FILE.unlink(missing_ok=True)
            LEGACY_CAFFEINATE_PID_FILE.unlink(missing_ok=True)
        except Exception:
            pass

    @property
    def is_active(self) -> bool:
        """Check if caffeinate is currently running."""
        if self.process is not None:
            return self.process.poll() is None
        pid_file = CAFFEINATE_PID_FILE if CAFFEINATE_PID_FILE.exists() else LEGACY_CAFFEINATE_PID_FILE
        if not pid_file.exists():
            return False
        try:
            pid = int(pid_file.read_text().strip())
            os.kill(pid, 0)
            return True
        except (ProcessLookupError, ValueError, OSError):
            try:
                pid_file.unlink(missing_ok=True)
            except Exception:
                pass
            return False


class BreakReminder:
    """Sends a single upfront break-cadence notification on session start.

    Daemon-thread approach was abandoned because pb start exits immediately
    after setup — daemon threads die with the process, so a 30-minute
    reminder scheduled via a background timer could never fire. Instead,
    a single notification is sent at session start reminding the user of
    the break cadence.

    Per D-06: 30-minute break interval constant retained for reference.
    """

    INTERVAL_MINUTES = 30

    def __init__(self) -> None:
        self.running: bool = False

    def start(self) -> None:
        """Send a single break-cadence notification and mark as running."""
        send_notification(
            "Focus session started",
            "Remember to take breaks every 30 min",
        )
        self.running = True

    def stop(self) -> None:
        """Mark reminder as stopped (no thread to cancel)."""
        self.running = False


@dataclass
class TimerState:
    """Active timer state for a session."""

    session_id: str
    start_time: datetime
    duration_minutes: Optional[int]
    task_title: str


class TimerManager:
    """Manages session timers, break reminders, and caffeinate.

    Integration point: SessionManager will call start_session_timers() on session start
    and stop_session_timers() on pause/finish.
    """

    def __init__(self) -> None:
        self.state: Optional[TimerState] = None
        self.break_reminder: BreakReminder = BreakReminder()
        self.caffeinate: CaffeinateManager = CaffeinateManager()

    def start_session_timers(
        self,
        session_id: str,
        duration_minutes: Optional[int],
        task_title: str,
    ) -> None:
        """Start timers for a new session.

        Session timing is recorded in-process and via the persisted session row,
        but default learning flows intentionally avoid OS-level automation hooks.
        That means no notifications, no scheduled at-jobs, and no caffeinate
        side effects on session start.
        """
        self.stop_session_timers()

        self.state = TimerState(
            session_id=session_id,
            start_time=datetime.now(),
            duration_minutes=duration_minutes,
            task_title=task_title,
        )

    def stop_session_timers(self) -> None:
        """Stop all timers and allow sleep.

        Legacy scheduled jobs are still cancelled defensively in case they were
        created by an older build before the prompt-first session model.
        """
        _cancel_auto_finish()  # D-11: cancel scheduled auto-trigger
        self.break_reminder.stop()
        self.caffeinate.stop()
        self.state = None

    @staticmethod
    def write_expired_flag() -> None:
        """Write timer_expired.flag. Called by at-job script when timer elapses (D-14).

        Also called explicitly if needed for testing. The next pb command will
        intercept this flag and show the expiry picker before executing.
        """
        TIMER_EXPIRED_FLAG.parent.mkdir(parents=True, exist_ok=True)
        TIMER_EXPIRED_FLAG.touch()

    def get_elapsed_minutes(self) -> Optional[int]:
        """Get elapsed minutes since session start (per D-11)."""
        if self.state is None:
            return None
        elapsed = datetime.now() - self.state.start_time
        return int(elapsed.total_seconds() / 60)

    def get_remaining_minutes(self) -> Optional[int]:
        """Get remaining minutes if duration set (per D-11)."""
        if self.state is None or self.state.duration_minutes is None:
            return None
        elapsed = self.get_elapsed_minutes()
        if elapsed is None:
            return None
        remaining = self.state.duration_minutes - elapsed
        return max(0, remaining)
