"""PTY-backed stdin tests for the bare `pb` and `brain` REPL entrypoints."""

from __future__ import annotations

import os
import pty
import re
import select
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from typer.testing import CliRunner

from pb.cli.main import app
from pb.core.models import Session, Task
from pb.runtime import build_runtime_context
from pb.storage.database import init_db, set_db_path
from pb.storage.repository import Repository


runner = CliRunner()
ROOT = Path(__file__).resolve().parents[2]
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
ENTRY_SCRIPT = (
    "import sys\n"
    "from typer.main import get_command\n"
    "from pb.cli.main import app\n"
    "get_command(app).main(args=sys.argv[2:], prog_name=sys.argv[1], standalone_mode=False)\n"
)


def _strip_ansi(text: str) -> str:
    text = ANSI_RE.sub("", text)
    return text.replace("\r", "")


def _init_runtime(tmp_path: Path) -> tuple[Path, Path]:
    config_path = tmp_path / "config.toml"
    vault_path = tmp_path / "vault"
    env = {
        "PRODUCTIVEBRAIN_CONFIG_PATH": str(config_path),
        "HOME": str(tmp_path),
    }
    result = runner.invoke(
        app,
        [
            "init",
            "--non-interactive",
            "--vault-name",
            "main",
            "--vault-path",
            str(vault_path),
            "--provider",
            "gemini",
            "--model",
            "gemini-3-flash-preview",
            "--yes",
        ],
        env=env,
    )
    assert result.exit_code == 0, result.output
    return config_path, vault_path


def _seed_active_session(tmp_path: Path, config_path: Path) -> None:
    old_config = os.environ.get("PRODUCTIVEBRAIN_CONFIG_PATH")
    old_home = os.environ.get("HOME")
    try:
        os.environ["PRODUCTIVEBRAIN_CONFIG_PATH"] = str(config_path)
        os.environ["HOME"] = str(tmp_path)
        runtime = build_runtime_context(config_path=config_path, yes=True, force_reload=True)
        set_db_path(runtime.db_path)
        init_db(runtime.db_path)
        repo = Repository()
        task = Task(title="German conjugation recovery", work_type="study")
        repo.create_task(task)
        repo.create_session(
            Session(
                task_id=task.id,
                branch="study",
                subject_scope="German conjugation",
                start_at=datetime.utcnow(),
            )
        )
    finally:
        if old_config is None:
            os.environ.pop("PRODUCTIVEBRAIN_CONFIG_PATH", None)
        else:
            os.environ["PRODUCTIVEBRAIN_CONFIG_PATH"] = old_config
        if old_home is None:
            os.environ.pop("HOME", None)
        else:
            os.environ["HOME"] = old_home


def _run_repl(
    tmp_path: Path,
    *,
    prog_name: str,
    stdin_text: str = "",
    stdin_chunks: list[str] | None = None,
    stdin_steps: list[tuple[str, str | None]] | None = None,
    shell_test_mode: bool = True,
    seed_active_session: bool = False,
) -> tuple[int, str]:
    config_path, _vault_path = _init_runtime(tmp_path)
    if seed_active_session:
        _seed_active_session(tmp_path, config_path)
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(ROOT / "src"),
            "PRODUCTIVEBRAIN_CONFIG_PATH": str(config_path),
            "HOME": str(tmp_path),
        }
    )
    if shell_test_mode:
        env["PRODUCTIVEBRAIN_SHELL_TEST_MODE"] = "1"
    else:
        env.pop("PRODUCTIVEBRAIN_SHELL_TEST_MODE", None)

    master_fd, slave_fd = pty.openpty()
    process = subprocess.Popen(
        [sys.executable, "-c", ENTRY_SCRIPT, prog_name],
        cwd=ROOT,
        env=env,
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        close_fds=True,
    )
    os.close(slave_fd)

    def _drain_output(output: bytearray) -> None:
        while True:
            ready, _, _ = select.select([master_fd], [], [], 0.1)
            if not ready:
                break
            chunk = os.read(master_fd, 4096)
            if not chunk:
                break
            output.extend(chunk)

    def _wait_for(output: bytearray, pattern: str | None, timeout: float = 12.0) -> None:
        if pattern is None:
            time.sleep(1.0)
            _drain_output(output)
            return
        deadline = time.time() + timeout
        while time.time() < deadline:
            _drain_output(output)
            cleaned = _strip_ansi(output.decode("utf-8", errors="ignore"))
            if pattern in cleaned:
                time.sleep(0.6)
                _drain_output(output)
                return
            if process.poll() is not None:
                return
            time.sleep(0.05)

    output = bytearray()
    try:
        time.sleep(0.5)
        if stdin_steps is not None:
            for chunk, pattern in stdin_steps:
                if chunk:
                    os.write(master_fd, chunk.encode("utf-8"))
                _wait_for(output, pattern)
        else:
            chunks = stdin_chunks if stdin_chunks is not None else [stdin_text]
            for chunk in chunks:
                if chunk:
                    os.write(master_fd, chunk.encode("utf-8"))
                time.sleep(0.5)
                _drain_output(output)

        deadline = time.time() + 10
        while time.time() < deadline:
            ready, _, _ = select.select([master_fd], [], [], 0.1)
            if ready:
                chunk = os.read(master_fd, 4096)
                if chunk:
                    output.extend(chunk)
                continue
            if process.poll() is not None:
                break
        try:
            process.wait(timeout=6)
        except subprocess.TimeoutExpired:
            process.kill()
            _drain_output(output)
            process.wait(timeout=1)
    finally:
        os.close(master_fd)

    return process.returncode, output.decode("utf-8", errors="ignore")


def _strip_ansi(text: str) -> str:
    import re
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def test_pb_repl_accepts_exit(tmp_path: Path) -> None:
    returncode, output = _run_repl(tmp_path, prog_name="pb", stdin_text="exit\n")
    clean = _strip_ansi(output)

    assert returncode == 0, output
    assert "pb shell" in clean
    assert "exit=exit|quit|Ctrl-D" in clean
    assert "Traceback" not in output


def test_brain_alias_opens_same_repl(tmp_path: Path) -> None:
    returncode, output = _run_repl(tmp_path, prog_name="brain", stdin_text="exit\n")

    assert returncode == 0, output
    assert "pb shell" in output
    assert "Traceback" not in output


def test_repl_routes_free_text_without_crashing(tmp_path: Path) -> None:
    returncode, output = _run_repl(
        tmp_path,
        prog_name="pb",
        stdin_text="I need to revise eigenvalues tomorrow\nexit\n",
    )

    assert returncode == 0, output
    assert "pb shell" in output
    assert "Do" in output
    assert (
        "Capture a quick thought" in output
        or "Capture an upcoming task" in output
        or "Study " in output
    )
    assert "Traceback" not in output


def test_repl_handles_next_style_question(tmp_path: Path) -> None:
    returncode, output = _run_repl(
        tmp_path,
        prog_name="pb",
        stdin_text="what should I do next?\nexit\n",
    )

    assert returncode == 0, output
    assert "pb shell" in output
    assert "pb " in output
    assert "Traceback" not in output


def test_pb_startup_picker_accepts_number_key(tmp_path: Path) -> None:
    returncode, output = _run_repl(
        tmp_path,
        prog_name="pb",
        stdin_steps=[
            ("", "Start here"),
            ("\n", "pb>"),
            ("\x04", None),
        ],
        shell_test_mode=False,
    )

    clean = _strip_ansi(output)
    assert "Start here" in clean
    assert "pb>" in clean
    assert "Traceback" not in clean


def test_pb_startup_picker_q_cancels_cleanly(tmp_path: Path) -> None:
    returncode, output = _run_repl(
        tmp_path,
        prog_name="pb",
        stdin_steps=[
            ("q", "pb>"),
            ("\x04", None),
        ],
        shell_test_mode=False,
    )

    clean = _strip_ansi(output)
    assert "Start here" in clean
    assert "pb>" in clean
    assert "Traceback" not in clean


def test_next_picker_selection_still_works_after_nested_command(tmp_path: Path) -> None:
    returncode, output = _run_repl(
        tmp_path,
        prog_name="pb",
        stdin_steps=[
            ("1", "pb ["),
            ("ls\n", "pb ["),
            ("next\n", "Choose next direction"),
            ("2", "pb session list"),
            ("\x04", None),
        ],
        shell_test_mode=False,
        seed_active_session=True,
    )

    clean = _strip_ansi(output)
    assert returncode == 0, clean
    assert "Choose next direction" in clean
    assert "pb session list" in clean
    assert "Traceback" not in clean


def test_active_session_shell_routes_free_text_to_coaching_and_blocks_new_learning_commands(tmp_path: Path) -> None:
    returncode, output = _run_repl(
        tmp_path,
        prog_name="pb",
        stdin_steps=[
            ("", "pb ["),
            ("how should i approach this\n", "Give one worked example."),
            ("learn linear algebra\n", "pb finish --skip"),
            ("\x04", None),
        ],
        shell_test_mode=True,
        seed_active_session=True,
    )

    clean = _strip_ansi(output)
    assert returncode == 0, clean
    lowered = clean.lower()
    assert "core idea behind" in lowered
    assert "worked example" in lowered
    assert "Session active: German conjugation recovery." in clean
    assert "pb finish --skip" in clean
    assert "pb pause" in clean
    assert "Active session conflict" not in clean
    assert "You already have an active session:" not in clean
    assert "Resolve the active session before starting something new." not in clean
    assert '{"error":' not in clean
    assert '"event":' not in clean
    assert "Traceback" not in clean
