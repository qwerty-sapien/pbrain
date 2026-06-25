# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Launch the sibling swarm UX harness against this ProductiveBrain checkout.

The actual adaptive harness lives beside this repo at ``../swarm``. This script
keeps the ProductiveBrain-side invocation deterministic: it passes the repo's
persona roster, writes bundles under ``subagent_runs/``, and defaults to the
20-action user-flow sweep requested for UX audits.
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SWARM_PARENT = REPO_ROOT.parent
SWARM_ROOT = SWARM_PARENT / "swarm"
PERSONAS_DIR = REPO_ROOT / "personas"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "subagent_runs"
DEFAULT_MAX_ACTIONS = 20


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _base_swarm_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "swarm.swarm",
        "--personas-dir",
        str(args.personas_dir),
        "--output",
        str(args.output),
    ]

    if args.list:
        command.append("--list")
    elif args.dry_run:
        command.append("--dry-run")
    elif args.canonical:
        command.append("--canonical")
    elif args.persona:
        command.extend(["--persona", args.persona])
    else:
        command.append("--all")

    if not args.canonical:
        command.extend(["--max-actions", str(args.max_actions)])
    if args.report and not args.dry_run and not args.list:
        command.append("--report")
    return command


def _swarm_env() -> dict[str, str]:
    env = dict(os.environ)
    pythonpath = [str(SWARM_PARENT)]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    env.setdefault("PRODUCTIVEBRAIN_AUTO_YES", "1")
    return env


def _quote_env(env: dict[str, str]) -> str:
    keys = ["PYTHONPATH", "PRODUCTIVEBRAIN_AUTO_YES"]
    return " ".join(f"{key}={shlex.quote(env[key])}" for key in keys if key in env)


def _launch_tmux(command: list[str], env: dict[str, str], *, session: str, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    shell_command = (
        f"cd {shlex.quote(str(SWARM_PARENT))} && "
        f"{_quote_env(env)} "
        f"{' '.join(shlex.quote(part) for part in command)} "
        f"2>&1 | tee {shlex.quote(str(log_path))}"
    )
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", session, shell_command],
        check=True,
    )
    print(f"Started tmux session: {session}")
    print(f"Attach with: tmux attach -t {session}")
    print(f"Log: {log_path}")
    return 0


def _launch_nohup(command: list[str], env: dict[str, str], *, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log:
        proc = subprocess.Popen(
            command,
            cwd=str(SWARM_PARENT),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    print(f"Started detached swarm process: pid {proc.pid}")
    print(f"Log: {log_path}")
    return 0


def _run_foreground(command: list[str], env: dict[str, str]) -> int:
    completed = subprocess.run(command, cwd=str(SWARM_PARENT), env=env)
    return int(completed.returncode)


def _select_runner(args: argparse.Namespace) -> str:
    if args.runner != "auto":
        return args.runner
    if args.dry_run or args.list:
        return "foreground"
    return "tmux" if shutil.which("tmux") else "nohup"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--persona", help="Run a single persona id")
    mode.add_argument("--all", action="store_true", help="Run the full persona roster")
    mode.add_argument("--canonical", action="store_true", help="Run the swarm canonical preset")
    mode.add_argument("--dry-run", action="store_true", help="List would-run personas without LLM calls")
    mode.add_argument("--list", action="store_true", help="List available personas")
    parser.add_argument("--max-actions", type=int, default=DEFAULT_MAX_ACTIONS)
    parser.add_argument("--report", action="store_true", default=True)
    parser.add_argument("--no-report", action="store_false", dest="report")
    parser.add_argument("--personas-dir", type=Path, default=PERSONAS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--runner",
        choices=["auto", "tmux", "nohup", "foreground"],
        default="auto",
        help="Execution backend. auto uses foreground for dry/list and tmux for long runs when available.",
    )
    parser.add_argument("--session-name", default=f"pb-swarm-{_timestamp()}")
    parser.add_argument("--log", type=Path, default=None)
    parser.add_argument("--print-command", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not SWARM_ROOT.exists():
        print(f"ERROR: sibling swarm checkout not found: {SWARM_ROOT}", file=sys.stderr)
        return 1
    if not args.personas_dir.exists():
        print(f"ERROR: personas directory not found: {args.personas_dir}", file=sys.stderr)
        return 1

    command = _base_swarm_command(args)
    env = _swarm_env()
    runner = _select_runner(args)
    log_path = args.log or (args.output / "logs" / f"{args.session_name}.log")

    printable = " ".join(shlex.quote(part) for part in command)
    if args.print_command:
        print(f"cd {SWARM_PARENT} && {printable}", flush=True)

    if runner == "foreground":
        return _run_foreground(command, env)
    if runner == "tmux":
        if not shutil.which("tmux"):
            print("ERROR: tmux requested but not found on PATH", file=sys.stderr)
            return 1
        return _launch_tmux(command, env, session=args.session_name, log_path=log_path)
    return _launch_nohup(command, env, log_path=log_path)


if __name__ == "__main__":
    raise SystemExit(main())
