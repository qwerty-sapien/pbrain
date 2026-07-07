"""Tests for the orchestrator dry-run mode and basic module contracts.

Verifies:
- `python -m swarm --dry-run` exits 0 and lists personas (no LLM call)
- orchestrator.py defines run_persona and run_swarm
- orchestrator composes all required modules (BrainSession, run_parity_probes,
  check_mcp_server, EvidenceWriter, build_agent_env, write_context_packets)
- The brain client is built from os.environ, not agent_env
- orchestrator does NOT contain get_vault_path or LLMRuntime (SWARM-03, Pitfall 1)
"""

from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SWARM_ROOT = REPO_ROOT / "swarm"
if not SWARM_ROOT.exists():
    SWARM_ROOT = REPO_ROOT.parent / "swarm"
SWARM_CWD = SWARM_ROOT.parent
SWARM_MODULE = "swarm.swarm"
if str(SWARM_CWD) not in sys.path:
    sys.path.insert(0, str(SWARM_CWD))
PERSONAS_DIR = REPO_ROOT / "personas"


# ---------------------------------------------------------------------------
# Dry-run: subprocess test (no LLM call guarantee)
# ---------------------------------------------------------------------------

def test_dry_run_exits_zero():
    """--dry-run exits 0 and lists at least one persona."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            SWARM_MODULE,
            "--dry-run",
            "--personas-dir",
            str(PERSONAS_DIR),
        ],
        capture_output=True,
        text=True,
        cwd=str(SWARM_CWD),
        timeout=30,
    )
    assert result.returncode == 0, (
        f"--dry-run exited {result.returncode}.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    # Must mention at least one persona
    assert "Would run" in result.stdout, (
        f"Expected 'Would run' in dry-run output.\nstdout: {result.stdout}"
    )


def test_dry_run_lists_personas():
    """--dry-run lists all available personas from the personas/ directory."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            SWARM_MODULE,
            "--dry-run",
            "--personas-dir",
            str(PERSONAS_DIR),
        ],
        capture_output=True,
        text=True,
        cwd=str(SWARM_CWD),
        timeout=30,
    )
    assert result.returncode == 0, f"--dry-run failed: {result.stderr}"
    # At least the two smoke personas must appear
    assert "overwhelmed-planner" in result.stdout
    assert "german-learner" in result.stdout


def test_dry_run_no_llm_import_path():
    """Verify --dry-run stdout contains no brain API call evidence.

    The dry-run path must NOT construct a brain client. We check this by
    asserting the output doesn't contain typical LLM call traces (e.g.
    'generate_with_model' errors). The test cannot intercept network calls
    easily in subprocess mode, but the architectural guarantee is that
    _load_personas + dry-run print exits before any orchestrator.run_swarm
    call — and therefore before _make_brain_client is called.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            SWARM_MODULE,
            "--dry-run",
            "--personas-dir",
            str(PERSONAS_DIR),
        ],
        capture_output=True,
        text=True,
        cwd=str(SWARM_CWD),
        timeout=30,
    )
    assert result.returncode == 0
    # No LLM error traces should appear
    assert "generate_with_model" not in result.stdout
    assert "generate_with_model" not in result.stderr
    assert "brain LLM error" not in result.stdout


# ---------------------------------------------------------------------------
# Orchestrator module contracts (static / import-time checks)
# ---------------------------------------------------------------------------

def test_orchestrator_defines_run_persona():
    """orchestrator.run_persona is defined and callable."""
    from swarm.swarm import orchestrator
    assert hasattr(orchestrator, "run_persona"), "orchestrator.run_persona missing"
    assert callable(orchestrator.run_persona)


def test_orchestrator_defines_run_swarm():
    """orchestrator.run_swarm is defined and callable."""
    from swarm.swarm import orchestrator
    assert hasattr(orchestrator, "run_swarm"), "orchestrator.run_swarm missing"
    assert callable(orchestrator.run_swarm)


def test_orchestrator_source_contains_brain_session():
    """orchestrator.py source references BrainSession (adaptive loop composed)."""
    orchestrator_path = SWARM_ROOT / "swarm" / "orchestrator.py"
    source = orchestrator_path.read_text()
    assert "BrainSession" in source, "orchestrator.py must reference BrainSession"


def test_orchestrator_source_contains_run_parity_probes():
    """orchestrator.py source references run_parity_probes (D-20/D-21 end-of-session)."""
    orchestrator_path = SWARM_ROOT / "swarm" / "orchestrator.py"
    source = orchestrator_path.read_text()
    assert "run_parity_probes" in source


def test_orchestrator_source_contains_check_mcp_server():
    """orchestrator.py source references check_mcp_server (MCP parity)."""
    orchestrator_path = SWARM_ROOT / "swarm" / "orchestrator.py"
    source = orchestrator_path.read_text()
    assert "check_mcp_server" in source


def test_orchestrator_source_contains_evidence_writer():
    """orchestrator.py source references EvidenceWriter (bundle writing)."""
    orchestrator_path = SWARM_ROOT / "swarm" / "orchestrator.py"
    source = orchestrator_path.read_text()
    assert "EvidenceWriter" in source


def test_orchestrator_source_contains_build_agent_env():
    """orchestrator.py source references build_agent_env (isolation)."""
    orchestrator_path = SWARM_ROOT / "swarm" / "orchestrator.py"
    source = orchestrator_path.read_text()
    assert "build_agent_env" in source


def test_orchestrator_source_contains_write_context_packets():
    """orchestrator.py source references write_context_packets (D-19 bundle)."""
    orchestrator_path = SWARM_ROOT / "swarm" / "orchestrator.py"
    source = orchestrator_path.read_text()
    assert "write_context_packets" in source


def test_orchestrator_no_get_vault_path():
    """orchestrator.py must NOT call get_vault_path (SWARM-03 — no singleton access).

    Checks for actual call invocations, not comments/docstrings mentioning it.
    """
    orchestrator_path = SWARM_ROOT / "swarm" / "orchestrator.py"
    source = orchestrator_path.read_text()
    # Filter out comments and docstring lines (lines starting with # or containing only
    # docstring prose). Look for actual import or call of get_vault_path.
    code_lines = [
        line for line in source.splitlines()
        if not line.strip().startswith("#") and not line.strip().startswith('"""') and not line.strip().startswith("'")
    ]
    code_only = "\n".join(code_lines)
    assert "get_vault_path" not in code_only, (
        "orchestrator.py must not call get_vault_path — reads real user config (SWARM-03)"
    )


def test_orchestrator_no_llm_runtime():
    """orchestrator.py must NOT instantiate LLMRuntime (Pitfall 1 — reads real user config)."""
    orchestrator_path = SWARM_ROOT / "swarm" / "orchestrator.py"
    source = orchestrator_path.read_text()
    assert "LLMRuntime(" not in source, (
        "orchestrator.py must not instantiate LLMRuntime — use concrete client classes"
    )


def test_make_brain_client_not_passed_agent_env():
    """_make_brain_client is called WITHOUT agent_env (os.environ only — Pitfall 4).

    Checks the orchestrator source to confirm _make_brain_client is called
    before any agent_env variable is passed to it.
    """
    orchestrator_path = SWARM_ROOT / "swarm" / "orchestrator.py"
    source = orchestrator_path.read_text()
    assert "_make_brain_client" in source, "_make_brain_client not called"
    # agent_env must be passed only to BrainSession / run_cli_command / check_mcp_server,
    # not to _make_brain_client. Check the call site is just (_make_brain_client(provider))
    # not _make_brain_client(agent_env, ...).
    lines = source.splitlines()
    for line in lines:
        if "_make_brain_client" in line and "agent_env" in line:
            pytest.fail(
                f"agent_env appears on same line as _make_brain_client: {line!r}\n"
                "Brain client must be built from os.environ only (Pitfall 4)"
            )


# ---------------------------------------------------------------------------
# __main__.py contracts (static checks)
# ---------------------------------------------------------------------------

def test_main_defines_argparse():
    """__main__.py uses argparse."""
    main_path = SWARM_ROOT / "swarm" / "__main__.py"
    source = main_path.read_text()
    assert "argparse" in source


def test_main_defines_dry_run_flag():
    """__main__.py defines the --dry-run flag."""
    main_path = SWARM_ROOT / "swarm" / "__main__.py"
    source = main_path.read_text()
    assert '"--dry-run"' in source


def test_main_defines_persona_flag():
    """__main__.py defines the --persona flag."""
    main_path = SWARM_ROOT / "swarm" / "__main__.py"
    source = main_path.read_text()
    assert '"--persona"' in source


def test_main_defines_replay_flag():
    """__main__.py defines the --replay flag."""
    main_path = SWARM_ROOT / "swarm" / "__main__.py"
    source = main_path.read_text()
    assert '"--replay"' in source


# ---------------------------------------------------------------------------
# Bundle layout contract (D-19) — tested via a smoke of the dry-run persona list
# ---------------------------------------------------------------------------

def test_personas_dir_exists():
    """The personas/ directory exists and contains at least 10 .yaml files."""
    assert PERSONAS_DIR.exists(), f"personas/ directory not found at {PERSONAS_DIR}"
    yamls = list(PERSONAS_DIR.glob("*.yaml"))
    assert len(yamls) >= 10, (
        f"Expected >= 10 persona .yaml files, found {len(yamls)}: "
        f"{[p.name for p in yamls]}"
    )


def test_smoke_personas_load():
    """overwhelmed-planner and german-learner load without error."""
    from swarm.swarm.personas import load_persona
    for pid in ("overwhelmed-planner", "german-learner"):
        path = PERSONAS_DIR / f"{pid}.yaml"
        assert path.exists(), f"Missing required smoke persona: {path}"
        persona = load_persona(path)
        assert persona["id"] == pid
        assert persona.get("domain_tag")
        assert persona.get("win_condition")
        assert persona.get("anti_goal")


def test_write_context_packets_called_in_orchestrator():
    """orchestrator invokes write_context_packets for D-19 bundle completeness."""
    orchestrator_path = SWARM_ROOT / "swarm" / "orchestrator.py"
    source = orchestrator_path.read_text()
    # Must be called with env.data_dir argument (not just defined)
    assert "write_context_packets(env.data_dir)" in source, (
        "orchestrator.py must call writer.write_context_packets(env.data_dir) — D-19"
    )
