"""Test fixtures and configuration."""

import tempfile
from pathlib import Path

import pytest

from datetime import datetime, timedelta

from pb.domain.enums import Horizon, ProjectStatus, ProjectType, SessionMode, TaskState
from pb.domain.models import GoalArc, Project, Session, Task, Track
from pb.storage import config as config_module
from pb.storage import database as db_module
from pb.storage.config import Config, GeneralConfig, StorageConfig
from pb.storage.database import DB_FILENAME, init_db, set_db_path
from pb.storage.repository import Repository


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_db(temp_dir):
    """Create a temporary database."""
    db_path = temp_dir / DB_FILENAME
    set_db_path(db_path)
    init_db(db_path)
    yield db_path
    db_module._db_path = None


@pytest.fixture
def temp_config(temp_dir):
    """Create a temporary config."""
    vault_path = temp_dir / "vault"
    vault_path.mkdir()

    config = Config(
        general=GeneralConfig(vault_path=str(vault_path)),
        storage=StorageConfig(data_dir=str(temp_dir)),
    )
    config_module._config = config
    yield config
    config_module._config = None


@pytest.fixture
def repo(temp_db, temp_config):
    """Create a repository with temp database."""
    return Repository()


@pytest.fixture
def sample_task():
    """Create a sample task."""
    return Task(
        title="Test task",
        description="A test task",
        horizon=Horizon.TODAY,
        state=TaskState.ACTIVE,
    )


@pytest.fixture
def sample_project(temp_dir):
    """Create a sample project with packet path."""
    packet_path = temp_dir / "test_project.md"
    return Project(
        name="Test Project",
        project_type=ProjectType.BUILD,
        packet_path=str(packet_path),
        status=ProjectStatus.READY,
    )


@pytest.fixture
def sample_track():
    """Create a sample track."""
    return Track(
        name="Test Track",
        description="A test track",
    )


@pytest.fixture
def sample_goal():
    """Create a sample goal arc."""
    return GoalArc(
        title="Test Goal",
        description="A test goal",
        horizon=Horizon.SIX_MONTH,
    )


@pytest.fixture
def active_task(repo, sample_task):
    """Create a task in ready state (can be activated)."""
    sample_task.state = TaskState.ACTIVE
    return repo.create_task(sample_task)


@pytest.fixture
def sample_time_block(repo, sample_task):
    """Create a sample time block for testing. Requires sample_task to be persisted first."""
    from pb.domain.models import TimeBlock

    repo.create_task(sample_task)

    start_time = datetime.utcnow().replace(hour=9, minute=0, second=0, microsecond=0)
    return TimeBlock(
        task_id=sample_task.id,
        start_time=start_time,
        duration_minutes=60,
    )


@pytest.fixture
def waiting_project(repo, temp_dir):
    """Create a project in WAITING status."""
    packet_path = temp_dir / "waiting-project.md"
    packet_path.write_text("# Waiting Project")

    project = Project(
        name="Waiting Project",
        packet_path=str(packet_path),
        status=ProjectStatus.WAITING,
    )
    repo.create_project(project)
    return project


@pytest.fixture
def blocked_project(repo, temp_dir):
    """Create a project in BLOCKED status."""
    packet_path = temp_dir / "blocked-project.md"
    packet_path.write_text("# Blocked Project")

    project = Project(
        name="Blocked Project",
        packet_path=str(packet_path),
        status=ProjectStatus.BLOCKED,
    )
    repo.create_project(project)
    return project


@pytest.fixture
def sample_session(repo, sample_task):
    """Create a sample session for a task."""
    repo.create_task(sample_task)
    session = Session(
        task_id=sample_task.id,
        start_at=datetime.utcnow() - timedelta(hours=2),
        end_at=datetime.utcnow() - timedelta(hours=1),
        intended_outcome="Test session",
        mode=SessionMode.FLOW,
    )
    repo.create_session(session)
    return session
