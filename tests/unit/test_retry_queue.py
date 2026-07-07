"""Tests for RetryQueueWriter CRUD operations.

Per Phase 2 Plan 02 TEST-02.
Pattern follows tests/unit/test_session_log_writer.py.
All tests are deterministic: no LLM calls, no network, SQLite in tmp_path only.
"""

from datetime import datetime, timedelta


# -- Helpers -----------------------------------------------------------------


def _init_db(tmp_path):
    """Initialize a fresh SQLite DB in tmp_path and return the path."""
    from pb.storage.database import init_db, set_db_path

    db_path = tmp_path / "test.db"
    set_db_path(db_path)
    init_db(db_path)
    return db_path


# -- Tests -------------------------------------------------------------------


class TestRetryQueueEnqueue:
    """Enqueue operations: single item and batch from assessment."""

    def test_enqueue_returns_item_id(self, tmp_path):
        from pb.core.retry_queue import RetryQueueWriter

        _init_db(tmp_path)
        writer = RetryQueueWriter()
        item_id = writer.enqueue(
            domain="math",
            item_text="Review eigenvalues",
            source="manual",
            priority=3,
        )
        assert item_id is not None
        assert isinstance(item_id, str)
        assert len(item_id) > 0

    def test_enqueue_stores_in_database(self, tmp_path):
        from pb.core.retry_queue import RetryQueueWriter
        from pb.storage.database import get_connection

        _init_db(tmp_path)
        writer = RetryQueueWriter()
        item_id = writer.enqueue(
            domain="math",
            item_text="Review eigenvalues",
            source="manual",
            priority=3,
        )
        assert item_id is not None

        with get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM retry_queue WHERE id = ?", (item_id,)
            ).fetchall()

        assert len(rows) == 1
        row = dict(rows[0])
        assert row["domain"] == "math"
        assert row["item_text"] == "Review eigenvalues"
        assert row["source"] == "manual"
        assert row["priority"] == 3
        assert row["status"] == "pending"

    def test_enqueue_from_assessment_creates_multiple(self, tmp_path):
        from pb.core.retry_queue import RetryQueueWriter
        from pb.storage.database import get_connection

        _init_db(tmp_path)
        writer = RetryQueueWriter()
        ids = writer.enqueue_from_assessment(
            domain="math",
            retry_items=["item1", "item2", "item3"],
            evidence_id="ev-001",
        )

        assert len(ids) == 3
        assert all(isinstance(i, str) and len(i) > 0 for i in ids)

        with get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM retry_queue WHERE evidence_id = ?", ("ev-001",)
            ).fetchall()

        assert len(rows) == 3
        for row in rows:
            r = dict(row)
            assert r["source"] == "assessment"
            assert r["priority"] == 1
            assert r["evidence_id"] == "ev-001"


class TestRetryQueueResolve:
    """Resolve operations: marks items resolved or handles missing IDs."""

    def test_resolve_marks_item_resolved(self, tmp_path):
        from pb.core.retry_queue import RetryQueueWriter
        from pb.storage.database import get_connection

        _init_db(tmp_path)
        writer = RetryQueueWriter()
        item_id = writer.enqueue(domain="math", item_text="Resolve me", source="manual")
        assert item_id is not None

        result = writer.resolve(item_id)
        assert result is True

        with get_connection() as conn:
            row = conn.execute(
                "SELECT status FROM retry_queue WHERE id = ?", (item_id,)
            ).fetchone()
        assert row is not None
        assert row[0] == "resolved"

    def test_resolve_nonexistent_returns_false(self, tmp_path):
        from pb.core.retry_queue import RetryQueueWriter

        _init_db(tmp_path)
        writer = RetryQueueWriter()
        result = writer.resolve("nonexistent-id-that-does-not-exist")
        assert result is False


class TestRetryQueueReschedule:
    """Reschedule operations: set cooldown date or default to tomorrow."""

    def test_reschedule_sets_cooldown_date(self, tmp_path):
        from pb.core.retry_queue import RetryQueueWriter
        from pb.storage.database import get_connection

        _init_db(tmp_path)
        writer = RetryQueueWriter()
        item_id = writer.enqueue(domain="math", item_text="Reschedule me", source="manual")
        assert item_id is not None

        result = writer.reschedule(item_id, "2026-06-01")
        assert result is True

        with get_connection() as conn:
            row = conn.execute(
                "SELECT cooldown_until FROM retry_queue WHERE id = ?", (item_id,)
            ).fetchone()
        assert row is not None
        assert row[0] == "2026-06-01"

    def test_reschedule_defaults_to_tomorrow(self, tmp_path):
        from pb.core.retry_queue import RetryQueueWriter
        from pb.storage.database import get_connection

        _init_db(tmp_path)
        writer = RetryQueueWriter()
        item_id = writer.enqueue(domain="math", item_text="Default cooldown", source="manual")
        assert item_id is not None

        result = writer.reschedule(item_id)  # no date -- defaults to tomorrow
        assert result is True

        tomorrow = (datetime.utcnow() + timedelta(days=1)).strftime("%Y-%m-%d")
        with get_connection() as conn:
            row = conn.execute(
                "SELECT cooldown_until FROM retry_queue WHERE id = ?", (item_id,)
            ).fetchone()
        assert row is not None
        assert row[0] == tomorrow

    def test_reschedule_nonexistent_returns_false(self, tmp_path):
        from pb.core.retry_queue import RetryQueueWriter

        _init_db(tmp_path)
        writer = RetryQueueWriter()
        result = writer.reschedule("nonexistent-id-reschedule", "2026-06-01")
        assert result is False


class TestRetryQueueListPending:
    """list_pending() with domain filter, cooldown, and resolved exclusion."""

    def test_list_pending_returns_items(self, tmp_path):
        from pb.core.retry_queue import RetryQueueWriter

        _init_db(tmp_path)
        writer = RetryQueueWriter()
        writer.enqueue(domain="math", item_text="item A", source="manual")
        writer.enqueue(domain="math", item_text="item B", source="manual")
        writer.enqueue(domain="math", item_text="item C", source="manual")

        items = writer.list_pending()
        assert len(items) == 3

    def test_list_pending_filters_by_domain(self, tmp_path):
        from pb.core.retry_queue import RetryQueueWriter

        _init_db(tmp_path)
        writer = RetryQueueWriter()
        writer.enqueue(domain="math", item_text="math item 1", source="manual")
        writer.enqueue(domain="math", item_text="math item 2", source="manual")
        writer.enqueue(domain="german", item_text="german item 1", source="manual")

        math_items = writer.list_pending(domain="math")
        assert len(math_items) == 2
        for item in math_items:
            assert item.domain == "math"

        german_items = writer.list_pending(domain="german")
        assert len(german_items) == 1
        assert german_items[0].domain == "german"

    def test_list_pending_excludes_resolved(self, tmp_path):
        from pb.core.retry_queue import RetryQueueWriter

        _init_db(tmp_path)
        writer = RetryQueueWriter()
        id1 = writer.enqueue(domain="math", item_text="keep this", source="manual")
        id2 = writer.enqueue(domain="math", item_text="resolve this", source="manual")
        assert id2 is not None
        writer.resolve(id2)

        items = writer.list_pending()
        assert len(items) == 1
        assert items[0].item_text == "keep this"

    def test_list_pending_respects_cooldown(self, tmp_path):
        from pb.core.retry_queue import RetryQueueWriter

        _init_db(tmp_path)
        writer = RetryQueueWriter()
        item_id = writer.enqueue(domain="math", item_text="cooldown item", source="manual")
        assert item_id is not None

        # Reschedule to far future
        writer.reschedule(item_id, "2099-01-01")

        # Item should NOT be visible when today is before cooldown
        items_before = writer.list_pending(today="2026-05-24")
        assert all(i.id != item_id for i in items_before), (
            "Item should be suppressed by cooldown"
        )

        # Item SHOULD be visible after cooldown date passes
        items_after = writer.list_pending(today="2099-01-02")
        item_ids_after = [i.id for i in items_after]
        assert item_id in item_ids_after, (
            "Item should appear once cooldown has passed"
        )


class TestRetryQueuePriority:
    """list_pending() returns items ordered by priority ASC."""

    def test_list_pending_orders_by_priority(self, tmp_path):
        from pb.core.retry_queue import RetryQueueWriter

        _init_db(tmp_path)
        writer = RetryQueueWriter()
        writer.enqueue(domain="math", item_text="priority 3 item", source="manual", priority=3)
        writer.enqueue(domain="math", item_text="priority 1 item", source="assessment", priority=1)
        writer.enqueue(domain="math", item_text="priority 2 item", source="incomplete", priority=2)

        items = writer.list_pending(domain="math")
        assert len(items) == 3
        assert items[0].priority == 1
        assert items[1].priority == 2
        assert items[2].priority == 3
