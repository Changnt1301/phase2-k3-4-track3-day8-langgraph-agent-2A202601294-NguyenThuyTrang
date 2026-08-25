"""Checkpointer adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver


def build_checkpointer(
    kind: str = "memory", database_url: str | None = None
) -> BaseCheckpointSaver | None:
    """Return a LangGraph checkpointer.

    - "none": no persistence.
    - "memory": in-process MemorySaver (lost when the process exits).
    - "sqlite": SqliteSaver backed by a file on disk, so state survives a process
      restart. thread_id (one per scenario) scopes each run's checkpoint history.
    - "postgres": optional extension, not implemented.
    """
    if kind == "none":
        return None
    if kind == "memory":
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()
    if kind == "sqlite":
        import sqlite3

        from langgraph.checkpoint.sqlite import SqliteSaver

        db_path = database_url or "checkpoints.db"
        if db_path.startswith("sqlite:///"):
            db_path = db_path[len("sqlite:///") :]

        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        saver = SqliteSaver(conn=conn)
        saver.setup()
        return saver
    if kind == "postgres":
        raise NotImplementedError(
            "TODO(student): implement Postgres checkpointer (optional extension)"
        )
    raise ValueError(f"Unknown checkpointer kind: {kind}")
