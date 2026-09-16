"""The boundary between the bot and whatever platform it posts to.

Everything upstream of this file talks to SocialDriver and never learns that X
exists. LocalDriver is a complete stand-in backed by SQLite: it records posts,
holds DM threads, and lets you converse with the persona through the local UI
long before any API credentials exist. XDriver drops in beside it later.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Protocol, runtime_checkable

PostId = str
ThreadId = str

INBOUND = "in"
OUTBOUND = "out"

SCHEMA = """
CREATE TABLE IF NOT EXISTS posts (
    id         TEXT PRIMARY KEY,
    image      TEXT NOT NULL,
    caption    TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id    TEXT NOT NULL,
    direction    TEXT NOT NULL CHECK (direction IN ('in', 'out')),
    body         TEXT NOT NULL,
    created_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS messages_thread ON messages (thread_id, id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class DM:
    id: int
    thread_id: ThreadId
    direction: str
    body: str
    created_at: str

    @property
    def inbound(self) -> bool:
        return self.direction == INBOUND


@dataclass(frozen=True)
class Post:
    id: PostId
    image: str
    caption: str
    created_at: str


@runtime_checkable
class SocialDriver(Protocol):
    """The whole surface the bot needs from a social platform.

    Runtime-checkable so a driver can be sanity-checked at startup; note that
    isinstance() verifies method *names* only, not signatures.
    """

    def post(self, image: Path, caption: str) -> PostId: ...

    def fetch_dms(self, since: int = 0) -> list[DM]:
        """Inbound messages with id > since, oldest first."""
        ...

    def send_dm(self, thread: ThreadId, text: str) -> None: ...

    def recent_posts(self, limit: int = 5) -> list[Post]:
        """Used to keep DM replies coherent with the timeline."""
        ...


class LocalDriver:
    """SQLite-backed SocialDriver. No network, no credentials, no rate limits."""

    def __init__(self, db_path: str | Path = "content/local.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "LocalDriver":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- SocialDriver ----------------------------------------------------

    def post(self, image: Path, caption: str) -> PostId:
        post_id = f"local-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"
        self._conn.execute(
            "INSERT INTO posts (id, image, caption, created_at) VALUES (?, ?, ?, ?)",
            (post_id, str(image), caption, _now()),
        )
        self._conn.commit()
        return post_id

    def fetch_dms(self, since: int = 0) -> list[DM]:
        rows = self._conn.execute(
            "SELECT * FROM messages WHERE direction = ? AND id > ? ORDER BY id",
            (INBOUND, since),
        ).fetchall()
        return [_as_dm(r) for r in rows]

    def send_dm(self, thread: ThreadId, text: str) -> None:
        self._append(thread, OUTBOUND, text)

    def recent_posts(self, limit: int = 5) -> list[Post]:
        rows = self._conn.execute(
            "SELECT * FROM posts ORDER BY created_at DESC, rowid DESC LIMIT ?", (limit,)
        ).fetchall()
        return [Post(r["id"], r["image"], r["caption"], r["created_at"]) for r in rows]

    # -- local-only helpers (no XDriver equivalent) ----------------------

    def receive_dm(self, thread: ThreadId, text: str) -> DM:
        """Simulate someone messaging the account. Drives the local UI."""
        return self._append(thread, INBOUND, text)

    def thread(self, thread: ThreadId, limit: int = 100) -> list[DM]:
        rows = self._conn.execute(
            "SELECT * FROM messages WHERE thread_id = ? ORDER BY id DESC LIMIT ?",
            (thread, limit),
        ).fetchall()
        return [_as_dm(r) for r in reversed(rows)]

    def threads(self) -> list[dict]:
        rows = self._conn.execute(
            """
            SELECT thread_id,
                   COUNT(*)        AS n,
                   MAX(created_at) AS last_at
            FROM messages GROUP BY thread_id ORDER BY last_at DESC
            """
        ).fetchall()
        return [dict(r) for r in rows]

    def timeline(self, limit: int = 50) -> list[Post]:
        return self.recent_posts(limit)

    def clear_thread(self, thread: ThreadId) -> None:
        """Start the conversation over. Used by the tuning UI's reset."""
        self._conn.execute("DELETE FROM messages WHERE thread_id = ?", (thread,))
        self._conn.commit()

    def _append(self, thread: ThreadId, direction: str, text: str) -> DM:
        cur = self._conn.execute(
            "INSERT INTO messages (thread_id, direction, body, created_at) VALUES (?, ?, ?, ?)",
            (thread, direction, text, _now()),
        )
        self._conn.commit()
        return DM(int(cur.lastrowid), thread, direction, text, _now())


def _as_dm(row: sqlite3.Row) -> DM:
    return DM(row["id"], row["thread_id"], row["direction"], row["body"], row["created_at"])


def seed_demo(driver: LocalDriver, captions: Iterable[str] = ()) -> None:
    """Put something in the timeline so the UI isn't empty on first run."""
    for i, caption in enumerate(captions):
        driver.post(Path(f"content/img/demo-{i:03d}.jpg"), caption)
