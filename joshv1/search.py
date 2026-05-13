"""Notes search backed by SQLite FTS5.

No heavy ML deps — FTS5 is built into Python's stdlib sqlite3. It's full-text,
not literally semantic, but the agent itself supplies the semantic ranking
when you pipe top-K hits into a `joshv1 run` follow-up query.

CLI:
    joshv1 index <directory>       # walk and index *.md/*.txt files
    joshv1 search "query"          # return top matches
    joshv1 search "query" --json   # machine-readable
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

DB_PATH = Path.home() / ".josh-memory" / "notes.db"
TEXT_EXTS = {".md", ".txt", ".markdown", ".org", ".rst"}


@dataclass
class Hit:
    path: str
    score: float
    snippet: str


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS notes USING fts5("
        "path UNINDEXED, body, tokenize='porter unicode61')"
    )
    return conn


def index_directory(directory: Path | str, rebuild: bool = False) -> int:
    """Walk `directory`, index every text file. Returns number of files indexed."""
    directory = Path(directory).expanduser().resolve()
    conn = _connect()
    if rebuild:
        conn.execute("DELETE FROM notes")
    cursor = conn.cursor()

    count = 0
    for path in directory.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_EXTS:
            continue
        try:
            body = path.read_text(errors="ignore")
        except OSError:
            continue
        # Replace existing row for this path so re-indexing is idempotent.
        cursor.execute("DELETE FROM notes WHERE path = ?", (str(path),))
        cursor.execute("INSERT INTO notes(path, body) VALUES (?, ?)", (str(path), body))
        count += 1

    conn.commit()
    conn.close()
    return count


def search(query: str, limit: int = 10) -> list[Hit]:
    """FTS5 MATCH query. Returns hits with snippets."""
    if not DB_PATH.exists():
        return []
    conn = _connect()
    sanitized = _sanitize_query(query)
    try:
        rows = conn.execute(
            """
            SELECT path,
                   bm25(notes) AS score,
                   snippet(notes, 1, '[', ']', '...', 12) AS snip
            FROM notes
            WHERE notes MATCH ?
            ORDER BY score
            LIMIT ?
            """,
            (sanitized, limit),
        ).fetchall()
    except sqlite3.OperationalError as e:
        conn.close()
        raise ValueError(f"Bad search query: {e}") from e
    conn.close()
    # bm25 returns negative scores (lower = better). Flip sign for display.
    return [Hit(path=p, score=-s, snippet=snip) for p, s, snip in rows]


def _sanitize_query(query: str) -> str:
    """Make user queries FTS5-safe: split on whitespace, wrap each token in quotes."""
    tokens = [t for t in query.replace('"', "").split() if t]
    if not tokens:
        return '""'
    return " ".join(f'"{t}"' for t in tokens)


def render_hits(hits: list[Hit]) -> str:
    if not hits:
        return "(no matches)"
    lines = []
    for i, hit in enumerate(hits, 1):
        rel = hit.path
        lines.append(f"{i:>2}. [{hit.score:.2f}] {rel}\n     {hit.snippet}")
    return "\n".join(lines)


def render_hits_json(hits: list[Hit]) -> str:
    return json.dumps(
        [{"path": h.path, "score": h.score, "snippet": h.snippet} for h in hits],
        indent=2,
    )
