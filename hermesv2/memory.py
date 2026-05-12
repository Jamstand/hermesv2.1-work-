"""Persistent agent memory.

Lives at ~/.hermes-memory/ by default. The directory contains plain markdown
files the agent reads at the start of every turn (via the system prompt) and
writes to via the Write tool when it learns something durable about the user.

USER.md is the Honcho-style user model: preferences, recent topics, tone,
ongoing projects. Other .md files in the same dir are loaded too — think
notes/recurring-tasks/jargon — and can be edited by the agent freely.

The contents get appended to the system prompt as a `<memory>...</memory>`
section so the agent always sees them. No round-trips to a separate service.
"""

from __future__ import annotations

from pathlib import Path

DEFAULT_USER_MD = """# About me

(Hermes v2 will update this file as it learns durable facts about you:
preferences, ongoing projects, tone, work context. Feel free to seed it
manually too.)

## Preferences

-

## Ongoing projects

-

## Tone / interaction style

-
"""


def memory_dir(path: str | Path | None = None) -> Path:
    p = Path(path).expanduser() if path else (Path.home() / ".hermes-memory")
    p.mkdir(parents=True, exist_ok=True)
    return p


def ensure_user_md(directory: Path) -> Path:
    user_md = directory / "USER.md"
    if not user_md.exists():
        user_md.write_text(DEFAULT_USER_MD)
    return user_md


def load_memory(directory: Path, max_total_bytes: int = 16_000) -> str:
    """Return a concatenated string of all *.md files in `directory`.

    Truncates each file to keep total size under `max_total_bytes` so a runaway
    memory directory doesn't blow up the system prompt.
    """
    if not directory.is_dir():
        return ""
    chunks: list[str] = []
    used = 0
    for md in sorted(directory.glob("*.md")):
        try:
            text = md.read_text(errors="ignore")
        except OSError:
            continue
        body = text[: max(0, max_total_bytes - used)]
        if not body:
            break
        chunks.append(f"# --- {md.name} ---\n{body}")
        used += len(body)
        if used >= max_total_bytes:
            break
    return "\n\n".join(chunks)


def memory_system_block(directory: Path) -> str:
    """Wrap the loaded memory in a clearly-delimited XML block for the prompt."""
    contents = load_memory(directory)
    if not contents:
        return ""
    return (
        "\n\n<memory>\n"
        "Your persistent notes about this user, refreshed every turn. "
        "When you learn something durable (a preference, a recurring task, a "
        f"naming convention), update the relevant file under {directory} via "
        "the Write tool. Keep entries short, factual, and trim outdated lines.\n\n"
        f"{contents}\n"
        "</memory>"
    )
