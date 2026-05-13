"""Self-improvement audit: point the agent at the joshv1 repo and ask
it to list concrete things worth changing.

Used by:
- `joshv1 audit` CLI subcommand (one-shot scan, writes a markdown report)
- `/audit` slash command (mid-session scan, streams findings into chat)
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from joshv1.memory import memory_dir


AUDIT_PROMPT = """\
You are auditing the joshv1 codebase for self-improvement. The repo \
contains the same agent that's reading this prompt — be honest, specific, \
and concrete.

Walk the project tree (start with `Read` on README.md and pyproject.toml, \
then list `joshv1/` and explore the modules that look most interesting). \
Spend most of your time on the Python source under `joshv1/`. Skip \
generated files, `__pycache__`, `.venv`, build artifacts.

Produce a markdown report with these sections, in order:

## Summary
2–4 sentences. Overall health, biggest themes.

## High-impact findings
The 3–5 changes you'd ship first. For each:
- **What** (one sentence)
- **Where** (`path/to/file.py:LINE` or a range)
- **Why it matters** (bug, perf, UX, security, maintainability)
- **Suggested fix** (concrete — a code sketch is fine)

## Smaller wins
A bulleted list of 5–15 smaller improvements. Each line: `path/to/file.py:LINE — what to change, in one sentence.`

## Architecture notes
2–4 paragraphs on patterns worth keeping, patterns worth reconsidering, \
or whole-system observations that don't fit a single file.

## Out of scope / deferred
Things you noticed but explicitly chose not to flag (with one-line reason).

Rules:
- No vague advice. Every finding has a file path and a concrete next step.
- Don't list things that are already fine. Lead with the worst stuff.
- Don't propose rewrites of working code. Prefer surgical edits.
- If a finding requires assumptions about runtime behavior, say so.
- It's fine — encouraged, even — to say "nothing major in section X."
"""


def audit_dir(base_memory_dir: str | Path | None = None) -> Path:
    """Where audit reports are saved. Created if missing."""
    d = memory_dir(base_memory_dir) / "audits"
    d.mkdir(parents=True, exist_ok=True)
    return d


def next_report_path(base_memory_dir: str | Path | None = None) -> Path:
    """Fresh timestamped report file under the audits/ directory."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return audit_dir(base_memory_dir) / f"{ts}.md"


def find_hermes_repo() -> Path | None:
    """Walk up from this module to find the joshv1 repo root (.git dir)."""
    here = Path(__file__).resolve()
    for ancestor in [here, *here.parents]:
        if (ancestor / ".git").is_dir():
            return ancestor
    return None


def prompt_for_repo(repo: Path) -> str:
    """The audit prompt with the target repo path interpolated.

    When invoked mid-session via `/audit`, the agent's cwd is the user's
    workspace, not the joshv1 repo. Embedding the absolute path here
    lets the agent use Read/Grep with full paths regardless of cwd.
    """
    return f"{AUDIT_PROMPT}\nThe repo to audit is at: `{repo}`\n"
