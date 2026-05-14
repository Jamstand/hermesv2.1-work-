"""Terminal UI helpers for the `joshv1 chat` REPL.

Renders an ANSI Shadow banner and bordered welcome panel similar to the
upstream NousResearch hermes-agent TUI, then a styled prompt + tool-call
trace + per-turn footer for each interaction.

Only used by `joshv1 chat`. `joshv1 run` stays plain so it pipes cleanly.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import re
from pathlib import Path
from typing import Any

from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text

from joshv1 import __version__, themes
from joshv1.agent import DEFAULT_TOOLS
from joshv1.config import AgentSettings

# Colors are NOT hardcoded here — every style references a semantic
# `josh.*` name (e.g. `josh.title`, `josh.section`) which the
# active Theme (built from joshv1.themes) maps to a hex code. That
# means swapping `/theme` repaints every element without touching this
# file. See joshv1/themes.py for the available palettes.

BANNER = r"""
     ██╗ ██████╗ ███████╗██╗  ██╗   ██╗   ██╗    ██╗
     ██║██╔═══██╗██╔════╝██║  ██║   ██║   ██║   ███║
     ██║██║   ██║███████╗███████║   ██║   ██║   ╚██║
██   ██║██║   ██║╚════██║██╔══██║   ╚██╗ ██╔╝    ██║
╚█████╔╝╚██████╔╝███████║██║  ██║██╗ ╚████╔╝ ██╗ ██║
 ╚════╝  ╚═════╝ ╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝  ╚═╝ ╚═╝
                   W   O   R   K
"""

# Detailed V1 logo for the welcome panel: ANSI-Shadow "V" stacked on "1",
# framed by a winged emblem border with JOSH / WORK / v1 wordmark below.
# About 22 lines tall × 23 cols wide — fits in a side column on most terminals.
LOGO = r"""
       ╔═════════╗
    ╔══╝         ╚══╗
  ╔═╝   ██╗   ██╗   ╚═╗
  ║     ██║   ██║     ║
  ║     ██║   ██║     ║
  ║     ╚██╗ ██╔╝     ║
  ║      ╚████╔╝      ║
  ║       ╚═══╝       ║
  ║        ██╗        ║
  ║       ███║        ║
  ║        ██║        ║
  ║        ██║        ║
  ║      ██████╗      ║
  ╚═╗    ╚══════╝   ╔═╝
    ╚══╗         ╔══╝
       ╚═════════╝
      ━━━━━━━━━━━━━━
       ◈  JOSH   ◈
       ◈   WORK   ◈
             v1
"""

TOOL_GROUPS: dict[str, list[str]] = {
    "files":   ["Read", "Write", "Edit", "NotebookEdit"],
    "search":  ["Grep", "Glob"],
    "shell":   ["Bash", "BashOutput", "KillShell"],
    "web":     ["WebFetch", "WebSearch"],
    "agents":  ["Task", "TodoWrite"],
    "skills":  ["Skill", "SlashCommand", "ExitPlanMode"],
}


def discover_skills(limit: int = 8) -> list[tuple[str, str]]:
    """Scan ~/.claude/skills/ for sub-directories containing SKILL.md.

    Returns a list of (name, description) tuples. Description is pulled from
    the SKILL.md YAML frontmatter when available; otherwise empty.
    """
    skills_dir = Path.home() / ".claude" / "skills"
    if not skills_dir.is_dir():
        return []

    found: list[tuple[str, str]] = []
    for child in sorted(skills_dir.iterdir()):
        if not child.is_dir():
            continue
        skill_md = child / "SKILL.md"
        if not skill_md.is_file():
            continue
        description = ""
        try:
            head = skill_md.read_text(errors="ignore")[:1500]
            # Parse YAML frontmatter description: line
            m = re.search(r"^---\s*\n(.*?)\n---", head, re.DOTALL | re.MULTILINE)
            if m:
                for line in m.group(1).splitlines():
                    if line.lower().startswith("description:"):
                        description = line.split(":", 1)[1].strip().strip('"').strip("'")
                        break
        except OSError:
            pass
        found.append((child.name, description))
        if len(found) >= limit:
            break
    return found


def render_startup(
    console: Console,
    settings: AgentSettings,
    session_name: str | None = None,
) -> None:
    """Print the banner + welcome panel. Called once at the start of `chat`."""
    console.print(Text(BANNER, style="josh.title"), highlight=False)
    tagline = Text()
    tagline.append("    Personal Work Agent  ", style="dim italic")
    tagline.append("·", style="dim")
    tagline.append("  Max-subscription billing  ", style="dim italic")
    tagline.append("·  ", style="dim")
    tagline.append(f"v{__version__}", style="josh.highlight.bold")
    console.print(tagline)
    console.print()
    console.print(_welcome_panel(settings, session_name))
    console.print()


def _welcome_panel(settings: AgentSettings, session_name: str | None) -> Panel:
    # Build the right-side info column.
    info_lines: list[Text] = [Text()]

    # Available Tools section
    info_lines.append(Text("  Available Tools", style="josh.section"))
    for group, tools in TOOL_GROUPS.items():
        line = Text("    ")
        line.append(f"{group:<8}", style="dim")
        line.append(": ", style="dim")
        line.append(", ".join(tools), style="josh.text")
        info_lines.append(line)
    if settings.mcp_servers:
        line = Text("    ")
        line.append(f"{'mcp':<8}", style="dim")
        line.append(": ", style="dim")
        line.append(", ".join(settings.mcp_servers.keys()), style="josh.secondary")
        info_lines.append(line)
    info_lines.append(Text())

    # Available Skills section
    skills = discover_skills(limit=8)
    if skills:
        info_lines.append(Text("  Available Skills", style="josh.section"))
        for name, desc in skills:
            line = Text("    ")
            line.append(f"{name}", style="josh.info")
            if desc:
                short = desc if len(desc) < 60 else desc[:60] + "..."
                line.append(": ", style="dim")
                line.append(short, style="dim")
            info_lines.append(line)
        info_lines.append(Text())

    # Config table
    config_table = [
        ("Model",      settings.model),
        ("Workspace",  settings.workspace_dir),
        ("Effort",     settings.effort),
        ("Thinking",   f"{settings.thinking} · {settings.thinking_display}"),
        ("Permission", settings.permission_mode),
    ]
    for key, val in config_table:
        line = Text("  ")
        line.append(f"{key:<12}", style="dim")
        line.append(str(val), style="josh.info")
        info_lines.append(line)

    # Session line — prominent, formatted like upstream hermes-agent
    session_line = Text("  ")
    session_line.append(f"{'Session':<12}", style="dim")
    session_line.append(session_name or "(ephemeral)", style="josh.highlight.bold")
    info_lines.append(session_line)
    info_lines.append(Text())

    # Counts + help footer
    total_tools = len(DEFAULT_TOOLS) + sum(
        1 for _ in settings.mcp_servers
    )
    counts = Text("  ")
    counts.append(f"{total_tools} tools", style="josh.success")
    counts.append(" · ", style="dim")
    counts.append(f"{len(skills)} skills" if skills else "0 skills", style="josh.success")
    counts.append(" · ", style="dim")
    counts.append("/help for commands · /exit quits", style="dim")
    info_lines.append(counts)
    info_lines.append(Text())

    # Left-side logo column.
    logo = Text(LOGO, style="josh.warm")

    # Compose side-by-side via Columns.
    from rich.columns import Columns
    body = Columns(
        [logo, Group(*info_lines)],
        equal=False,
        expand=False,
        padding=(0, 2),
    )

    active_theme = themes.load_active_palette().name
    title = Text()
    title.append(" Joshv1 ", style="josh.title")
    title.append("(work) ", style="josh.highlight.bold")
    title.append(f"v{__version__}", style="josh.warm")
    title.append(" · ", style="dim")
    title.append(settings.model, style="josh.success")
    title.append(" · ", style="dim")
    title.append("Max subscription", style="josh.chevron")
    title.append("  ·  ", style="dim")
    title.append(active_theme, style="josh.info")
    title.append(" ", style="dim")

    return Panel(
        body,
        title=title,
        title_align="left",
        border_style="josh.border",
        padding=(0, 1),
    )


# Rotating verbs so the spinner doesn't feel stuck on long thinks.
THINKING_VERBS = [
    "thinking",
    "pondering",
    "reasoning",
    "considering",
    "analyzing",
    "deliberating",
    "exploring",
    "reflecting",
    "weighing options",
    "drafting a response",
]


class ThinkingSpinner:
    """A Rich spinner with a rotating verb, for the gap between submit and first reply.

    Start before iterating the agent stream, stop on the first event so the
    assistant label and content can render cleanly.
    """

    def __init__(self, console: Console, rotate_seconds: float = 1.5) -> None:
        self._console = console
        self._rotate_seconds = rotate_seconds
        self._status: Any = None
        self._task: asyncio.Task[None] | None = None
        self._verbs = itertools.cycle(THINKING_VERBS)
        self._stopped = False

    def _label(self, verb: str) -> str:
        return f"[josh.chevron]{verb}...[/]"

    def start(self) -> None:
        if self._status is not None:
            return
        verb = next(self._verbs)
        self._status = self._console.status(self._label(verb), spinner="dots", spinner_style="josh.border")
        self._status.__enter__()
        self._task = asyncio.create_task(self._rotate())

    async def _rotate(self) -> None:
        try:
            while not self._stopped:
                await asyncio.sleep(self._rotate_seconds)
                if self._stopped or self._status is None:
                    return
                self._status.update(self._label(next(self._verbs)))
        except asyncio.CancelledError:
            pass

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        if self._task is not None:
            self._task.cancel()
            self._task = None
        if self._status is not None:
            with contextlib.suppress(Exception):
                self._status.__exit__(None, None, None)
            self._status = None


def prompt_label() -> str:
    # Used by the non-prompt-toolkit path (legacy). New chat uses HTML prompt.
    return "\n[bold cyan]▎[/] [bold cyan]you[/] [josh.chevron]❱[/] "


def render_text_delta(console: Console, text: str) -> None:
    console.print(text, end="", soft_wrap=True, highlight=False)


# Known Claude models for the /model picker. Tuples of (model_id, label).
KNOWN_MODELS: list[tuple[str, str]] = [
    ("claude-opus-4-7",   "Opus 4.7  ·  most capable (default)"),
    ("claude-opus-4-6",   "Opus 4.6  ·  previous Opus generation"),
    ("claude-sonnet-4-6", "Sonnet 4.6  ·  faster, cheaper on Max quota"),
    ("claude-haiku-4-5",  "Haiku 4.5  ·  fastest, simplest tasks"),
]


# Effort levels for the /effort picker. Tuples of (effort_id, label).
KNOWN_EFFORTS: list[tuple[str, str]] = [
    ("low",    "minimal thinking · cheap, snappy replies"),
    ("medium", "moderate thinking · balanced"),
    ("high",   "deep thinking · default"),
    ("xhigh",  "extra-deep thinking · hard problems"),
    ("max",    "maximum thinking budget · only when you really need it"),
]


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}


def find_recent_screenshot(extra_dirs: list[str] | None = None) -> Path | None:
    """Find the most-recently-modified image in well-known screenshot folders.

    Looks under WSL's view of every Windows user's Pictures/Screenshots dir,
    plus ~/Pictures/Screenshots and ~/Pictures, plus any extra dirs caller
    supplies. Returns None if nothing matches.
    """
    candidates: list[Path] = []
    win_users = Path("/mnt/c/Users")
    if win_users.is_dir():
        for user_dir in win_users.iterdir():
            if not user_dir.is_dir():
                continue
            candidates.append(user_dir / "Pictures" / "Screenshots")
            candidates.append(user_dir / "OneDrive" / "Pictures" / "Screenshots")
    candidates.append(Path.home() / "Pictures" / "Screenshots")
    candidates.append(Path.home() / "Pictures")
    if extra_dirs:
        candidates.extend(Path(p).expanduser() for p in extra_dirs)

    best: tuple[float, Path] | None = None
    for directory in candidates:
        if not directory.is_dir():
            continue
        for path in directory.iterdir():
            if path.suffix.lower() not in IMAGE_EXTS or not path.is_file():
                continue
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            if best is None or mtime > best[0]:
                best = (mtime, path)
    return best[1] if best else None


async def pick_model_dialog(console: Console, current_model: str) -> str | None:
    """Numbered-list model picker. Returns chosen model_id, or None if cancelled.

    Was originally a full radiolist_dialog modal, but launching another
    prompt-toolkit Application from inside the running chat session caused
    key events not to register (Enter/Esc did nothing). Numbered input via
    a fresh PromptSession is bulletproof and works in every terminal.
    """
    from prompt_toolkit import PromptSession
    from prompt_toolkit.formatted_text import HTML

    console.print()
    console.print(
        f"  [josh.chevron]▎[/] [bold cyan]Model Picker[/] "
        f"[dim]· currently on[/] [josh.secondary]{current_model}[/]"
    )
    console.print()
    width = max(len(mid) for mid, _ in KNOWN_MODELS) + 1
    for i, (model_id, label) in enumerate(KNOWN_MODELS, 1):
        marker = " [josh.highlight.bold]← current[/]" if model_id == current_model else ""
        console.print(
            f"    [josh.highlight.bold]{i}.[/] "
            f"[josh.info]{model_id:<{width}}[/] "
            f"[dim]· {label}[/]{marker}"
        )
    console.print()

    ps: PromptSession[str] = PromptSession()
    n_models = len(KNOWN_MODELS)
    try:
        raw = await ps.prompt_async(
            HTML(themes.picker_prompt_html(
                themes.load_active_palette(),
                f"pick a number (1-{n_models}), or Enter alone to cancel:",
            ))
        )
    except (EOFError, KeyboardInterrupt):
        console.print()
        return None

    raw = raw.strip()
    if not raw:
        return None
    try:
        idx = int(raw) - 1
        if 0 <= idx < len(KNOWN_MODELS):
            return KNOWN_MODELS[idx][0]
    except ValueError:
        pass
    console.print("  [josh.error]invalid choice[/]")
    return None


async def pick_effort_dialog(console: Console, current_effort: str) -> str | None:
    """Numbered-list effort picker. Returns chosen effort, or None if cancelled."""
    from prompt_toolkit import PromptSession
    from prompt_toolkit.formatted_text import HTML

    console.print()
    console.print(
        f"  [josh.chevron]▎[/] [bold cyan]Effort Picker[/] "
        f"[dim]· currently on[/] [josh.secondary]{current_effort}[/]"
    )
    console.print()
    width = max(len(eid) for eid, _ in KNOWN_EFFORTS) + 1
    for i, (effort_id, label) in enumerate(KNOWN_EFFORTS, 1):
        marker = " [josh.highlight.bold]← current[/]" if effort_id == current_effort else ""
        console.print(
            f"    [josh.highlight.bold]{i}.[/] "
            f"[josh.info]{effort_id:<{width}}[/] "
            f"[dim]· {label}[/]{marker}"
        )
    console.print()

    ps: PromptSession[str] = PromptSession()
    n_efforts = len(KNOWN_EFFORTS)
    try:
        raw = await ps.prompt_async(
            HTML(themes.picker_prompt_html(
                themes.load_active_palette(),
                f"pick a number (1-{n_efforts}), or Enter alone to cancel:",
            ))
        )
    except (EOFError, KeyboardInterrupt):
        console.print()
        return None

    raw = raw.strip()
    if not raw:
        return None
    try:
        idx = int(raw) - 1
        if 0 <= idx < len(KNOWN_EFFORTS):
            return KNOWN_EFFORTS[idx][0]
    except ValueError:
        pass
    console.print("  [josh.error]invalid choice[/]")
    return None


def render_assistant_markdown(console: Console, text: str) -> None:
    """Render a buffered assistant text segment as Markdown.

    Used instead of streaming raw text so things like **bold**, bullet lists,
    `inline code`, and code fences render with proper styling.

    Falls back to plain text if Rich's markdown renderer (or its pygments
    dependency for code fences) raises — a broken pygments install
    shouldn't crash the whole agent mid-reply.
    """
    if not text.strip():
        return
    from rich.markdown import Markdown
    try:
        md = Markdown(text, code_theme="monokai")
        console.print(md)
    except Exception:
        console.print(text, highlight=False)


def assistant_label(console: Console) -> None:
    # Full line, so the following markdown block renders cleanly below.
    console.print("[josh.chevron]▎ josh ❰[/]")


def render_thinking_delta(console: Console, text: str) -> None:
    console.print(f"[dim italic]{text}[/]", end="", soft_wrap=True, highlight=False)


async def pick_answer_for_question(
    console: Console,
    question: str,
    options: list[dict],
    header: str = "",
    multi_select: bool = False,
) -> str | list[str] | None:
    """Interactive picker for AskUserQuestion. Arrow keys + Enter to select, Esc to cancel.

    Returns the chosen option's label (str), a list of labels for multi-select,
    or None if the user cancelled. Always offers an extra "Other (type your own)"
    row at the bottom — Enter on that opens an inline text prompt.
    """
    from prompt_toolkit.application import Application
    from prompt_toolkit.formatted_text import HTML
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import Layout, Window
    from prompt_toolkit.layout.containers import HSplit
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.styles import Style

    if not options:
        return None

    rows = list(options) + [{"label": "Other (type your own)", "description": "", "_other": True}]
    n = len(rows)
    state: dict = {"selected": 0, "marked": set(), "result": "__pending__"}

    palette = themes.load_active_palette()

    def render_lines():
        out: list[tuple[str, str]] = []
        if header:
            out.append(("class:hdr", f"  [{header}]\n"))
        out.append(("class:q", f"  {question}\n\n"))
        for i, opt in enumerate(rows):
            is_sel = i == state["selected"]
            cursor = "▸ " if is_sel else "  "
            mark = ""
            if multi_select and not opt.get("_other"):
                mark = "[✓] " if i in state["marked"] else "[ ] "
            label = opt.get("label", "")
            desc = opt.get("description", "")
            row_style = "class:focused" if is_sel else "class:row"
            out.append((row_style, f"  {cursor}{mark}{label}\n"))
            if desc and is_sel:
                out.append(("class:desc", f"      {desc}\n"))
        hint = "\n  ↑/↓ navigate · Enter select · Esc cancel"
        if multi_select:
            hint += " · Space toggle"
        out.append(("class:hint", hint))
        return out

    kb = KeyBindings()

    @kb.add("up")
    @kb.add("k")
    def _(event):
        state["selected"] = (state["selected"] - 1) % n

    @kb.add("down")
    @kb.add("j")
    def _(event):
        state["selected"] = (state["selected"] + 1) % n

    @kb.add("enter")
    def _(event):
        opt = rows[state["selected"]]
        if opt.get("_other"):
            state["result"] = "__other__"
        elif multi_select:
            state["result"] = [rows[i].get("label", "") for i in sorted(state["marked"])] or [opt.get("label", "")]
        else:
            state["result"] = opt.get("label", "")
        event.app.exit()

    @kb.add("escape", eager=True)
    @kb.add("c-c")
    def _(event):
        state["result"] = None
        event.app.exit()

    if multi_select:
        @kb.add(" ")
        def _(event):
            if rows[state["selected"]].get("_other"):
                return
            if state["selected"] in state["marked"]:
                state["marked"].remove(state["selected"])
            else:
                state["marked"].add(state["selected"])

    control = FormattedTextControl(render_lines, focusable=True, key_bindings=kb)
    window = Window(content=control, always_hide_cursor=True)
    layout = Layout(HSplit([window]))
    style = Style.from_dict({
        "hdr":     f"{palette.secondary} bold",
        "q":       f"{palette.primary} bold",
        "focused": f"reverse {palette.primary}",
        "row":     palette.text,
        "desc":    f"italic {palette.dim}",
        "hint":    palette.dim,
    })

    console.print()
    app: Application = Application(
        layout=layout, key_bindings=kb, style=style,
        full_screen=False, mouse_support=False,
    )
    await app.run_async()

    if state["result"] == "__other__":
        # Inline freeform answer for the "Other" row
        from prompt_toolkit import PromptSession
        ps: PromptSession[str] = PromptSession()
        try:
            text = await ps.prompt_async(
                HTML(themes.picker_prompt_html(palette, "type your answer (Enter to submit, Esc to cancel):"))
            )
        except (EOFError, KeyboardInterrupt):
            return None
        text = text.strip()
        return text if text else None

    return state["result"] if state["result"] != "__pending__" else None


def format_picker_answers(questions: list[dict], answers: dict[str, str | list[str]]) -> str:
    """Turn picker answers into a user message Claude can act on.

    Format is one line per question: "<header or short question>: <answer>".
    Multi-select answers are comma-joined.
    """
    parts: list[str] = []
    for q in questions:
        key = q.get("header") or (q.get("question") or "answer")
        ans = answers.get(key)
        if ans is None:
            continue
        if isinstance(ans, list):
            parts.append(f"{key}: {', '.join(ans)}")
        else:
            parts.append(f"{key}: {ans}")
    return "\n".join(parts)


def _format_tool_input(name: str, tool_input: dict) -> str:
    """Compact, readable preview of a tool's input arguments. Falls back to
    a truncated dict dump for tool names we don't know."""

    def _line(s: str, max_len: int = 100) -> str:
        s = s.replace("\n", " ⏎ ")
        return s if len(s) <= max_len else s[:max_len] + "…"

    if name == "Bash":
        cmd = tool_input.get("command", "")
        desc = tool_input.get("description", "")
        first = _line(cmd.split("\n", 1)[0], 120)
        out = f"[josh.text]{first}[/]"
        if "\n" in cmd:
            extra_lines = cmd.count("\n")
            out += f" [dim](+{extra_lines} more line{'s' if extra_lines > 1 else ''})[/]"
        if desc:
            out += f"\n      [dim]↳ {desc}[/]"
        return out

    if name == "Read":
        path = tool_input.get("file_path", "")
        offset = tool_input.get("offset")
        limit = tool_input.get("limit")
        out = f"[josh.info]{path}[/]"
        if offset or limit:
            out += f" [dim](lines {offset or 1}{f'..+{limit}' if limit else ''})[/]"
        return out

    if name == "Write":
        path = tool_input.get("file_path", "")
        content = tool_input.get("content", "")
        n_lines = content.count("\n") + (1 if content else 0)
        return f"[josh.info]{path}[/] [dim]({len(content)} chars, {n_lines} line{'s' if n_lines != 1 else ''})[/]"

    if name == "Edit":
        path = tool_input.get("file_path", "")
        old = _line(tool_input.get("old_string", ""), 70)
        new = _line(tool_input.get("new_string", ""), 70)
        replace_all = tool_input.get("replace_all", False)
        all_marker = " [dim](all)[/]" if replace_all else ""
        return (f"[josh.info]{path}[/]{all_marker}\n"
                f"      [josh.error.bold]- [/][dim]{old}[/]\n"
                f"      [josh.success]+ [/][dim]{new}[/]")

    if name == "NotebookEdit":
        path = tool_input.get("notebook_path", "")
        cell = tool_input.get("cell_id") or tool_input.get("cell_number", "")
        return f"[josh.info]{path}[/] [dim]cell {cell}[/]"

    if name == "Grep":
        pattern = tool_input.get("pattern", "")
        path = tool_input.get("path", "")
        mode = tool_input.get("output_mode", "")
        out = f"[josh.text]{_line(pattern, 80)}[/]"
        if path:
            out += f" [dim]in {path}[/]"
        if mode and mode != "files_with_matches":
            out += f" [dim]({mode})[/]"
        return out

    if name == "Glob":
        pattern = tool_input.get("pattern", "")
        path = tool_input.get("path", "")
        return f"[josh.text]{pattern}[/]" + (f" [dim]in {path}[/]" if path else "")

    if name == "WebFetch":
        url = tool_input.get("url", "")
        prompt = tool_input.get("prompt", "")
        out = f"[josh.info]{url}[/]"
        if prompt:
            out += f"\n      [dim]↳ {_line(prompt, 90)}[/]"
        return out

    if name == "WebSearch":
        query = tool_input.get("query", "")
        return f"[josh.text]{_line(query, 100)}[/]"

    if name == "TodoWrite":
        todos = tool_input.get("todos", [])
        if not todos:
            return "[dim](empty)[/]"
        active = next((t for t in todos if t.get("status") == "in_progress"), None)
        lines = [f"[dim]{len(todos)} todo{'s' if len(todos) != 1 else ''}[/]"]
        if active:
            lines.append(f"      [josh.warm]▶[/] {_line(active.get('content', ''), 80)}")
        else:
            first = todos[0]
            lines.append(f"      [dim]·[/] {_line(first.get('content', ''), 80)}")
        return "\n".join(lines)

    if name == "Task":
        agent_type = tool_input.get("subagent_type", "agent")
        desc = tool_input.get("description", "")
        return f"[josh.secondary]{agent_type}[/] [dim]— {_line(desc, 100)}[/]"

    if name == "Skill":
        skill = tool_input.get("skill", "")
        args = tool_input.get("args", "")
        out = f"[josh.secondary]{skill}[/]"
        if args:
            out += f" [dim]{_line(str(args), 80)}[/]"
        return out

    if name == "SlashCommand":
        return f"[josh.highlight]{_line(tool_input.get('command', ''), 120)}[/]"

    # Unknown tool — fall back to truncated repr
    preview = str(tool_input)
    if len(preview) > 200:
        preview = preview[:200] + "…"
    return f"[dim]{preview}[/]"


def render_tool_call(console: Console, name: str, tool_input: dict) -> None:
    # Special-case AskUserQuestion so the question + options render as a
    # readable prompt instead of a raw dict dump.
    if name == "AskUserQuestion" and isinstance(tool_input, dict):
        questions = tool_input.get("questions") or []
        if questions:
            console.print(f"\n  [josh.info]⚙ {name}[/]")
            for q in questions:
                qtext = q.get("question", "")
                header = q.get("header", "")
                multi = q.get("multiSelect", False)
                hint = " [dim](multi-select)[/]" if multi else ""
                tag = f" [josh.secondary][{header}][/]" if header else ""
                console.print(f"  [bold]{qtext}[/]{tag}{hint}")
                for opt in q.get("options", []):
                    label = opt.get("label", "")
                    desc = opt.get("description", "")
                    console.print(f"    • [bold]{label}[/] [dim]— {desc}[/]")
                console.print()
            return

    if isinstance(tool_input, dict):
        formatted = _format_tool_input(name, tool_input)
        console.print(f"\n  [josh.info]⚙ {name}[/]  {formatted}")
        return

    preview = str(tool_input)
    if len(preview) > 200:
        preview = preview[:200] + "..."
    console.print(f"\n  [josh.info]⚙ {name}[/] [dim]{preview}[/]")


def render_tool_result(console: Console, name: str, output: str, is_error: bool) -> None:
    color = "josh.error" if is_error else "josh.success"
    preview = output if len(output) < 400 else output[:400] + "..."
    console.print(f"  [{color}]↳ {name}[/] [dim]{preview}[/]")


def render_turn_footer(
    console: Console,
    stop_reason: str,
    cost_usd: float | None,
    usage: dict,
) -> None:
    line = Text()
    line.append("  ⚕ ", style="josh.chevron")
    line.append(f"stop={stop_reason}", style="dim")
    if usage:
        input_tokens = usage.get("input_tokens") or 0
        output_tokens = usage.get("output_tokens") or 0
        if input_tokens or output_tokens:
            line.append(" · ", style="dim")
            line.append(f"in {input_tokens} / out {output_tokens}", style="dim")
    line.append(" · ", style="dim")
    if cost_usd:
        line.append(f"~${cost_usd:.4f} equiv", style="dim")
        line.append(" · ", style="dim")
    line.append("Max subscription", style="josh.secondary")
    console.print()
    console.print(line)


def render_status_bar(
    console: Console,
    model: str,
    ctx_pct: float | None,
    turn_seconds: float,
    session_seconds: float,
    session_id: str = "default",
) -> None:
    """Status line printed between turns, mimicking the upstream bottom bar."""
    if ctx_pct is None:
        ctx_str = "ctx --"
        bar = "[░░░░░░░░░░] --"
    else:
        filled = max(0, min(10, int(round(ctx_pct / 10))))
        bar_chars = "█" * filled + "░" * (10 - filled)
        ctx_str = f"ctx {ctx_pct:.0f}%"
        bar = f"[{bar_chars}] {ctx_pct:.0f}%"

    parts = Text()
    parts.append(" ⚕ ", style="josh.chevron")
    parts.append(model, style="josh.info")
    parts.append(" │ ", style="dim")
    parts.append(f"⌖ {session_id}", style="josh.session.marker")
    parts.append(" │ ", style="dim")
    parts.append(ctx_str, style="josh.success" if ctx_pct and ctx_pct < 70 else "yellow")
    parts.append(" │ ", style="dim")
    parts.append(bar, style="dim")
    parts.append(" │ ", style="dim")
    parts.append(f"{_fmt_seconds(session_seconds)}", style="dim")
    parts.append(" │ 🕐 ", style="dim")
    parts.append(f"{_fmt_seconds(turn_seconds)}", style="dim")
    console.print(parts)


def render_sessions(console: Console, sessions: list) -> None:
    """Print a table of saved Claude Code sessions."""
    from datetime import datetime

    from rich.table import Table

    table = Table(border_style="dim", header_style="josh.title")
    table.add_column("Session ID", style="josh.info", no_wrap=True)
    table.add_column("Modified", style="dim")
    table.add_column("Branch / cwd", style="dim")
    table.add_column("Summary", style="josh.text")

    for s in sessions:
        ts = getattr(s, "last_modified", 0) or 0
        when = datetime.fromtimestamp(ts / 1000).strftime("%Y-%m-%d %H:%M") if ts else "?"
        loc = getattr(s, "git_branch", None) or getattr(s, "cwd", "") or ""
        summary = (
            getattr(s, "custom_title", None)
            or getattr(s, "summary", None)
            or getattr(s, "first_prompt", None)
            or "(empty)"
        )
        if len(summary) > 60:
            summary = summary[:60] + "..."
        table.add_row(s.session_id[:16] + "...", when, loc[:30], summary)
    console.print(table)


def _fmt_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m"


# Slash commands as structured (command, description) pairs so they can be
# rendered both as text help AND as completer suggestions.
SLASH_COMMANDS: list[tuple[str, str]] = [
    # General
    ("/help",       "Show available slash commands"),
    ("/exit",       "Quit joshv1"),
    ("/quit",       "Quit joshv1 (alias for /exit)"),
    # Session control
    ("/new",        "Start a new session (fresh history)"),
    ("/reset",      "Start a new session (alias for /new)"),
    ("/clear",      "Clear the screen"),
    ("/redraw",     "Re-render banner + welcome panel"),
    ("/title",      "Set a title for the current session (usage: /title <name>)"),
    ("/history",    "Show recent user prompts in this session"),
    ("/retry",      "Re-send the last user prompt to the agent"),
    ("/img",        "Send an image to the agent (usage: /img <path> [question])"),
    ("/screenshot", "Send your most recent screenshot to the agent (optional question)"),
    ("/maps",       "Open Google Maps + braille-map for a place (usage: /maps <query> [zN])"),
    ("/zoomin",     "Re-render the last /maps result one zoom level closer"),
    ("/zoomout",    "Re-render the last /maps result one zoom level wider"),
    ("/theme",      "Switch color theme (usage: /theme <name>, or /theme to list)"),
    ("/plugin",     "Manage Claude Code plugins (usage: /plugin install <name>@<marketplace>)"),
    ("/branch",     "Fork the current session under a new name (usage: /branch <name>)"),
    ("/fork",       "Fork the current session (alias for /branch)"),
    ("/compress",   "Manually compact the conversation summary"),
    # Permission modes
    ("/plan",       "Switch to plan mode (Claude proposes a plan before executing)"),
    ("/safe",       "Switch to default permission mode (prompts before destructive)"),
    ("/auto",       "Switch to acceptEdits mode (auto-approve file edits)"),
    ("/yolo",       "Switch to bypassPermissions mode — full trust, no prompts"),
    ("/permission", "Set permission mode explicitly (usage: /permission <mode>)"),
    # Workspace
    ("/cwd",        "Show the current workspace + any extra mounted dirs"),
    ("/cd",         "Add an extra directory the agent can touch (usage: /cd <path>)"),
    ("/sysprompt",  "Print the current system prompt"),
    # Status / info
    ("/context",    "Show current context-window usage"),
    ("/stats",      "Show this session's totals (turns, tokens, time)"),
    ("/sessions",   "List saved Claude Code sessions"),
    ("/tools",      "List built-in tools (Read, Write, Bash, etc.)"),
    ("/model",      "Switch model mid-session (usage: /model <name>)"),
    ("/effort",     "Change agent effort (interactive picker; persistent)"),
    ("/update",     "git pull the latest joshv1 from origin"),
    ("/audit",      "Have joshv1 scan its own repo and report improvements"),
    ("/ensemble",   "Multi-model ensemble: /ensemble (status) | on|off|toggle | add|remove|reset|retune | run <prompt> | <prompt>"),
    # New: full Hermes-style command surface. Aliases share their target's
    # description prefix with `alias for /<target>`. See SLASH_GROUPS below for
    # the grouped layout rendered by /help. The flat list is kept so the
    # autocomplete completer in cli.py still works without changes.
    ("/save",       "Save the current conversation to a markdown file"),
    ("/undo",       "Remove the last user/assistant exchange from history"),
    ("/branch",     "Branch the current session (explore a different path) (usage: /branch [name])"),
    ("/fork",       "alias for /branch"),
    ("/compress",   "Manually compress conversation context (usage: /compress [focus topic])"),
    ("/rollback",   "List or restore filesystem checkpoints (usage: /rollback [number])"),
    ("/snapshot",   "Create or restore state snapshots (usage: /snapshot [create|restore <id>|list|prune])"),
    ("/snap",       "alias for /snapshot"),
    ("/stop",       "Kill all running background tasks"),
    ("/background", "Run a prompt in the background (usage: /background <prompt>)"),
    ("/bg",         "alias for /background"),
    ("/btw",        "alias for /background"),
    ("/agents",     "Show active background tasks"),
    ("/tasks",      "alias for /agents"),
    ("/queue",      "Queue a prompt to run after the current turn (usage: /queue <prompt>)"),
    ("/q",          "alias for /queue"),
    ("/steer",      "Inject a message after the next tool call (usage: /steer <prompt>)"),
    ("/status",     "Show session info (id, turns, ctx %, time)"),
    ("/resume",     "Resume a previously-named session (usage: /resume [name])"),
    ("/profile",    "Show active profile name and home directory"),
    ("/gquota",     "Show Google Gemini quota usage"),
    ("/usage",      "Show token usage and cost totals for the session"),
    ("/insights",   "Show usage insights and analytics (usage: /insights [days])"),
    ("/platforms",  "Show gateway/messaging platform status (Slack/Discord)"),
    ("/gateway",    "alias for /platforms"),
    ("/copy",       "Copy the last assistant response to clipboard (usage: /copy [n])"),
    ("/paste",      "Attach clipboard image to your next prompt"),
    ("/debug",      "Generate a debug report (system info + recent logs)"),
    ("/config",     "Show current configuration"),
    ("/provider",   "alias for /model"),
    ("/personality","Set a predefined personality (usage: /personality [name])"),
    ("/statusbar",  "Toggle the context/model status bar"),
    ("/sb",         "alias for /statusbar"),
    ("/verbose",    "Cycle tool progress display: off → new → all → verbose"),
    ("/reasoning",  "Manage reasoning effort and display (usage: /reasoning [level|show|hide])"),
    ("/skin",       "Show or change the display skin/theme (usage: /skin [name])"),
    ("/voice",      "Transcribe an audio file as the next prompt (usage: /voice <path>)"),
    ("/busy",       "Control what Enter does while busy (usage: /busy [queue|steer|interrupt|status])"),
    ("/toolsets",   "List available toolsets"),
    ("/skills",     "Search, install, inspect, or manage skills"),
    ("/cron",       "Manage scheduled tasks (usage: /cron [list|add|remove])"),
    ("/reload",     "Reload .env variables into the running session"),
    ("/reload-mcp", "Reload MCP servers from config"),
    ("/reload_mcp", "alias for /reload-mcp"),
    ("/browser",    "Connect browser tools to your live Chrome via CDP (usage: /browser [connect|disconnect|status])"),
    ("/plugins",    "List installed plugins and their status"),
]


HIGH_VALUE_COMMANDS: set[str] = {
    "/queue", "/steer", "/background", "/snapshot", "/undo",
    "/save", "/copy", "/paste", "/usage", "/config", "/ensemble",
}


# Grouped layout for the new /help. Order intentionally mirrors the Hermes TUI
# reference ([Session, Info, Configuration, Tools & Skills, Exit]) so users
# coming from there find what they expect.
SLASH_GROUPS: list[tuple[str, list[tuple[str, str]]]] = [
    ("Session", [
        ("/new",        "Start a new session (fresh session ID + history)"),
        ("/reset",      "alias for /new"),
        ("/clear",      "Clear screen and start a new session"),
        ("/redraw",     "Force a full UI repaint (recovers from terminal drift)"),
        ("/history",    "Show conversation history"),
        ("/save",       "Save the current conversation to a markdown file"),
        ("/retry",      "Retry the last message (resend to agent)"),
        ("/undo",       "Remove the last user/assistant exchange"),
        ("/title",      "Set a title for the current session (usage: /title [name])"),
        ("/branch",     "Branch the current session (usage: /branch [name])"),
        ("/fork",       "alias for /branch"),
        ("/compress",   "Manually compress conversation context (usage: /compress [focus])"),
        ("/rollback",   "List or restore filesystem checkpoints"),
        ("/snapshot",   "Create or restore state snapshots (usage: /snapshot [create|restore <id>|list|prune])"),
        ("/snap",       "alias for /snapshot"),
        ("/stop",       "Kill all running background tasks"),
        ("/background", "Run a prompt in the background (usage: /background <prompt>)"),
        ("/bg",         "alias for /background"),
        ("/btw",        "alias for /background"),
        ("/agents",     "Show active background tasks"),
        ("/tasks",      "alias for /agents"),
        ("/queue",      "Queue a prompt for the next turn (usage: /queue <prompt>)"),
        ("/q",          "alias for /queue"),
        ("/steer",      "Inject a message after the next tool call (usage: /steer <prompt>)"),
        ("/status",     "Show session info (id, turns, ctx %, time)"),
        ("/resume",     "Resume a previously-named session (usage: /resume [name])"),
    ]),
    ("Info", [
        ("/profile",    "Show active profile name and home directory"),
        ("/gquota",     "Show Google Gemini quota usage"),
        ("/help",       "Show this command list"),
        ("/usage",      "Show token usage and cost totals for the session"),
        ("/insights",   "Show usage insights and analytics (usage: /insights [days])"),
        ("/platforms",  "Show Slack/Discord gateway status"),
        ("/gateway",    "alias for /platforms"),
        ("/copy",       "Copy the last assistant response to clipboard (usage: /copy [n])"),
        ("/paste",      "Attach a clipboard image to your next prompt"),
        ("/image",      "Attach a local image file (usage: /image <path>)"),
        ("/img",        "alias for /image"),
        ("/screenshot", "Attach the most recent screenshot from your Pictures dir"),
        ("/debug",      "Generate a debug report (system info + recent logs)"),
    ]),
    ("Configuration", [
        ("/config",     "Show current configuration"),
        ("/model",      "Switch model for this session (usage: /model [model])"),
        ("/provider",   "alias for /model"),
        ("/personality","Set a predefined personality (usage: /personality [name])"),
        ("/statusbar",  "Toggle the context/model status bar"),
        ("/sb",         "alias for /statusbar"),
        ("/verbose",    "Cycle tool progress display: off → new → all → verbose"),
        ("/yolo",       "Set permission mode to bypassPermissions (full trust)"),
        ("/auto",       "Set permission mode to acceptEdits"),
        ("/safe",       "Set permission mode to default (prompt before destructive)"),
        ("/plan",       "Set permission mode to plan (Claude proposes before executing)"),
        ("/permission", "Set permission mode explicitly (usage: /permission <mode>)"),
        ("/effort",     "Change agent effort level (interactive picker)"),
        ("/reasoning",  "Manage reasoning effort and display (usage: /reasoning [level|show|hide])"),
        ("/skin",       "Show or change the display skin/theme (usage: /skin [name])"),
        ("/voice",      "Transcribe an audio file as the next prompt (usage: /voice <path>)"),
        ("/busy",       "Control what Enter does while busy (usage: /busy [queue|steer|interrupt|status])"),
        ("/cwd",        "Show the current workspace + any extra mounted dirs"),
        ("/cd",         "Add an extra directory the agent can touch (usage: /cd <path>)"),
        ("/sysprompt",  "Print the current system prompt"),
    ]),
    ("Tools & Skills", [
        ("/tools",      "List built-in tools (Read, Write, Bash, etc.)"),
        ("/toolsets",   "List available toolsets"),
        ("/skills",     "Search, install, inspect, or manage skills"),
        ("/cron",       "Manage scheduled tasks (usage: /cron [list|add|remove])"),
        ("/reload",     "Reload .env variables into the running session"),
        ("/reload-mcp", "Reload MCP servers from config"),
        ("/reload_mcp", "alias for /reload-mcp"),
        ("/browser",    "Connect browser tools via CDP (usage: /browser [connect|disconnect|status])"),
        ("/plugins",    "List installed plugins and their status"),
        ("/ensemble",   "Multi-model ensemble (see: /ensemble for subcommands)"),
        ("/maps",       "Geocode + braille map (usage: /maps <place> [z<N>])"),
        ("/audit",      "Have joshv1 scan its own repo and report improvements"),
        ("/update",     "git pull the latest joshv1 from origin"),
        ("/sessions",   "List saved Claude Code sessions"),
    ]),
    ("Exit", [
        ("/quit",       "Exit the CLI"),
        ("/exit",       "alias for /quit"),
    ]),
]


def render_help(console: Console) -> None:
    """Grouped help layout. High-value commands marked with ★."""
    console.print()
    console.print("  ┌" + "─" * 53 + "┐")
    console.print("  │             [bold cyan](^_^)? Available Commands[/]              │")
    console.print("  └" + "─" * 53 + "┘")
    for group_name, cmds in SLASH_GROUPS:
        console.print(f"\n  [josh.section]── {group_name} ──[/]")
        for cmd, desc in cmds:
            marker = "[josh.success]★[/]" if cmd in HIGH_VALUE_COMMANDS else " "
            console.print(
                f"   {marker} [josh.highlight]{cmd:<14}[/] "
                f"[dim]- {desc}[/]"
            )
    console.print(f"\n  [dim]★ = high-value command. Type any /command to use it.[/]")
    console.print("  [dim]Tip: tab-complete works on all commands.[/]\n")


def render_help_flat(console: Console) -> None:
    """Original flat help (kept for callers that want the simpler listing)."""
    console.print("\n[bold cyan]Slash commands[/]")
    for cmd, desc in SLASH_COMMANDS:
        console.print(f"  [josh.highlight]{cmd:<12}[/]  [dim]{desc}[/]")
    console.print()


def render_tools(console: Console) -> None:
    console.print("\n[josh.section]Built-in tools[/]")
    for group, tools in TOOL_GROUPS.items():
        line = Text("  ")
        line.append(f"{group:<8}", style="dim")
        line.append(", ".join(tools), style="josh.text")
        console.print(line)
    console.print()


def render_stats(
    console: Console, turns: int, total_input: int, total_output: int,
    total_cost: float, session_seconds: float,
) -> None:
    line = Text()
    line.append("\n  session  ", style="josh.title")
    line.append(f"{turns} turn{'s' if turns != 1 else ''}", style="josh.success")
    line.append(" · ", style="dim")
    line.append(f"in {total_input} / out {total_output} tokens", style="dim")
    line.append(" · ", style="dim")
    if total_cost > 0:
        line.append(f"~${total_cost:.4f} equiv", style="dim")
        line.append(" · ", style="dim")
    line.append(f"{_fmt_seconds(session_seconds)} elapsed", style="dim")
    line.append(" · ", style="dim")
    line.append("billed to Max subscription", style="josh.secondary")
    console.print(line)
    console.print()


def render_kv_block(
    console: Console, title: str, items: list[tuple[str, str]],
    key_width: int | None = None,
) -> None:
    """Two-column key/value block. Used by /config, /status, /profile, /usage."""
    console.print(f"\n[josh.title]{title}[/]")
    width = key_width or max((len(k) for k, _ in items), default=0)
    for k, v in items:
        console.print(f"  [dim]{k:<{width}}[/]  [josh.text]{v}[/]")
    console.print()
