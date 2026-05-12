"""Terminal UI helpers for the `hermesv2 chat` REPL.

Renders an ANSI Shadow banner and bordered welcome panel similar to the
upstream NousResearch hermes-agent TUI, then a styled prompt + tool-call
trace + per-turn footer for each interaction.

Only used by `hermesv2 chat`. `hermesv2 run` stays plain so it pipes cleanly.
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

from hermesv2 import __version__
from hermesv2.agent import DEFAULT_TOOLS
from hermesv2.config import AgentSettings

BANNER = r"""
██╗  ██╗███████╗██████╗ ███╗   ███╗███████╗███████╗   ██╗   ██╗██████╗     ██╗
██║  ██║██╔════╝██╔══██╗████╗ ████║██╔════╝██╔════╝   ██║   ██║╚════██╗   ███║
███████║█████╗  ██████╔╝██╔████╔██║█████╗  ███████╗   ██║   ██║ █████╔╝   ╚██║
██╔══██║██╔══╝  ██╔══██╗██║╚██╔╝██║██╔══╝  ╚════██║   ╚██╗ ██╔╝██╔═══╝     ██║
██║  ██║███████╗██║  ██║██║ ╚═╝ ██║███████╗███████║██╗ ╚████╔╝ ███████╗██╗ ██║
╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝╚══════╝╚═╝  ╚═══╝  ╚══════╝╚═╝ ╚═╝
                                                       W   O   R   K
"""

# Detailed V2 logo for the welcome panel: ANSI-Shadow "V" stacked on "2",
# framed by a winged emblem border with HERMES / WORK / v2.1 wordmark below.
# About 22 lines tall × 23 cols wide — fits in a side column on most terminals.
LOGO = r"""
       ╓──────────╖
    ╓──╜          ╚──╖
  ╔═╝   ██╗   ██╗   ╚═╗
  ║     ██║   ██║     ║
  ║     ██║   ██║     ║
  ║     ╚██╗ ██╔╝     ║
  ║      ╚████╔╝      ║
  ║       ╚═══╝       ║
  ║      ██████╗      ║
  ║      ╚════██╗     ║
  ║       █████╔╝     ║
  ║      ██╔═══╝      ║
  ║      ███████╗     ║
  ╚═╗    ╚══════╝   ╔═╝
    ╙──╖          ╓──╜
       ╙──────────╜
      ━━━━━━━━━━━━━━
       ◈  HERMES  ◈
       ◈   WORK   ◈
            v2.1
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
    console.print(Text(BANNER, style="bold cyan"), highlight=False)
    tagline = Text()
    tagline.append("    Personal Work Agent  ", style="dim italic")
    tagline.append("·", style="dim")
    tagline.append("  Max-subscription billing  ", style="dim italic")
    tagline.append("·", style="dim")
    tagline.append(f"  v{__version__}", style="dim italic")
    console.print(tagline)
    console.print()
    console.print(_welcome_panel(settings, session_name))
    console.print()


def _welcome_panel(settings: AgentSettings, session_name: str | None) -> Panel:
    # Build the right-side info column.
    info_lines: list[Text] = [Text()]

    # Available Tools section
    info_lines.append(Text("  Available Tools", style="bold green"))
    for group, tools in TOOL_GROUPS.items():
        line = Text("    ")
        line.append(f"{group:<8}", style="dim")
        line.append(": ", style="dim")
        line.append(", ".join(tools), style="white")
        info_lines.append(line)
    if settings.mcp_servers:
        line = Text("    ")
        line.append(f"{'mcp':<8}", style="dim")
        line.append(": ", style="dim")
        line.append(", ".join(settings.mcp_servers.keys()), style="magenta")
        info_lines.append(line)
    info_lines.append(Text())

    # Available Skills section
    skills = discover_skills(limit=8)
    if skills:
        info_lines.append(Text("  Available Skills", style="bold green"))
        for name, desc in skills:
            line = Text("    ")
            line.append(f"{name}", style="cyan")
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
        line.append(str(val), style="cyan")
        info_lines.append(line)

    # Session line — prominent, formatted like upstream hermes-agent
    session_line = Text("  ")
    session_line.append(f"{'Session':<12}", style="dim")
    session_line.append(session_name or "(ephemeral)", style="bold yellow")
    info_lines.append(session_line)
    info_lines.append(Text())

    # Counts + help footer
    total_tools = len(DEFAULT_TOOLS) + sum(
        1 for _ in settings.mcp_servers
    )
    counts = Text("  ")
    counts.append(f"{total_tools} tools", style="green")
    counts.append(" · ", style="dim")
    counts.append(f"{len(skills)} skills" if skills else "0 skills", style="green")
    counts.append(" · ", style="dim")
    counts.append("/help for commands · /exit quits", style="dim")
    info_lines.append(counts)
    info_lines.append(Text())

    # Left-side logo column.
    logo = Text(LOGO, style="magenta")

    # Compose side-by-side via Columns.
    from rich.columns import Columns
    body = Columns(
        [logo, Group(*info_lines)],
        equal=False,
        expand=False,
        padding=(0, 2),
    )

    title = Text()
    title.append(" Hermesv2.1 ", style="bold cyan")
    title.append("(work)", style="bold yellow")
    title.append(" · ", style="dim")
    title.append(settings.model, style="green")
    title.append(" · ", style="dim")
    title.append("Max subscription ", style="bold magenta")

    return Panel(
        body,
        title=title,
        title_align="left",
        border_style="cyan",
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
        return f"[bold magenta]{verb}...[/]"

    def start(self) -> None:
        if self._status is not None:
            return
        verb = next(self._verbs)
        self._status = self._console.status(self._label(verb), spinner="dots", spinner_style="magenta")
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
    return "\n[bold cyan]▎[/] [bold cyan]you[/] [bold magenta]❱[/] "


def render_text_delta(console: Console, text: str) -> None:
    console.print(text, end="", soft_wrap=True, highlight=False)


def render_assistant_markdown(console: Console, text: str) -> None:
    """Render a buffered assistant text segment as Markdown.

    Used instead of streaming raw text so things like **bold**, bullet lists,
    `inline code`, and code fences render with proper styling.
    """
    if not text.strip():
        return
    from rich.markdown import Markdown
    md = Markdown(text, code_theme="monokai", inline_code_lexer="python")
    console.print(md)


def assistant_label(console: Console) -> None:
    # Full line, so the following markdown block renders cleanly below.
    console.print("[bold magenta]▎ hermes ❰[/]")


def render_thinking_delta(console: Console, text: str) -> None:
    console.print(f"[dim italic]{text}[/]", end="", soft_wrap=True, highlight=False)


def render_tool_call(console: Console, name: str, tool_input: dict) -> None:
    preview = str(tool_input)
    if len(preview) > 200:
        preview = preview[:200] + "..."
    console.print(f"\n  [cyan]⚙ {name}[/] [dim]{preview}[/]")


def render_tool_result(console: Console, name: str, output: str, is_error: bool) -> None:
    color = "red" if is_error else "green"
    preview = output if len(output) < 400 else output[:400] + "..."
    console.print(f"  [{color}]↳ {name}[/] [dim]{preview}[/]")


def render_turn_footer(
    console: Console,
    stop_reason: str,
    cost_usd: float | None,
    usage: dict,
) -> None:
    line = Text()
    line.append("  ⚕ ", style="bold magenta")
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
    line.append("Max subscription", style="magenta")
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
    parts.append(" ⚕ ", style="bold magenta")
    parts.append(model, style="cyan")
    parts.append(" │ ", style="dim")
    parts.append(f"⌖ {session_id}", style="bold blue")
    parts.append(" │ ", style="dim")
    parts.append(ctx_str, style="green" if ctx_pct and ctx_pct < 70 else "yellow")
    parts.append(" │ ", style="dim")
    parts.append(bar, style="dim")
    parts.append(" │ ", style="dim")
    parts.append(f"{_fmt_seconds(session_seconds)}", style="dim")
    parts.append(" │ ⏲ ", style="dim")
    parts.append(f"{_fmt_seconds(turn_seconds)}", style="dim")
    console.print(parts)


def render_sessions(console: Console, sessions: list) -> None:
    """Print a table of saved Claude Code sessions."""
    from datetime import datetime

    from rich.table import Table

    table = Table(border_style="dim", header_style="bold cyan")
    table.add_column("Session ID", style="cyan", no_wrap=True)
    table.add_column("Modified", style="dim")
    table.add_column("Branch / cwd", style="dim")
    table.add_column("Summary", style="white")

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
    ("/exit",       "Quit hermesv2"),
    ("/quit",       "Quit hermesv2 (alias for /exit)"),
    # Session control
    ("/new",        "Start a new session (fresh history)"),
    ("/reset",      "Start a new session (alias for /new)"),
    ("/clear",      "Clear the screen"),
    ("/redraw",     "Re-render banner + welcome panel"),
    ("/title",      "Set a title for the current session (usage: /title <name>)"),
    ("/history",    "Show recent user prompts in this session"),
    ("/retry",      "Re-send the last user prompt to the agent"),
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
    ("/update",     "git pull the latest hermesv2 from origin"),
]


def render_help(console: Console) -> None:
    console.print("\n[bold cyan]Slash commands[/]")
    for cmd, desc in SLASH_COMMANDS:
        console.print(f"  [yellow]{cmd:<12}[/]  [dim]{desc}[/]")
    console.print()


def render_tools(console: Console) -> None:
    console.print("\n[bold green]Built-in tools[/]")
    for group, tools in TOOL_GROUPS.items():
        line = Text("  ")
        line.append(f"{group:<8}", style="dim")
        line.append(", ".join(tools), style="white")
        console.print(line)
    console.print()


def render_stats(
    console: Console, turns: int, total_input: int, total_output: int,
    total_cost: float, session_seconds: float,
) -> None:
    line = Text()
    line.append("\n  session  ", style="bold cyan")
    line.append(f"{turns} turn{'s' if turns != 1 else ''}", style="green")
    line.append(" · ", style="dim")
    line.append(f"in {total_input} / out {total_output} tokens", style="dim")
    line.append(" · ", style="dim")
    if total_cost > 0:
        line.append(f"~${total_cost:.4f} equiv", style="dim")
        line.append(" · ", style="dim")
    line.append(f"{_fmt_seconds(session_seconds)} elapsed", style="dim")
    line.append(" · ", style="dim")
    line.append("billed to Max subscription", style="magenta")
    console.print(line)
    console.print()
