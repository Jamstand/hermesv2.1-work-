"""Terminal UI helpers for the `hermesv2 chat` REPL.

Renders an ANSI Shadow banner and bordered welcome panel similar to the
upstream NousResearch hermes-agent TUI, then a styled prompt + tool-call
trace + per-turn footer for each interaction.

Only used by `hermesv2 chat`. `hermesv2 run` stays plain so it pipes cleanly.
"""

from __future__ import annotations

from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text

from hermesv2 import __version__
from hermesv2.agent import DEFAULT_TOOLS
from hermesv2.config import AgentSettings

BANNER = r"""
██╗  ██╗███████╗██████╗ ███╗   ███╗███████╗███████╗    ██╗   ██╗██████╗
██║  ██║██╔════╝██╔══██╗████╗ ████║██╔════╝██╔════╝    ██║   ██║╚════██╗
███████║█████╗  ██████╔╝██╔████╔██║█████╗  ███████╗    ██║   ██║ █████╔╝
██╔══██║██╔══╝  ██╔══██╗██║╚██╔╝██║██╔══╝  ╚════██║    ╚██╗ ██╔╝██╔═══╝
██║  ██║███████╗██║  ██║██║ ╚═╝ ██║███████╗███████║     ╚████╔╝ ███████╗
╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝╚══════╝      ╚═══╝  ╚══════╝
"""

TOOL_GROUPS: dict[str, list[str]] = {
    "files":  ["Read", "Write", "Edit"],
    "search": ["Grep", "Glob"],
    "shell":  ["Bash"],
    "web":    ["WebFetch", "WebSearch"],
}


def render_startup(console: Console, settings: AgentSettings) -> None:
    """Print the banner + welcome panel. Called once at the start of `chat`."""
    console.print(Text(BANNER, style="bold cyan"), highlight=False)
    console.print(_welcome_panel(settings))
    console.print()


def _welcome_panel(settings: AgentSettings) -> Panel:
    lines: list[Text] = [Text()]

    lines.append(Text("  Built-in tools", style="bold green"))
    for group, tools in TOOL_GROUPS.items():
        line = Text("    ")
        line.append(f"{group:<7}", style="dim")
        line.append("  ")
        line.append(", ".join(tools), style="white")
        lines.append(line)
    lines.append(Text())

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
        lines.append(line)
    lines.append(Text())

    footer = Text("  ")
    footer.append("/reset", style="bold yellow")
    footer.append(" clears history · ", style="dim")
    footer.append("/exit", style="bold yellow")
    footer.append(" or Ctrl-D quits · ", style="dim")
    footer.append(f"{len(DEFAULT_TOOLS)} tools available", style="dim")
    lines.append(footer)
    lines.append(Text())

    title = Text()
    title.append(" Hermes v2 ", style="bold cyan")
    title.append(f"v{__version__} ", style="dim")
    title.append("· ", style="dim")
    title.append(settings.model, style="green")
    title.append(" · ", style="dim")
    title.append("Max subscription ", style="bold magenta")

    return Panel(
        Group(*lines),
        title=title,
        title_align="left",
        border_style="cyan",
        padding=(0, 1),
    )


def prompt_label() -> str:
    return "\n[bold blue]you›[/] "


def assistant_label(console: Console) -> None:
    console.print("[bold magenta]hermes›[/] ", end="")


def render_text_delta(console: Console, text: str) -> None:
    console.print(text, end="", soft_wrap=True, highlight=False)


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
