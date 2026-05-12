"""Hermes v2 CLI: `hermesv2 chat | run | doctor | slack | discord`."""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import sys
from pathlib import Path

import click
from rich.console import Console

from hermesv2.agent import (
    Agent,
    TextDelta,
    ThinkingDelta,
    ToolCall,
    ToolResult,
    TurnDone,
)
from hermesv2.config import Config, load_config

console = Console()


def _render_event(event: object) -> None:
    if isinstance(event, TextDelta):
        console.print(event.text, end="", soft_wrap=True, highlight=False)
    elif isinstance(event, ThinkingDelta):
        console.print(
            f"[dim italic]{event.text}[/]", end="", soft_wrap=True, highlight=False
        )
    elif isinstance(event, ToolCall):
        preview = str(event.input)
        if len(preview) > 200:
            preview = preview[:200] + "..."
        console.print(f"\n[cyan]→ {event.name}[/] [dim]{preview}[/]")
    elif isinstance(event, ToolResult):
        color = "red" if event.is_error else "green"
        preview = event.output if len(event.output) < 400 else event.output[:400] + "..."
        console.print(f"[{color}]← {event.name}[/] [dim]{preview}[/]")
    elif isinstance(event, TurnDone):
        cost = "Max subscription" if not event.cost_usd else f"${event.cost_usd:.4f}"
        console.print(
            f"\n[dim](stop={event.stop_reason}, billing={cost})[/]"
        )


@click.group()
@click.version_option()
def main() -> None:
    """Hermes v2 — personal agent on top of Claude Code (uses your Max subscription)."""


@main.command()
@click.option("--config", "config_path", default=None, help="Path to config YAML.")
@click.argument("prompt", nargs=-1)
def run(config_path: str | None, prompt: tuple[str, ...]) -> None:
    """Run a single prompt and exit. Reads from stdin if no prompt is given."""
    if prompt:
        message = " ".join(prompt)
    elif not sys.stdin.isatty():
        message = sys.stdin.read().strip()
    else:
        console.print("[red]No prompt given. Pass one as args or pipe via stdin.[/]")
        sys.exit(2)

    cfg = load_config(config_path)
    asyncio.run(_run_one(cfg, message))


async def _run_one(cfg: Config, message: str) -> None:
    async with Agent(cfg.agent, cwd=cfg.agent.workspace_dir) as agent:
        async for event in agent.run_stream(message):
            _render_event(event)
    console.print()


@main.command()
@click.option("--config", "config_path", default=None, help="Path to config YAML.")
def chat(config_path: str | None) -> None:
    """Interactive REPL. /reset clears history; Ctrl-D or 'exit' to quit."""
    cfg = load_config(config_path)
    asyncio.run(_chat(cfg))


async def _chat(cfg: Config) -> None:
    console.print(
        f"[bold green]Hermes v2[/] — model [cyan]{cfg.agent.model}[/], "
        f"effort [cyan]{cfg.agent.effort}[/], "
        f"workspace [cyan]{cfg.agent.workspace_dir}[/]. "
        "Type /reset to clear history."
    )
    async with Agent(cfg.agent, cwd=cfg.agent.workspace_dir) as agent:
        session_id = "default"
        while True:
            try:
                line = await asyncio.to_thread(console.input, "\n[bold blue]you›[/] ")
            except (EOFError, KeyboardInterrupt):
                console.print("\nbye.")
                return
            line = line.strip()
            if not line:
                continue
            if line in ("exit", "quit", "/exit", "/quit"):
                return
            if line == "/reset":
                session_id = agent.reset_session()
                console.print("[dim](history cleared)[/]")
                continue

            console.print("[bold magenta]hermes›[/] ", end="")
            async for event in agent.run_stream(line, session_id=session_id):
                _render_event(event)
            console.print()


@main.command()
def doctor() -> None:
    """Diagnose the install: Claude CLI present, logged in, config valid, etc."""
    checks: list[tuple[str, bool, str]] = []

    claude_path = shutil.which("claude")
    checks.append(
        ("Claude Code CLI on PATH", bool(claude_path), claude_path or "not found")
    )

    if claude_path:
        try:
            ver = subprocess.run(
                ["claude", "--version"], capture_output=True, text=True, timeout=10
            )
            checks.append(
                (
                    "Claude CLI runs",
                    ver.returncode == 0,
                    (ver.stdout or ver.stderr).strip() or "(no output)",
                )
            )
        except Exception as e:  # noqa: BLE001
            checks.append(("Claude CLI runs", False, f"{type(e).__name__}: {e}"))

        auth_dir = Path.home() / ".claude"
        checks.append(
            (
                "Claude auth state present",
                auth_dir.exists(),
                str(auth_dir) if auth_dir.exists() else "run `claude login`",
            )
        )

    try:
        cfg = load_config()
        checks.append(
            (
                "Config loaded",
                True,
                f"model={cfg.agent.model}, effort={cfg.agent.effort}, "
                f"perm={cfg.agent.permission_mode}",
            )
        )
        workspace = Path(cfg.agent.workspace_dir).expanduser()
        workspace.mkdir(parents=True, exist_ok=True)
        test = workspace / ".write_test"
        try:
            test.write_text("ok")
            test.unlink()
            checks.append(("Workspace writable", True, str(workspace)))
        except Exception as e:  # noqa: BLE001
            checks.append(("Workspace writable", False, str(e)))

        slack_set = bool(cfg.slack.get("bot_token") and cfg.slack.get("app_token"))
        checks.append(
            (
                "Slack tokens (optional)",
                slack_set,
                "set" if slack_set else "unset — only needed for `hermesv2 slack`",
            )
        )
        discord_set = bool(cfg.discord.get("bot_token"))
        checks.append(
            (
                "Discord token (optional)",
                discord_set,
                "set" if discord_set else "unset — only needed for `hermesv2 discord`",
            )
        )
    except Exception as e:  # noqa: BLE001
        checks.append(("Config loaded", False, f"{type(e).__name__}: {e}"))

    required_ok = True
    for name, ok, msg in checks:
        sym = "[green]OK[/]" if ok else "[red]X[/]"
        console.print(f"  {sym}  [bold]{name}[/]  [dim]{msg}[/]")
        if not ok and "optional" not in name:
            required_ok = False

    if required_ok:
        console.print(
            "\n[bold green]All required checks passed.[/] You should be able to run "
            "`hermesv2 run \"hi\"`."
        )
    else:
        console.print(
            "\n[bold red]Some required checks failed.[/] Common fixes:\n"
            "  - Install Claude Code: [cyan]npm install -g @anthropic-ai/claude-code[/]\n"
            "  - Log in to your Max account: [cyan]claude login[/]"
        )
        sys.exit(1)


@main.command()
@click.option("--config", "config_path", default=None, help="Path to config YAML.")
def slack(config_path: str | None) -> None:
    """Run the Slack bot (Socket Mode)."""
    from hermesv2.platforms.slack import run_slack_bot

    cfg = load_config(config_path)
    asyncio.run(run_slack_bot(cfg))


@main.command()
@click.option("--config", "config_path", default=None, help="Path to config YAML.")
def discord(config_path: str | None) -> None:
    """Run the Discord bot."""
    from hermesv2.platforms.discord import run_discord_bot

    cfg = load_config(config_path)
    asyncio.run(run_discord_bot(cfg))
