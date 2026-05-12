"""Hermes v2 CLI: `hermesv2 [chat] | run | doctor | slack | discord | update`."""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import sys
import time
from pathlib import Path

import click
from rich.console import Console

from hermesv2 import tui
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


# ---------------------------------------------------------------------------
# Event rendering
# ---------------------------------------------------------------------------


def _render_event_plain(event: object) -> None:
    """Compact, pipe-friendly rendering for `hermesv2 run`."""
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


def _render_event_tui(event: object, stats: SessionStats) -> None:
    """Fancy rendering for `hermesv2 chat`. Updates running stats too."""
    if isinstance(event, TextDelta):
        tui.render_text_delta(console, event.text)
    elif isinstance(event, ThinkingDelta):
        tui.render_thinking_delta(console, event.text)
    elif isinstance(event, ToolCall):
        tui.render_tool_call(console, event.name, event.input)
    elif isinstance(event, ToolResult):
        tui.render_tool_result(console, event.name, event.output, event.is_error)
    elif isinstance(event, TurnDone):
        tui.render_turn_footer(console, event.stop_reason, event.cost_usd, event.usage)
        stats.record(event)


# ---------------------------------------------------------------------------
# Session stats
# ---------------------------------------------------------------------------


class SessionStats:
    def __init__(self) -> None:
        self.start_ts = time.monotonic()
        self.turn_start_ts = self.start_ts
        self.turn_end_ts = self.start_ts
        self.turns = 0
        self.total_input = 0
        self.total_output = 0
        self.total_cost = 0.0

    def start_turn(self) -> None:
        self.turn_start_ts = time.monotonic()

    def record(self, event: TurnDone) -> None:
        self.turn_end_ts = time.monotonic()
        self.turns += 1
        usage = event.usage or {}
        self.total_input += usage.get("input_tokens") or 0
        self.total_output += usage.get("output_tokens") or 0
        if event.cost_usd:
            self.total_cost += event.cost_usd

    @property
    def session_seconds(self) -> float:
        return time.monotonic() - self.start_ts

    @property
    def last_turn_seconds(self) -> float:
        return max(0.0, self.turn_end_ts - self.turn_start_ts)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.group(invoke_without_command=True)
@click.option("--config", "config_path", default=None, help="Path to config YAML.")
@click.version_option()
@click.pass_context
def main(ctx: click.Context, config_path: str | None) -> None:
    """Hermes v2 — personal agent on top of Claude Code (uses your Max subscription)."""
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config_path
    if ctx.invoked_subcommand is None:
        ctx.invoke(chat)


@main.command()
@click.argument("prompt", nargs=-1)
@click.pass_context
def run(ctx: click.Context, prompt: tuple[str, ...]) -> None:
    """Run a single prompt and exit. Reads from stdin if no prompt is given."""
    if prompt:
        message = " ".join(prompt)
    elif not sys.stdin.isatty():
        message = sys.stdin.read().strip()
    else:
        console.print("[red]No prompt given. Pass one as args or pipe via stdin.[/]")
        sys.exit(2)

    cfg = load_config(ctx.obj.get("config_path"))
    asyncio.run(_run_one(cfg, message))


async def _run_one(cfg: Config, message: str) -> None:
    async with Agent(cfg.agent, cwd=cfg.agent.workspace_dir) as agent:
        async for event in agent.run_stream(message):
            _render_event_plain(event)
    console.print()


@main.command()
@click.pass_context
def chat(ctx: click.Context) -> None:
    """Interactive REPL. Default if no subcommand is given."""
    cfg = load_config(ctx.obj.get("config_path") if ctx.obj else None)
    asyncio.run(_chat(cfg))


async def _chat(cfg: Config) -> None:
    tui.render_startup(console, cfg.agent)
    stats = SessionStats()

    async with Agent(cfg.agent, cwd=cfg.agent.workspace_dir) as agent:
        session_id = "default"
        current_model = cfg.agent.model

        tui.render_status_bar(console, current_model, None, 0.0, 0.0)

        while True:
            try:
                line = await asyncio.to_thread(console.input, tui.prompt_label())
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]bye.[/]")
                return
            line = line.strip()
            if not line:
                continue

            if line.startswith("/"):
                action = await _handle_slash(line, agent, stats, current_model)
                if action == "exit":
                    return
                if isinstance(action, tuple) and action[0] == "model":
                    current_model = action[1]
                tui.render_status_bar(
                    console, current_model, await _ctx_pct(agent),
                    stats.last_turn_seconds, stats.session_seconds,
                )
                if action == "reset":
                    session_id = agent.reset_session()
                continue

            tui.assistant_label(console)
            stats.start_turn()
            async for event in agent.run_stream(line, session_id=session_id):
                _render_event_tui(event, stats)
            console.print()
            tui.render_status_bar(
                console, current_model, await _ctx_pct(agent),
                stats.last_turn_seconds, stats.session_seconds,
            )


async def _handle_slash(
    line: str, agent: Agent, stats: SessionStats, current_model: str
) -> str | tuple[str, str] | None:
    parts = line.split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if cmd in ("/exit", "/quit"):
        return "exit"
    if cmd == "/help":
        tui.render_help(console)
        return None
    if cmd == "/clear":
        console.clear()
        return None
    if cmd == "/reset":
        console.print("[dim](history cleared)[/]")
        return "reset"
    if cmd == "/tools":
        tui.render_tools(console)
        return None
    if cmd == "/stats":
        tui.render_stats(
            console, stats.turns, stats.total_input, stats.total_output,
            stats.total_cost, stats.session_seconds,
        )
        return None
    if cmd == "/context":
        usage = await _get_context_usage(agent)
        if usage is None:
            console.print("[dim]Context usage unavailable.[/]")
        else:
            console.print(f"  [dim]Context: {usage}[/]")
        return None
    if cmd == "/model":
        if not arg:
            console.print(f"  current model: [cyan]{current_model}[/]")
            return None
        if agent._client is None:
            console.print("[red]Agent not connected.[/]")
            return None
        try:
            await agent._client.set_model(arg)
            console.print(f"  [green]switched model to[/] [cyan]{arg}[/]")
            return ("model", arg)
        except Exception as e:  # noqa: BLE001
            console.print(f"[red]Failed to switch model: {e}[/]")
            return None
    if cmd == "/update":
        await _run_update_async()
        return None

    console.print(f"[red]Unknown command: {cmd}. Type /help for a list.[/]")
    return None


async def _ctx_pct(agent: Agent) -> float | None:
    """Best-effort context-usage percentage. Returns None if unsupported."""
    if agent._client is None:
        return None
    try:
        usage = await _maybe_await(agent._client.get_context_usage())
        # ContextUsageResponse field name has shifted across SDK versions —
        # try the common ones, fall back to None.
        for attr in ("percentage", "context_percentage", "used_pct"):
            val = getattr(usage, attr, None)
            if val is not None:
                return float(val)
        used = getattr(usage, "tokens_used", None) or getattr(usage, "used", None)
        total = getattr(usage, "total_tokens", None) or getattr(usage, "max_tokens", None)
        if used and total:
            return 100.0 * float(used) / float(total)
    except Exception:  # noqa: BLE001
        return None
    return None


async def _get_context_usage(agent: Agent) -> str | None:
    if agent._client is None:
        return None
    try:
        usage = await _maybe_await(agent._client.get_context_usage())
        return str(usage)
    except Exception as e:  # noqa: BLE001
        return f"(error: {e})"


async def _maybe_await(value):
    if hasattr(value, "__await__"):
        return await value
    return value


async def _run_update_async() -> None:
    await asyncio.to_thread(_run_update)


def _run_update() -> None:
    repo = _find_repo_root()
    if repo is None:
        console.print("[red]Couldn't locate the hermesv2 git checkout.[/]")
        return
    console.print(f"[dim]git pull in {repo}...[/]")
    proc = subprocess.run(
        ["git", "-C", str(repo), "pull"], capture_output=True, text=True, timeout=60
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    console.print(out.strip() or "(no output)")
    if proc.returncode == 0:
        console.print(
            "[green]Update fetched.[/] [dim]Restart hermesv2 chat to pick up changes.[/]"
        )
    else:
        console.print(f"[red]git pull failed (exit {proc.returncode}).[/]")


def _find_repo_root() -> Path | None:
    here = Path(__file__).resolve()
    for ancestor in [here, *here.parents]:
        if (ancestor / ".git").is_dir():
            return ancestor
    return None


@main.command()
def update() -> None:
    """`git pull` the latest hermesv2 from origin."""
    _run_update()


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
            "\n[bold green]All required checks passed.[/] Just type [cyan]hermesv2[/]."
        )
    else:
        console.print(
            "\n[bold red]Some required checks failed.[/] Common fixes:\n"
            "  - Install Claude Code: [cyan]npm install -g @anthropic-ai/claude-code[/]\n"
            "  - Log in to your Max account: [cyan]claude login[/]"
        )
        sys.exit(1)


@main.command()
@click.pass_context
def slack(ctx: click.Context) -> None:
    """Run the Slack bot (Socket Mode)."""
    from hermesv2.platforms.slack import run_slack_bot

    cfg = load_config(ctx.obj.get("config_path") if ctx.obj else None)
    asyncio.run(run_slack_bot(cfg))


@main.command()
@click.pass_context
def discord(ctx: click.Context) -> None:
    """Run the Discord bot."""
    from hermesv2.platforms.discord import run_discord_bot

    cfg = load_config(ctx.obj.get("config_path") if ctx.obj else None)
    asyncio.run(run_discord_bot(cfg))
