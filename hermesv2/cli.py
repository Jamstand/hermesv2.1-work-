"""Hermes v2 CLI: `hermesv2 [chat] | run | doctor | slack | discord | update`."""

from __future__ import annotations

import asyncio
import secrets
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import click
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion, FuzzyCompleter
from prompt_toolkit.formatted_text import HTML, FormattedText
from prompt_toolkit.styles import Style
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
# Slash command autocomplete + theming (prompt-toolkit)
# ---------------------------------------------------------------------------


# Color palette is harmonized with the welcome panel: cyan border (#87d7ff),
# magenta accents (#ff5fd7), navy menu background (#1c1c2e), indigo highlight
# for the selected row (#5f5fff).
DROPDOWN_STYLE = Style.from_dict({
    # Menu rows
    "completion-menu.completion":                     "bg:#1c1c2e #87d7ff",
    "completion-menu.completion.current":             "bg:#5f5fff #ffffff bold",
    "completion-menu.meta.completion":                "bg:#1c1c2e #888888",
    "completion-menu.meta.completion.current":        "bg:#5f5fff #d0d0d0",
    "completion-menu.multi-column-meta":              "bg:#1c1c2e #888888",
    # Scrollbar
    "scrollbar.background":                           "bg:#1c1c2e",
    "scrollbar.button":                               "bg:#5f5fff",
    # Inline classes used by FormattedText below
    "slash":                                          "#ff5fd7",
    "cmd-name":                                       "#87d7ff bold",
})


class SlashCommandCompleter(Completer):
    """Yields the full slash-command set whenever the user is typing a `/`.

    Yields everything (no prefix filter) on purpose — when wrapped in
    FuzzyCompleter, that wrapper does the matching. When used bare, all
    commands are listed and prompt-toolkit's default key bindings narrow
    them as you type.
    """

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if not text.startswith("/"):
            return
        for cmd, desc in tui.SLASH_COMMANDS:
            yield Completion(
                cmd,
                start_position=-len(text),
                display=FormattedText([
                    ("class:slash",    "/"),
                    ("class:cmd-name", cmd[1:]),
                ]),
                display_meta=desc,
            )


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
@click.option("--session", "session_name", default=None,
              help="Named session to start/resume. Persists across runs.")
@click.pass_context
def chat(ctx: click.Context, session_name: str | None) -> None:
    """Interactive REPL. Default if no subcommand is given."""
    cfg = load_config(ctx.obj.get("config_path") if ctx.obj else None)
    asyncio.run(_chat(cfg, session_name))


def _generate_session_id() -> str:
    """Timestamp + random suffix, format `20260512_112656_5fcccc`."""
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(3)}"


async def _chat(cfg: Config, session_name: str | None = None) -> None:
    session_id = session_name or _generate_session_id()
    tui.render_startup(console, cfg.agent, session_name=session_id)
    stats = SessionStats()
    user_history: list[str] = []

    prompt_session: PromptSession[str] = PromptSession(
        completer=FuzzyCompleter(SlashCommandCompleter()),
        complete_while_typing=True,
        complete_in_thread=True,
        style=DROPDOWN_STYLE,
    )

    async with Agent(
        cfg.agent,
        cwd=cfg.agent.workspace_dir,
        resume_session_id=session_name,
    ) as agent:
        current_model = cfg.agent.model

        tui.render_status_bar(console, current_model, None, 0.0, 0.0, session_id)

        while True:
            try:
                # The blank line is printed separately — keeping the prompt
                # itself a pure single line avoids prompt-toolkit's cursor
                # math going wrong on redraw (which made the prompt vanish
                # when backspacing through `/x` completion text).
                console.print()
                line = await prompt_session.prompt_async(
                    HTML(
                        "<b><ansicyan>▎</ansicyan></b> "
                        "<b><ansicyan>you</ansicyan></b> "
                        "<b><ansimagenta>❱</ansimagenta></b> "
                    )
                )
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]bye.[/]")
                return
            line = line.strip()
            if not line:
                continue

            if line.startswith("/"):
                action = await _handle_slash(
                    line, agent, stats, current_model, cfg, user_history,
                )
                if action == "exit":
                    return
                if isinstance(action, tuple) and action[0] == "model":
                    current_model = action[1]
                if action == "reset":
                    session_id = agent.reset_session()
                if action == "redraw":
                    tui.render_startup(console, cfg.agent, session_name=session_id)
                tui.render_status_bar(
                    console, current_model, await _ctx_pct(agent),
                    stats.last_turn_seconds, stats.session_seconds, session_id,
                )
                continue

            user_history.append(line)
            stats.start_turn()

            spinner = tui.ThinkingSpinner(console)
            spinner.start()
            first_event = True
            try:
                async for event in agent.run_stream(line, session_id=session_id):
                    if first_event:
                        spinner.stop()
                        tui.assistant_label(console)
                        first_event = False
                    _render_event_tui(event, stats)
            finally:
                spinner.stop()

            console.print()
            tui.render_status_bar(
                console, current_model, await _ctx_pct(agent),
                stats.last_turn_seconds, stats.session_seconds, session_id,
            )


async def _handle_slash(
    line: str,
    agent: Agent,
    stats: SessionStats,
    current_model: str,
    cfg: Config,
    user_history: list[str],
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
    if cmd in ("/reset", "/new"):
        console.print("[dim](starting fresh session)[/]")
        user_history.clear()
        return "reset"
    if cmd == "/redraw":
        console.clear()
        return "redraw"
    if cmd == "/title":
        if not arg:
            console.print("  [yellow]usage:[/] /title <name>")
            return None
        try:
            from claude_agent_sdk import rename_session
            rename_session(agent._client._session_id if agent._client else "default", arg)
            console.print(f"  [green]session renamed to[/] [cyan]{arg}[/]")
        except Exception as e:  # noqa: BLE001
            console.print(f"[red]rename failed: {e}[/]")
        return None
    if cmd == "/history":
        if not user_history:
            console.print("[dim]no user prompts yet in this session.[/]")
            return None
        console.print("\n[bold cyan]Your prompts this session[/]")
        for i, prompt in enumerate(user_history[-20:], 1):
            short = prompt if len(prompt) < 80 else prompt[:80] + "..."
            console.print(f"  [dim]{i:>2}.[/] {short}")
        console.print()
        return None
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
    if cmd == "/sessions":
        await _list_sessions_async()
        return None

    console.print(f"[red]Unknown command: {cmd}. Type /help for a list.[/]")
    return None


async def _list_sessions_async() -> None:
    from claude_agent_sdk import list_sessions
    try:
        sessions = await asyncio.to_thread(list_sessions, limit=20)
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]Failed to list sessions: {e}[/]")
        return
    if not sessions:
        console.print("[dim]No saved sessions yet.[/]")
        return
    tui.render_sessions(console, sessions)


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


@main.command(name="index")
@click.argument("directory", type=click.Path(exists=True, file_okay=False))
@click.option("--rebuild", is_flag=True, help="Wipe the index before re-walking.")
def index_cmd(directory: str, rebuild: bool) -> None:
    """Index a directory of text/markdown files for `hermesv2 search`."""
    from hermesv2.search import index_directory
    n = index_directory(directory, rebuild=rebuild)
    console.print(f"[green]Indexed {n} files[/] from [cyan]{directory}[/]")


@main.command(name="search")
@click.argument("query", nargs=-1, required=True)
@click.option("--limit", default=10, help="Max hits to return.")
@click.option("--json", "as_json", is_flag=True, help="Machine-readable output.")
def search_cmd(query: tuple[str, ...], limit: int, as_json: bool) -> None:
    """Full-text search over your indexed notes (FTS5)."""
    from hermesv2.search import render_hits, render_hits_json, search
    q = " ".join(query)
    try:
        hits = search(q, limit=limit)
    except ValueError as e:
        console.print(f"[red]{e}[/]")
        sys.exit(1)
    if as_json:
        print(render_hits_json(hits))
    else:
        console.print(render_hits(hits))


@main.command(name="voice")
@click.argument("audio_file", type=click.Path(exists=True, dir_okay=False))
@click.option("--model", default="small", help="Whisper model: tiny, base, small, medium, large-v3.")
@click.option("--language", default=None, help="ISO language code; auto-detect if omitted.")
@click.option("--run", "auto_run", is_flag=True, help="Pipe the transcript into `hermesv2 run`.")
def voice_cmd(audio_file: str, model: str, language: str | None, auto_run: bool) -> None:
    """Transcribe an audio file using local Whisper, optionally run it as a prompt."""
    from hermesv2.voice import transcribe
    try:
        console.print(f"[dim]transcribing {audio_file} with whisper-{model}...[/]")
        text = transcribe(audio_file, model_name=model, language=language)
    except RuntimeError as e:
        console.print(f"[red]{e}[/]")
        sys.exit(1)

    console.print(f"\n[bold cyan]Transcript:[/] {text}\n")
    if not auto_run or not text:
        return

    cfg = load_config()
    asyncio.run(_run_one(cfg, text))


@main.command(name="sessions")
@click.option("--delete", "delete_id", default=None, help="Delete a session by ID.")
@click.option("--limit", default=20, help="Max sessions to list.")
def sessions_cmd(delete_id: str | None, limit: int) -> None:
    """List, inspect, or delete saved Claude Code sessions."""
    from claude_agent_sdk import delete_session, list_sessions

    if delete_id:
        try:
            delete_session(delete_id)
            console.print(f"[green]Deleted session {delete_id}.[/]")
        except Exception as e:  # noqa: BLE001
            console.print(f"[red]Delete failed: {e}[/]")
        return

    try:
        sessions = list_sessions(limit=limit)
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]Failed to list sessions: {e}[/]")
        sys.exit(1)
    if not sessions:
        console.print("[dim]No saved sessions yet.[/]")
        return
    tui.render_sessions(console, sessions)


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

        # Real connectivity test: actually ping Anthropic via the CLI.
        try:
            ping = subprocess.run(
                ["claude", "-p", "respond with exactly: pong"],
                capture_output=True, text=True, timeout=60,
                env={**__import__("os").environ, "ANTHROPIC_API_KEY": ""},
            )
            ok = ping.returncode == 0 and "pong" in (ping.stdout or "").lower()
            msg = (ping.stdout or ping.stderr or "(no output)").strip().splitlines()[0][:80]
            checks.append(("Anthropic reachable (claude -p)", ok, msg))
        except Exception as e:  # noqa: BLE001
            checks.append(("Anthropic reachable (claude -p)", False, f"{type(e).__name__}: {e}"))

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
