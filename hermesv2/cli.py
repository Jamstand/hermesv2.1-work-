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
                    line, agent, stats, current_model, cfg, user_history, session_id,
                )
                if action == "exit":
                    return
                if isinstance(action, tuple):
                    if action[0] == "model":
                        current_model = action[1]
                    elif action[0] == "session":
                        session_id = action[1]
                        user_history.clear()
                    elif action[0] == "retry":
                        # Replay last prompt as if user just sent it again.
                        line = action[1]
                        # Fall through into the streaming block below.
                    else:
                        pass
                if action == "reset":
                    session_id = agent.reset_session()
                if action == "redraw":
                    tui.render_startup(console, cfg.agent, session_name=session_id)
                # If retry, fall through to the streaming block; otherwise loop.
                if not (isinstance(action, tuple) and action[0] == "retry"):
                    tui.render_status_bar(
                        console, current_model, await _ctx_pct(agent),
                        stats.last_turn_seconds, stats.session_seconds, session_id,
                    )
                    continue
                console.print(f"  [dim](retrying: {line[:60]}{'...' if len(line) > 60 else ''})[/]")

            user_history.append(line)
            stats.start_turn()
            await _stream_one_turn(agent, line, session_id, stats, current_model)
            tui.render_status_bar(
                console, current_model, await _ctx_pct(agent),
                stats.last_turn_seconds, stats.session_seconds, session_id,
            )


PERMISSION_ALIASES = {
    "/plan":  "plan",
    "/safe":  "default",
    "/auto":  "acceptEdits",
    "/yolo":  "bypassPermissions",
}


async def _stream_one_turn(
    agent: Agent,
    user_message: str,
    session_id: str,
    stats: SessionStats,
    current_model: str,
) -> None:
    """Run one user turn end-to-end: spinner → buffered text → markdown flush.

    Text deltas are buffered (not streamed inline) and rendered as a single
    Rich Markdown block when a tool call interrupts, or at TurnDone. This
    makes bullets / headers / code fences actually render with styling.
    """
    spinner = tui.ThinkingSpinner(console)
    spinner.start()
    spinner_running = True
    text_buffer: list[str] = []
    label_shown = False

    def flush_text() -> None:
        nonlocal label_shown
        if not text_buffer:
            return
        if not label_shown:
            tui.assistant_label(console)
            label_shown = True
        tui.render_assistant_markdown(console, "".join(text_buffer))
        text_buffer.clear()

    try:
        async for event in agent.run_stream(user_message, session_id=session_id):
            if isinstance(event, TextDelta):
                text_buffer.append(event.text)
            elif isinstance(event, (ThinkingDelta, ToolCall, ToolResult, TurnDone)):
                if spinner_running:
                    spinner.stop()
                    spinner_running = False
                flush_text()
                _render_event_tui(event, stats)
    finally:
        spinner.stop()
        flush_text()


async def _handle_slash(
    line: str,
    agent: Agent,
    stats: SessionStats,
    current_model: str,
    cfg: Config,
    user_history: list[str],
    session_id: str,
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
            rename_session(session_id, arg)
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

    # --- Permission modes --------------------------------------------------
    if cmd in PERMISSION_ALIASES:
        return await _set_perm(agent, PERMISSION_ALIASES[cmd])
    if cmd == "/permission":
        if not arg:
            console.print("  [yellow]usage:[/] /permission <default|acceptEdits|plan|bypassPermissions>")
            return None
        return await _set_perm(agent, arg)

    # --- Retry -------------------------------------------------------------
    if cmd == "/retry":
        if not user_history:
            console.print("[dim]nothing to retry — no prompts yet this session.[/]")
            return None
        return ("retry", user_history[-1])

    # --- Image / screenshot ------------------------------------------------
    if cmd == "/img":
        if not arg:
            console.print("  [yellow]usage:[/] /img <path> [question]")
            return None
        path_part, _, question = arg.partition(" ")
        path = Path(path_part).expanduser()
        if not path.is_file():
            console.print(f"[red]not a file: {path}[/]")
            return None
        question = question.strip() or "Describe what's in this image in detail."
        full_prompt = f"Read the image at `{path}` and answer: {question}"
        console.print(f"  [dim]attaching[/] [cyan]{path}[/]")
        return ("retry", full_prompt)

    if cmd == "/screenshot":
        latest = tui.find_recent_screenshot()
        if latest is None:
            console.print(
                "[red]No recent screenshot found.[/] [dim]Looked under "
                "/mnt/c/Users/*/Pictures/Screenshots and ~/Pictures.[/]"
            )
            return None
        question = arg.strip() or "Describe what's in this screenshot in detail."
        full_prompt = f"Read the image at `{latest}` and answer: {question}"
        console.print(f"  [dim]using[/] [cyan]{latest.name}[/] [dim]from {latest.parent}[/]")
        return ("retry", full_prompt)

    # --- Maps (open in browser + ask agent for details) -------------------
    if cmd == "/maps":
        if not arg:
            console.print("  [yellow]usage:[/] /maps <place or query>")
            return None
        import urllib.parse
        encoded = urllib.parse.quote_plus(arg)
        url = f"https://www.google.com/maps/search/?api=1&query={encoded}"
        opened = _open_in_browser(url)
        if opened:
            console.print(f"  [green]opened in browser:[/] [cyan]{url}[/]")
        else:
            console.print(f"  [dim]copy and open in browser:[/] [cyan]{url}[/]")
        full_prompt = (
            f"Look up '{arg}' on Google Maps and tell me the address, hours, rating, "
            f"and any notable details. Use web_search and web_fetch as needed. Cite sources.\n\n"
            f"Maps URL for reference: {url}"
        )
        return ("retry", full_prompt)

    # --- Plugin (shells to `claude plugin ...`) ----------------------------
    if cmd == "/plugin":
        if not arg:
            console.print(
                "  [yellow]usage:[/] /plugin <install|list|uninstall|update|enable|disable|marketplace> [args]\n"
                "  [dim]examples:[/]\n"
                "    [cyan]/plugin install github@claude-plugins-official[/]\n"
                "    [cyan]/plugin marketplace add anthropics/claude-plugins[/]\n"
                "    [cyan]/plugin list[/]"
            )
            return None
        _shell_claude_plugin(arg.split())
        return None

    # --- Workspace ---------------------------------------------------------
    if cmd == "/cwd":
        console.print(f"  [dim]workspace[/]   [cyan]{cfg.agent.workspace_dir}[/]")
        if cfg.agent.add_dirs:
            for extra in cfg.agent.add_dirs:
                console.print(f"  [dim]extra     [/]   [cyan]{extra}[/]")
        return None
    if cmd == "/cd":
        if not arg:
            console.print("  [yellow]usage:[/] /cd <path>")
            return None
        new_path = str(Path(arg).expanduser())
        if new_path not in cfg.agent.add_dirs:
            cfg.agent.add_dirs.append(new_path)
        console.print(
            f"  [green]added[/] [cyan]{new_path}[/] [dim]to allowed dirs.[/] "
            "[yellow]Use /new to start a fresh session and let Claude see it.[/]"
        )
        return None

    # --- Sysprompt ---------------------------------------------------------
    if cmd == "/sysprompt":
        console.print(f"\n[bold cyan]System prompt:[/]\n[dim]{cfg.agent.system_prompt}[/]\n")
        return None

    # --- Branch / fork -----------------------------------------------------
    if cmd in ("/branch", "/fork"):
        if not arg:
            console.print(f"  [yellow]usage:[/] {cmd} <new-name>")
            return None
        try:
            from claude_agent_sdk import fork_session
            result = fork_session(session_id, title=arg)
            new_id = getattr(result, "session_id", arg)
            console.print(
                f"  [green]forked[/] from [dim]{session_id[:16]}...[/] "
                f"to [cyan]{new_id}[/]"
            )
            return ("session", new_id)
        except Exception as e:  # noqa: BLE001
            console.print(f"[red]fork failed: {e}[/]")
            return None

    # --- Compress (best-effort) -------------------------------------------
    if cmd == "/compress":
        console.print(
            "  [dim]Manual compression is not exposed by the Agent SDK directly. "
            "Claude Code auto-compacts as you approach the context limit. "
            "For now, use /new to start a fresh session instead.[/]"
        )
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
        # No-arg form opens the picker dialog. Pass a model name to skip it.
        if not arg:
            chosen = await tui.pick_model_dialog(console, current_model)
            if not chosen:
                console.print("  [dim](no change)[/]")
                return None
            if chosen == current_model:
                console.print(f"  [dim]already on[/] [cyan]{chosen}[/]")
                return None
            arg = chosen
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


VALID_PERMISSION_MODES = {"default", "acceptEdits", "plan", "bypassPermissions", "dontAsk", "auto"}


async def _set_perm(agent: Agent, mode: str) -> str | None:
    """Switch the agent's permission mode mid-session. Returns 'permission' on success."""
    if mode not in VALID_PERMISSION_MODES:
        console.print(
            f"  [red]unknown mode '{mode}'.[/] "
            f"[dim]Valid: {', '.join(sorted(VALID_PERMISSION_MODES))}.[/]"
        )
        return None
    if agent._client is None:
        console.print("[red]Agent not connected.[/]")
        return None
    try:
        await agent._client.set_permission_mode(mode)  # type: ignore[arg-type]
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]Failed to switch permission mode: {e}[/]")
        return None
    desc = {
        "default":           "prompts before destructive actions",
        "plan":              "plan mode — Claude proposes a plan before executing",
        "acceptEdits":       "auto-approve file edits",
        "bypassPermissions": "full trust — no prompts (use carefully)",
        "dontAsk":           "don't ask permission",
        "auto":              "auto-decide",
    }.get(mode, mode)
    console.print(f"  [green]permission mode →[/] [bold cyan]{mode}[/] [dim]({desc})[/]")
    return "permission"


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


# ---------------------------------------------------------------------------
# `hermesv2 plugin ...` — thin wrapper around `claude plugin` subcommand
# ---------------------------------------------------------------------------


def _open_in_browser(url: str) -> bool:
    """Best-effort open a URL in the user's default browser.

    Tries WSL→Windows (cmd.exe), then Linux (xdg-open), then macOS (open).
    Returns True if something was launched, False otherwise.
    """
    candidates: list[list[str]] = []
    if shutil.which("cmd.exe"):
        candidates.append(["cmd.exe", "/c", "start", "", url])
    if shutil.which("xdg-open"):
        candidates.append(["xdg-open", url])
    if shutil.which("open"):
        candidates.append(["open", url])
    for argv in candidates:
        try:
            subprocess.Popen(
                argv,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            return True
        except Exception:  # noqa: BLE001
            continue
    return False


def _shell_claude_plugin(args: list[str]) -> None:
    """Invoke `claude plugin <args>` and surface stdout/stderr in our console."""
    if not shutil.which("claude"):
        console.print("[red]claude CLI not found. Install Claude Code first.[/]")
        return
    try:
        proc = subprocess.run(
            ["claude", "plugin", *args],
            capture_output=True, text=True, timeout=180,
        )
    except subprocess.TimeoutExpired:
        console.print("[red]claude plugin timed out.[/]")
        return
    if proc.stdout:
        console.print(proc.stdout.rstrip())
    if proc.returncode != 0:
        msg = (proc.stderr or "(no stderr)").rstrip()
        console.print(f"[red]exit {proc.returncode}:[/] {msg}")
        return
    console.print(
        "[dim](plugin changes take effect on /new in chat or on next `hermesv2` launch)[/]"
    )


@main.group(name="plugin")
def plugin_group() -> None:
    """Install and manage Claude Code plugins.

    Thin wrapper around `claude plugin`. Use plugin@marketplace for a specific
    marketplace, e.g. `hermesv2 plugin install github@claude-plugins-official`.
    """


@plugin_group.command(name="install")
@click.argument("plugin", required=True)
def plugin_install(plugin: str) -> None:
    """Install a plugin (use plugin@marketplace for specific marketplace)."""
    _shell_claude_plugin(["install", plugin])


@plugin_group.command(name="uninstall")
@click.argument("plugin")
def plugin_uninstall(plugin: str) -> None:
    """Uninstall a plugin."""
    _shell_claude_plugin(["uninstall", plugin])


@plugin_group.command(name="list")
def plugin_list() -> None:
    """List installed plugins."""
    _shell_claude_plugin(["list"])


@plugin_group.command(name="update")
@click.argument("plugin")
def plugin_update(plugin: str) -> None:
    """Update a plugin to the latest version."""
    _shell_claude_plugin(["update", plugin])


@plugin_group.command(name="enable")
@click.argument("plugin")
def plugin_enable(plugin: str) -> None:
    """Enable a disabled plugin."""
    _shell_claude_plugin(["enable", plugin])


@plugin_group.command(name="disable")
@click.argument("plugin")
def plugin_disable(plugin: str) -> None:
    """Disable an enabled plugin."""
    _shell_claude_plugin(["disable", plugin])


@plugin_group.group(name="marketplace")
def plugin_marketplace_group() -> None:
    """Manage plugin marketplaces."""


@plugin_marketplace_group.command(name="add")
@click.argument("source")
def plugin_marketplace_add(source: str) -> None:
    """Add a marketplace from URL, path, or GitHub repo (e.g. owner/repo)."""
    _shell_claude_plugin(["marketplace", "add", source])


@plugin_marketplace_group.command(name="list")
def plugin_marketplace_list() -> None:
    """List configured marketplaces."""
    _shell_claude_plugin(["marketplace", "list"])


@plugin_marketplace_group.command(name="remove")
@click.argument("name")
def plugin_marketplace_remove(name: str) -> None:
    """Remove a configured marketplace."""
    _shell_claude_plugin(["marketplace", "remove", name])


@plugin_marketplace_group.command(name="update")
@click.argument("name", required=False)
def plugin_marketplace_update(name: str | None) -> None:
    """Update marketplace(s) from their source (all if no name given)."""
    args = ["marketplace", "update"]
    if name:
        args.append(name)
    _shell_claude_plugin(args)


@main.group(name="skill")
def skill_group() -> None:
    """Browse, install, and manage Claude Code skills."""


@skill_group.command(name="list")
def skill_list() -> None:
    """List installed skills (in ~/.claude/skills/)."""
    from hermesv2 import skills as skillsmod
    entries = skillsmod.list_installed()
    if not entries:
        console.print("[dim]No skills installed.[/]")
        return
    for e in entries:
        line = f"  [cyan]{e.name:<24}[/]"
        if e.description:
            line += f" [dim]{e.description}[/]"
        if e.author:
            line += f" [dim](by {e.author})[/]"
        console.print(line)


@skill_group.command(name="browse")
def skill_browse() -> None:
    """Show all skills in the curated marketplace."""
    from hermesv2 import skills as skillsmod
    items = skillsmod.browse()
    if not items:
        console.print("[dim]Marketplace is empty.[/]")
        return
    for item in items:
        console.print(f"  [bold cyan]{item.get('name')}[/]  [dim]{item.get('description', '')}[/]")
        meta = []
        if item.get("author"):
            meta.append(f"by {item['author']}")
        if item.get("tags"):
            meta.append(", ".join(item["tags"]))
        if meta:
            console.print(f"    [dim]{' · '.join(meta)}[/]")


@skill_group.command(name="search")
@click.argument("query", nargs=-1, required=True)
def skill_search(query: tuple[str, ...]) -> None:
    """Filter the marketplace by name/description/tag."""
    from hermesv2 import skills as skillsmod
    q = " ".join(query)
    matches = skillsmod.search(q)
    if not matches:
        console.print(f"[dim]No marketplace matches for '{q}'.[/]")
        return
    for item in matches:
        console.print(f"  [cyan]{item.get('name')}[/]  [dim]{item.get('description', '')}[/]")


@skill_group.command(name="inspect")
@click.argument("name")
def skill_inspect(name: str) -> None:
    """Print a preview of an installed skill's SKILL.md."""
    from hermesv2 import skills as skillsmod
    text = skillsmod.inspect_skill(name)
    if text is None:
        console.print(f"[red]not installed:[/] {name}")
        sys.exit(1)
    entry = skillsmod.get_installed(name)
    if entry:
        console.print(f"[bold cyan]{entry.name}[/]  [dim]{entry.description}[/]")
        if entry.source:
            console.print(f"[dim]source: {entry.source}[/]")
        console.print(f"[dim]path: {entry.installed_at}[/]")
        console.print()
    console.print(text)


@skill_group.command(name="install")
@click.argument("identifier")
@click.option("--name", default=None, help="Override the install directory name.")
def skill_install(identifier: str, name: str | None) -> None:
    """Install a skill by marketplace name, owner/repo, or git URL."""
    from hermesv2 import skills as skillsmod
    try:
        entry = skillsmod.install(identifier, name=name)
    except skillsmod.SkillError as e:
        console.print(f"[red]install failed:[/] {e}")
        sys.exit(1)
    console.print(f"[green]installed[/] [cyan]{entry.name}[/] → [dim]{entry.installed_at}[/]")
    if entry.description:
        console.print(f"[dim]{entry.description}[/]")


@skill_group.command(name="uninstall")
@click.argument("name")
@click.option("--yes", is_flag=True, help="Skip the confirmation prompt.")
def skill_uninstall(name: str, yes: bool) -> None:
    """Remove an installed skill."""
    from hermesv2 import skills as skillsmod
    entry = skillsmod.get_installed(name)
    if entry is None:
        console.print(f"[red]not installed:[/] {name}")
        sys.exit(1)
    if not yes:
        console.print(f"  [yellow]about to delete:[/] [cyan]{entry.installed_at}[/]")
        if not click.confirm("  proceed?", default=False):
            console.print("[dim]cancelled.[/]")
            return
    try:
        skillsmod.uninstall(name)
    except skillsmod.SkillError as e:
        console.print(f"[red]uninstall failed:[/] {e}")
        sys.exit(1)
    console.print(f"[green]uninstalled[/] {name}")


@skill_group.command(name="snapshot")
@click.argument("direction", type=click.Choice(["export", "import"]))
@click.argument("path", type=click.Path())
def skill_snapshot(direction: str, path: str) -> None:
    """Export current skills to JSON, or import a snapshot back."""
    import json

    from hermesv2 import skills as skillsmod
    p = Path(path)
    if direction == "export":
        p.write_text(json.dumps(skillsmod.snapshot_export(), indent=2))
        console.print(f"[green]wrote snapshot →[/] [cyan]{p}[/]")
    else:
        try:
            data = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError) as e:
            console.print(f"[red]bad snapshot file:[/] {e}")
            sys.exit(1)
        installed = skillsmod.snapshot_import(data)
        if installed:
            console.print(f"[green]installed {len(installed)} new skills:[/] {', '.join(installed)}")
        else:
            console.print("[dim]nothing new to install (all already present).[/]")


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
