"""Josh v1 CLI: `joshv1 [chat] | run | doctor | slack | discord | update`."""

from __future__ import annotations

import asyncio
import json
import os
import platform
import secrets
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import click
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion, FuzzyCompleter
from prompt_toolkit.formatted_text import HTML, FormattedText
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.text import Text

from joshv1 import themes, tui
from joshv1.agent import (
    Agent,
    TextDelta,
    ThinkingDelta,
    ToolCall,
    ToolResult,
    TurnDone,
)
from joshv1.config import (
    VALID_EFFORTS,
    AgentSettings,
    Config,
    clear_ensemble_members_override,
    load_config,
    save_active_effort,
    save_active_ensemble_mode,
    save_ensemble_members_override,
)
from joshv1.providers import PROVIDER_PRESETS, OpenAICompatProvider


def build_backend(settings: AgentSettings, cwd: str | Path | None = None,
                  resume_session_id: str | None = None):
    """Pick the right backend based on settings.provider.

    Returns either an `Agent` (Claude, full tool-use via Agent SDK) or an
    `OpenAICompatProvider` (OpenRouter / Ollama / Gemini, text-only in
    phase 1). Both expose `connect / disconnect / run_stream / reset_session`.
    """
    if settings.provider == "claude":
        return Agent(settings, cwd=cwd, resume_session_id=resume_session_id)
    if settings.provider in PROVIDER_PRESETS:
        return OpenAICompatProvider(settings, provider_name=settings.provider, cwd=cwd)
    raise ValueError(
        f"Unknown provider: {settings.provider!r}. "
        f"Valid: claude, {', '.join(PROVIDER_PRESETS)}"
    )

# Active theme is loaded from ~/.josh-memory/theme (set via `/theme <name>`).
# The Console is built with a Rich Theme that maps every `josh.*` and
# `markdown.*` style name to the palette's hex codes — so swapping the
# active palette repaints the entire UI without touching tui.py or cli.py.
ACTIVE_PALETTE = themes.load_active_palette()
console = Console(theme=themes.build_theme(ACTIVE_PALETTE))

# Last /maps result, used by /zoomin and /zoomout for incremental re-render.
# Mutable dict (cleared/updated in place) so we don't need `global` declarations.
# Keys when populated: query, place, lat, lon, zoom.
_LAST_MAP: dict[str, Any] = {}


def _render_map_panel(result: dict[str, Any]) -> None:
    """Print the bordered braille map (or fallback) for one /maps result."""
    if result.get("place"):
        title = Text()
        title.append(" ", style="dim")
        title.append(result["place"][:80], style="josh.title")
        if result.get("lat") is not None and result.get("lon") is not None:
            title.append(
                f"  · {result['lat']:.4f}, {result['lon']:.4f}",
                style="dim",
            )
        if result.get("zoom") is not None:
            title.append(f"  · z{result['zoom']}", style="josh.warm")
    else:
        title = Text(" map ", style="josh.title")

    if result.get("braille"):
        from rich.panel import Panel as RPanel
        panel = RPanel(
            Text(result["braille"], style="josh.title"),
            title=title,
            border_style="josh.border",
            padding=(0, 1),
        )
        console.print(panel)
    else:
        if result.get("place"):
            console.print(f"  [josh.info]{result['place']}[/]")
        if result.get("lat") is not None:
            console.print(
                f"  [dim]coords:[/] [josh.info]{result['lat']:.4f}, {result['lon']:.4f}[/]"
            )
        if result.get("error"):
            console.print(f"  [josh.highlight]map render:[/] [dim]{result['error']}[/]")


# ---------------------------------------------------------------------------
# Slash command autocomplete + theming (prompt-toolkit)
# ---------------------------------------------------------------------------


# prompt-toolkit needs real hex codes (it doesn't understand Rich's
# `josh.*` style names), so this is generated from the active palette
# at import time. To pick up a theme change, restart joshv1.
DROPDOWN_STYLE = Style.from_dict(themes.build_dropdown_style_dict(ACTIVE_PALETTE))


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
    """Compact, pipe-friendly rendering for `joshv1 run`."""
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
        console.print(f"\n[josh.info]→ {event.name}[/] [dim]{preview}[/]")
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
    """Fancy rendering for `joshv1 chat`. Updates running stats too."""
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


# ---------------------------------------------------------------------------
# REPL state — passed through the chat loop and slash dispatcher together so
# new commands (queue/steer/background/snapshot/etc.) can mutate it without
# expanding `_handle_slash`'s parameter list every time.
# ---------------------------------------------------------------------------

PERSONALITIES: dict[str, str] = {
    "default": "",  # use cfg.agent.system_prompt verbatim
    "concise": (
        "Be aggressively terse. One sentence answers when possible. No "
        "preamble, no caveats, no closing pleasantries."
    ),
    "playful": (
        "Be friendly, warm, and a little playful. Use natural conversational "
        "language. Occasional light humor is welcome, but never at the user's "
        "expense and never sarcastic."
    ),
    "rubberduck": (
        "Be a rubber-duck debugging companion. Ask clarifying questions about "
        "the user's reasoning. Reflect back what they said in your own words. "
        "Help them think out loud — don't jump to solutions."
    ),
    "mentor": (
        "Be a patient senior engineer mentoring a junior. Explain reasoning, "
        "show what you'd consider, name relevant trade-offs. Teach as you go."
    ),
    "savage-pair": (
        "Be a direct, opinionated code reviewer. Call out bad ideas plainly. "
        "Never sugar-coat. Still respectful — critique the code, not the person."
    ),
}


VERBOSE_LEVELS: tuple[str, ...] = ("off", "new", "all", "verbose")
BUSY_MODES: tuple[str, ...] = ("queue", "steer", "interrupt")


@dataclass
class BackgroundTask:
    """One backgrounded prompt. Result lands in `result` when done."""
    id: str
    prompt: str
    started_at: float
    task: Any  # asyncio.Task — keep loose to avoid import gymnastics here
    result: str | None = None
    error: str | None = None
    done: bool = False


@dataclass
class ReplState:
    """Mutable per-session state for the chat REPL. One instance per /chat run.

    All new commands (Hermes-style /queue, /steer, /background, /snapshot,
    /undo, /personality, /statusbar, /verbose, /busy …) read or mutate this.
    """
    # Conversation history
    user_history: list[str] = field(default_factory=list)
    assistant_history: list[str] = field(default_factory=list)

    # Queued / steered prompts (drained between turns or after next tool call)
    queued_prompts: list[str] = field(default_factory=list)
    steer_prompts: list[str] = field(default_factory=list)

    # Background tasks
    background_tasks: dict[str, BackgroundTask] = field(default_factory=dict)
    next_bg_id: int = 1

    # TUI state toggles
    statusbar_enabled: bool = True
    verbose_level: str = "new"  # off | new | all | verbose
    busy_mode: str = "queue"     # what Enter does while the agent is working

    # Personality (overrides cfg.agent.system_prompt when set)
    personality: str = "default"

    # Active session label
    session_id: str = ""


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
    """Josh v1 — personal agent on top of Claude Code (uses your Max subscription)."""
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
        console.print("[josh.error]No prompt given. Pass one as args or pipe via stdin.[/]")
        sys.exit(2)

    cfg = load_config(ctx.obj.get("config_path"))
    asyncio.run(_run_one(cfg, message))


async def _run_one(cfg: Config, message: str) -> None:
    async with build_backend(cfg.agent, cwd=cfg.agent.workspace_dir) as agent:
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
    state = ReplState(session_id=session_id)
    user_history = state.user_history  # backward-compat alias inside this fn

    prompt_session: PromptSession[str] = PromptSession(
        completer=FuzzyCompleter(SlashCommandCompleter()),
        complete_while_typing=True,
        complete_in_thread=True,
        style=DROPDOWN_STYLE,
    )

    async with build_backend(
        cfg.agent,
        cwd=cfg.agent.workspace_dir,
        resume_session_id=session_name,
    ) as agent:
        current_model = cfg.agent.model

        tui.render_status_bar(console, current_model, None, 0.0, 0.0, session_id)

        # Populated by _stream_one_turn whenever Claude calls AskUserQuestion;
        # consumed at the top of the loop on the next iteration to show an
        # interactive picker instead of plain text input.
        pending_questions: list[dict] = []

        while True:
            try:
                if pending_questions:
                    # The agent asked one or more multiple-choice questions on
                    # the last turn. Show the picker(s); use the answers as the
                    # next user message. Esc on any picker cancels and drops
                    # back to the normal prompt.
                    answers: dict[str, str | list[str]] = {}
                    cancelled = False
                    for q in pending_questions:
                        ans = await tui.pick_answer_for_question(
                            console,
                            question=q.get("question", ""),
                            options=q.get("options", []),
                            header=q.get("header", ""),
                            multi_select=q.get("multiSelect", False),
                        )
                        if ans is None:
                            cancelled = True
                            break
                        answers[q.get("header") or q.get("question", "answer")] = ans
                    if cancelled:
                        console.print("  [dim](picker cancelled — type your answer at the prompt)[/]")
                        pending_questions = []
                        continue
                    line = tui.format_picker_answers(pending_questions, answers)
                    pending_questions = []
                    console.print(f"  [josh.label.you]▎ you ❱[/] [dim]{line}[/]")
                else:
                    # The blank line is printed separately — keeping the prompt
                    # itself a pure single line avoids prompt-toolkit's cursor
                    # math going wrong on redraw (which made the prompt vanish
                    # when backspacing through `/x` completion text).
                    console.print()
                    line = await prompt_session.prompt_async(
                        HTML(themes.prompt_html(ACTIVE_PALETTE))
                    )
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]bye.[/]")
                return
            line = line.strip()
            if not line:
                continue

            if line.startswith("/"):
                action = await _handle_slash(
                    line, agent, stats, current_model, cfg, state,
                )
                if action == "exit":
                    return
                if isinstance(action, tuple):
                    if action[0] == "model":
                        current_model = action[1]
                    elif action[0] == "session":
                        session_id = action[1]
                        state.session_id = session_id
                        user_history.clear()
                        state.assistant_history.clear()
                    elif action[0] == "retry":
                        line = action[1]
                    else:
                        pass
                if action == "reset":
                    session_id = agent.reset_session()
                    state.session_id = session_id
                if action == "redraw":
                    tui.render_startup(console, cfg.agent, session_name=session_id)
                # If retry, fall through to the streaming block; otherwise loop.
                if not (isinstance(action, tuple) and action[0] == "retry"):
                    if state.statusbar_enabled:
                        tui.render_status_bar(
                            console, current_model, await _ctx_pct(agent),
                            stats.last_turn_seconds, stats.session_seconds, session_id,
                        )
                    continue
                console.print(f"  [dim](retrying: {line[:60]}{'...' if len(line) > 60 else ''})[/]")

            user_history.append(line)
            stats.start_turn()
            # Drain any /queue'd prompts BEFORE this one so the queued items
            # are sent first (FIFO from the user's POV when they queued ahead).
            if state.queued_prompts:
                queued = state.queued_prompts.pop(0)
                console.print(f"  [dim](running queued prompt: {queued[:60]}{'...' if len(queued) > 60 else ''})[/]")
                state.queued_prompts.insert(0, line)  # current line goes back to queue head
                line = queued
            if cfg.agent.ensemble_mode and isinstance(agent, Agent):
                # Auto-route: gather drafts in parallel, then feed them to the
                # main Claude session as augmented context. Tools, scrollback,
                # and session continuity are all preserved — only the user's
                # input message is augmented before being sent on.
                await _stream_ensemble_turn(
                    agent, line, session_id, stats, current_model, cfg,
                    pending_questions=pending_questions,
                )
            else:
                await _stream_one_turn(
                    agent, line, session_id, stats, current_model,
                    pending_questions=pending_questions,
                    assistant_history=state.assistant_history,
                )
            if state.statusbar_enabled:
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
    pending_questions: list[dict] | None = None,
    assistant_history: list[str] | None = None,
) -> None:
    """Run one user turn end-to-end: spinner → buffered text → markdown flush.

    Text deltas are buffered (not streamed inline) and rendered as a single
    Rich Markdown block when a tool call interrupts, or at TurnDone. This
    makes bullets / headers / code fences actually render with styling.

    If `assistant_history` is provided, every flushed assistant segment is
    appended so /copy /undo /save can find the last reply.
    """
    spinner = tui.ThinkingSpinner(console)
    spinner.start()
    spinner_running = True
    text_buffer: list[str] = []
    full_assistant_text: list[str] = []
    label_shown = False

    def flush_text() -> None:
        nonlocal label_shown
        if not text_buffer:
            return
        if not label_shown:
            tui.assistant_label(console)
            label_shown = True
        joined = "".join(text_buffer)
        full_assistant_text.append(joined)
        tui.render_assistant_markdown(console, joined)
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
                # Capture AskUserQuestion so we can show an interactive picker
                # at the next prompt instead of the user having to type the answer.
                if (
                    pending_questions is not None
                    and isinstance(event, ToolCall)
                    and event.name == "AskUserQuestion"
                    and isinstance(event.input, dict)
                ):
                    for q in (event.input.get("questions") or []):
                        if isinstance(q, dict) and q.get("options"):
                            pending_questions.append(q)
    finally:
        spinner.stop()
        flush_text()
        if assistant_history is not None and full_assistant_text:
            assistant_history.append("".join(full_assistant_text))


async def _stream_ensemble_turn(
    agent: Agent,
    user_message: str,
    session_id: str,
    stats: SessionStats,
    current_model: str,
    cfg: Config,
    pending_questions: list[dict] | None = None,
) -> None:
    """Ensemble-augmented turn: fan out for drafts, then run main session w/ them.

    Unlike `_run_ensemble_once` (which synthesizes a final answer with NO tools
    and NO session memory), this routes through the user's primary Claude agent
    so tool use, scrollback, and persistent context all keep working. The
    ensemble drafts are injected as extra context inside the user message.
    """
    from joshv1 import ensemble as ens_mod

    members = ens_mod.members_from_settings(cfg.agent)
    if not members:
        # No ensemble configured; fall back to a plain turn.
        await _stream_one_turn(
            agent, user_message, session_id, stats, current_model,
            pending_questions=pending_questions,
        )
        return

    console.print(f"  [dim]ensemble: polling {len(members)} drafters in parallel...[/]")

    def _on_draft(member, draft):
        if draft.success:
            console.print(
                f"  [josh.success]✓[/] [josh.info]{member.model}[/] "
                f"[dim]({len(draft.text)} chars)[/]"
            )
        else:
            console.print(f"  [josh.error]✗[/] [josh.info]{member.model}[/]")
            console.print(f"      [dim]error: {draft.error!r}[/]")

    drafts = await ens_mod.collect_drafts(
        user_message, cfg.agent, members=members, on_draft_complete=_on_draft,
    )
    successful = [d for d in drafts if d.success]

    if not successful:
        console.print("  [dim]all drafts failed; running plain turn.[/]")
        await _stream_one_turn(
            agent, user_message, session_id, stats, current_model,
            pending_questions=pending_questions,
        )
        return

    console.print(
        f"  [dim]feeding {len(successful)}/{len(drafts)} drafts to main session...[/]"
    )
    augmented = ens_mod.format_drafts_as_context(user_message, successful)
    await _stream_one_turn(
        agent, augmented, session_id, stats, current_model,
        pending_questions=pending_questions,
    )


async def _handle_slash(
    line: str,
    agent: Agent,
    stats: SessionStats,
    current_model: str,
    cfg: Config,
    state: ReplState,
) -> str | tuple[str, str] | None:
    # Rebind for readability; both names point at the same list/string.
    user_history = state.user_history
    session_id = state.session_id

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
            console.print("  [josh.highlight]usage:[/] /title <name>")
            return None
        try:
            from claude_agent_sdk import rename_session
            rename_session(session_id, arg)
            console.print(f"  [josh.success]session renamed to[/] [josh.info]{arg}[/]")
        except Exception as e:  # noqa: BLE001
            console.print(f"[josh.error]rename failed: {e}[/]")
        return None
    if cmd == "/history":
        if not user_history:
            console.print("[dim]no user prompts yet in this session.[/]")
            return None
        console.print("\n[josh.title]Your prompts this session[/]")
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
            console.print("  [josh.highlight]usage:[/] /permission <default|acceptEdits|plan|bypassPermissions>")
            return None
        return await _set_perm(agent, arg)

    # --- Retry -------------------------------------------------------------
    if cmd == "/retry":
        if not user_history:
            console.print("[dim]nothing to retry — no prompts yet this session.[/]")
            return None
        return ("retry", user_history[-1])

    # --- Self-audit --------------------------------------------------------
    if cmd == "/audit":
        from joshv1 import audit as audit_mod
        repo = audit_mod.find_hermes_repo()
        if repo is None:
            console.print("[josh.error]Couldn't find the joshv1 repo on disk.[/]")
            return None
        console.print(f"  [dim]auditing[/] [josh.info]{repo}[/]")
        return ("retry", audit_mod.prompt_for_repo(repo))

    # --- Ensemble: fan out to multiple models, synthesize one answer -------
    if cmd == "/ensemble":
        return await _handle_ensemble(arg, cfg)

    # --- Image / screenshot ------------------------------------------------
    if cmd == "/img":
        if not arg:
            console.print("  [josh.highlight]usage:[/] /img <path> [question]")
            return None
        path_part, _, question = arg.partition(" ")
        path = Path(path_part).expanduser()
        if not path.is_file():
            console.print(f"[josh.error]not a file: {path}[/]")
            return None
        question = question.strip() or "Describe what's in this image in detail."
        full_prompt = f"Read the image at `{path}` and answer: {question}"
        console.print(f"  [dim]attaching[/] [josh.info]{path}[/]")
        return ("retry", full_prompt)

    if cmd == "/screenshot":
        latest = tui.find_recent_screenshot()
        if latest is None:
            console.print(
                "[josh.error]No recent screenshot found.[/] [dim]Looked under "
                "/mnt/c/Users/*/Pictures/Screenshots and ~/Pictures.[/]"
            )
            return None
        question = arg.strip() or "Describe what's in this screenshot in detail."
        full_prompt = f"Read the image at `{latest}` and answer: {question}"
        console.print(f"  [dim]using[/] [josh.info]{latest.name}[/] [dim]from {latest.parent}[/]")
        return ("retry", full_prompt)

    # --- Maps (geocode + stitched braille map + browser open + agent ask) ---
    if cmd == "/maps":
        if not arg:
            console.print("  [josh.highlight]usage:[/] /maps <place or query> [z<N>]")
            return None
        import urllib.parse

        from joshv1 import maps as mapsmod

        # Allow "z<N>" anywhere in the query to override zoom (e.g. /maps Bayside z17)
        zoom = 15
        tokens = []
        for tok in arg.split():
            if tok.lower().startswith("z") and tok[1:].isdigit():
                zoom = max(1, min(19, int(tok[1:])))
            else:
                tokens.append(tok)
        query = " ".join(tokens).strip() or arg

        encoded = urllib.parse.quote_plus(query)
        url = f"https://www.google.com/maps/search/?api=1&query={encoded}"

        console.print(f"  [dim]searching maps for[/] [josh.info]{query}[/] [dim](z{zoom})[/]...")
        term_w = max(40, min(console.size.width - 6, 90))
        result = mapsmod.render_map(query, zoom=zoom, cols=term_w, rows=18, tiles=2)

        _render_map_panel(result)

        if result.get("lat") is not None:
            _LAST_MAP.clear()
            _LAST_MAP.update({
                "query": query,
                "place": result["place"],
                "lat": result["lat"],
                "lon": result["lon"],
                "zoom": result["zoom"],
            })

        opened = _open_in_browser(url)
        if opened:
            console.print(f"  [josh.success]opened in browser:[/] [josh.info]{url}[/]")
        else:
            console.print(f"  [dim]copy and open in browser:[/] [josh.info]{url}[/]")
        console.print(
            "  [dim]/zoomin · /zoomout to re-render at a different zoom level[/]"
        )

        full_prompt = (
            f"Look up '{query}' on Google Maps and tell me the address, hours, rating, "
            f"and any notable details. Use web_search and web_fetch as needed. "
            f"Cite sources.\n\nMaps URL for reference: {url}"
        )
        return ("retry", full_prompt)

    # --- Zoom in / out (re-render the last /maps result) ------------------
    if cmd in ("/zoomin", "/zoomout"):
        if not _LAST_MAP:
            console.print(
                "  [josh.error]no map yet[/] [dim]— run /maps <query> first[/]"
            )
            return None
        delta = 1 if cmd == "/zoomin" else -1
        new_zoom = max(1, min(19, _LAST_MAP["zoom"] + delta))
        if new_zoom == _LAST_MAP["zoom"]:
            console.print(f"  [dim]already at zoom limit (z{new_zoom})[/]")
            return None
        from joshv1 import maps as mapsmod
        term_w = max(40, min(console.size.width - 6, 90))
        console.print(
            f"  [dim]re-rendering[/] [josh.info]{_LAST_MAP['place'][:60]}[/] "
            f"[dim]at z{new_zoom}...[/]"
        )
        result = mapsmod.render_map_at(
            _LAST_MAP["lat"], _LAST_MAP["lon"], _LAST_MAP["place"],
            zoom=new_zoom, cols=term_w, rows=18, tiles=2,
        )
        _render_map_panel(result)
        _LAST_MAP["zoom"] = new_zoom
        return None

    # --- Theme switcher ---------------------------------------------------
    if cmd == "/theme":
        if not arg or arg == "list":
            _print_theme_list()
            return None
        if arg not in themes.PALETTES:
            console.print(f"  [josh.error]unknown theme:[/] [josh.info]{arg}[/]")
            console.print(f"  [dim]available:[/] {', '.join(themes.PALETTES)}")
            return None
        try:
            themes.save_active_theme(arg)
        except Exception as e:  # noqa: BLE001
            console.print(f"  [josh.error]failed to save theme:[/] {e}")
            return None
        # Hot-swap Rich's theme so the welcome panel + agent replies
        # pick up the new colors immediately.
        new_palette = themes.get_palette(arg)
        console.push_theme(themes.build_theme(new_palette))
        console.print(
            f"  [josh.success]theme switched to[/] [josh.session]{arg}[/]  "
            f"{themes.render_theme_swatch(new_palette)}\n"
            f"  [dim]Restart joshv1 to also update the prompt + dropdown.[/]"
        )
        return None

    # --- Plugin (shells to `claude plugin ...`) ----------------------------
    if cmd == "/plugin":
        if not arg:
            console.print(
                "  [josh.highlight]usage:[/] /plugin <install|list|uninstall|update|enable|disable|marketplace> [args]\n"
                "  [dim]examples:[/]\n"
                "    [josh.info]/plugin install github@claude-plugins-official[/]\n"
                "    [josh.info]/plugin marketplace add anthropics/claude-plugins[/]\n"
                "    [josh.info]/plugin list[/]"
            )
            return None
        _shell_claude_plugin(arg.split())
        return None

    # --- Workspace ---------------------------------------------------------
    if cmd == "/cwd":
        console.print(f"  [dim]workspace[/]   [josh.info]{cfg.agent.workspace_dir}[/]")
        if cfg.agent.add_dirs:
            for extra in cfg.agent.add_dirs:
                console.print(f"  [dim]extra     [/]   [josh.info]{extra}[/]")
        return None
    if cmd == "/cd":
        if not arg:
            console.print("  [josh.highlight]usage:[/] /cd <path>")
            return None
        new_path = str(Path(arg).expanduser())
        if new_path not in cfg.agent.add_dirs:
            cfg.agent.add_dirs.append(new_path)
        console.print(
            f"  [josh.success]added[/] [josh.info]{new_path}[/] [dim]to allowed dirs.[/] "
            "[josh.highlight]Use /new to start a fresh session and let Claude see it.[/]"
        )
        return None

    # --- Sysprompt ---------------------------------------------------------
    if cmd == "/sysprompt":
        console.print(f"\n[josh.title]System prompt:[/]\n[dim]{cfg.agent.system_prompt}[/]\n")
        return None

    # --- Branch / fork -----------------------------------------------------
    if cmd in ("/branch", "/fork"):
        if not arg:
            console.print(f"  [josh.highlight]usage:[/] {cmd} <new-name>")
            return None
        try:
            from claude_agent_sdk import fork_session
            result = fork_session(session_id, title=arg)
            new_id = getattr(result, "session_id", arg)
            console.print(
                f"  [josh.success]forked[/] from [dim]{session_id[:16]}...[/] "
                f"to [josh.info]{new_id}[/]"
            )
            return ("session", new_id)
        except Exception as e:  # noqa: BLE001
            console.print(f"[josh.error]fork failed: {e}[/]")
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
                console.print(f"  [dim]already on[/] [josh.info]{chosen}[/]")
                return None
            arg = chosen
        if isinstance(agent, Agent):
            if agent._client is None:
                console.print("[josh.error]Agent not connected.[/]")
                return None
            try:
                await agent._client.set_model(arg)
                console.print(f"  [josh.success]switched model to[/] [josh.info]{arg}[/]")
                return ("model", arg)
            except Exception as e:  # noqa: BLE001
                console.print(f"[josh.error]Failed to switch model: {e}[/]")
                return None
        # Non-Claude backend: update settings and reconnect.
        agent.settings.model = arg
        try:
            await agent.reconfigure()
        except Exception as e:  # noqa: BLE001
            console.print(f"[josh.error]Failed to switch model: {e}[/]")
            return None
        console.print(f"  [josh.success]switched model to[/] [josh.info]{arg}[/]")
        return ("model", arg)
    if cmd == "/effort":
        # No-arg form opens the picker dialog. Pass an effort name to skip it.
        if not arg:
            chosen = await tui.pick_effort_dialog(console, cfg.agent.effort)
            if not chosen:
                console.print("  [dim](no change)[/]")
                return None
            if chosen == cfg.agent.effort:
                console.print(f"  [dim]already on[/] [josh.info]{chosen}[/]")
                return None
            arg = chosen
        if arg not in VALID_EFFORTS:
            console.print(f"  [josh.error]unknown effort:[/] [josh.info]{arg}[/]")
            console.print(f"  [dim]valid:[/] {', '.join(VALID_EFFORTS)}")
            return None
        if not isinstance(agent, Agent):
            console.print("  [dim]effort is a Claude-only setting; non-Claude providers don't use it.[/]")
            return None
        if agent._client is None:
            console.print("[josh.error]Agent not connected.[/]")
            return None
        prev = cfg.agent.effort
        cfg.agent.effort = arg
        try:
            save_active_effort(arg, cfg.agent.memory_dir)
        except Exception as e:  # noqa: BLE001
            console.print(f"  [josh.error]failed to save effort:[/] {e}")
            cfg.agent.effort = prev
            return None
        console.print(f"  [dim]reconfiguring agent ({prev} → {arg})...[/]")
        console.print("  [dim](agent restart — model loses prior-turn context; your scrollback stays)[/]")
        try:
            await agent.reconfigure()
        except Exception as e:  # noqa: BLE001
            console.print(f"  [josh.error]reconnect failed:[/] {e}")
            return None
        console.print(f"  [josh.success]switched effort to[/] [josh.info]{arg}[/]")
        return None
    if cmd == "/update":
        await _run_update_async()
        return None
    if cmd == "/sessions":
        await _list_sessions_async()
        return None

    # =======================================================================
    # SESSION GROUP — /save /undo /branch /fork /compress /rollback /snapshot
    # /stop /background /agents /queue /steer /status /resume
    # =======================================================================
    if cmd == "/save":
        return _cmd_save(arg, state, session_id)
    if cmd == "/undo":
        return _cmd_undo(state)
    if cmd in ("/branch", "/fork"):
        return await _cmd_branch(arg, agent, state, session_id)
    if cmd == "/compress":
        return await _cmd_compress(arg, agent, state)
    if cmd == "/rollback":
        return _cmd_rollback(arg)
    if cmd in ("/snapshot", "/snap"):
        return _cmd_snapshot(arg, cfg, state, current_model)
    if cmd == "/stop":
        return _cmd_stop(state)
    if cmd in ("/background", "/bg", "/btw"):
        return await _cmd_background(arg, agent, state, session_id, stats, current_model)
    if cmd in ("/agents", "/tasks"):
        return _cmd_agents(state)
    if cmd in ("/queue", "/q"):
        return _cmd_queue(arg, state)
    if cmd == "/steer":
        return _cmd_steer(arg, state)
    if cmd == "/status":
        return await _cmd_status(agent, state, current_model, stats, session_id)
    if cmd == "/resume":
        return _cmd_resume(arg)

    # =======================================================================
    # INFO GROUP — /profile /gquota /usage /insights /platforms /copy /paste /debug
    # =======================================================================
    if cmd == "/profile":
        return _cmd_profile(cfg)
    if cmd == "/gquota":
        return await _cmd_gquota()
    if cmd == "/usage":
        return _cmd_usage(stats, current_model)
    if cmd == "/insights":
        return _cmd_insights(arg, cfg)
    if cmd in ("/platforms", "/gateway"):
        return _cmd_platforms(cfg)
    if cmd == "/copy":
        return _cmd_copy(arg, state)
    if cmd == "/paste":
        return _cmd_paste()
    if cmd == "/debug":
        return _cmd_debug(cfg, state, stats, current_model)

    # =======================================================================
    # CONFIGURATION GROUP — /config /provider /personality /statusbar /verbose
    # /reasoning /skin /voice /busy
    # =======================================================================
    if cmd == "/config":
        return _cmd_config(cfg, state)
    if cmd == "/provider":
        # Alias for /model — re-dispatch with the same args
        return await _handle_slash(
            "/model " + arg if arg else "/model", agent, stats, current_model, cfg, state,
        )
    if cmd == "/personality":
        return await _cmd_personality(arg, cfg, agent, state)
    if cmd in ("/statusbar", "/sb"):
        return _cmd_statusbar(arg, state)
    if cmd == "/verbose":
        return _cmd_verbose(state)
    if cmd == "/reasoning":
        return await _cmd_reasoning(arg, agent, cfg)
    if cmd == "/skin":
        return _cmd_skin(arg)
    if cmd == "/voice":
        return _cmd_voice(arg)
    if cmd == "/busy":
        return _cmd_busy(arg, state)

    # =======================================================================
    # TOOLS & SKILLS GROUP — /toolsets /skills /cron /reload /reload-mcp
    # /browser /plugins
    # =======================================================================
    if cmd == "/toolsets":
        return _cmd_toolsets()
    if cmd == "/skills":
        return _cmd_skills(arg)
    if cmd == "/cron":
        return _cmd_cron(arg)
    if cmd == "/reload":
        return _cmd_reload()
    if cmd in ("/reload-mcp", "/reload_mcp"):
        return await _cmd_reload_mcp(agent)
    if cmd == "/browser":
        return _cmd_browser(arg)
    if cmd == "/plugins":
        return _cmd_plugins()

    console.print(f"[josh.error]Unknown command: {cmd}. Type /help for a list.[/]")
    return None


async def _list_sessions_async() -> None:
    from claude_agent_sdk import list_sessions
    try:
        sessions = await asyncio.to_thread(list_sessions, limit=20)
    except Exception as e:  # noqa: BLE001
        console.print(f"[josh.error]Failed to list sessions: {e}[/]")
        return
    if not sessions:
        console.print("[dim]No saved sessions yet.[/]")
        return
    tui.render_sessions(console, sessions)


# ===========================================================================
# Slash command implementations — every Hermes-style command we wired into the
# dispatcher above lives here. Order matches the groups in _handle_slash.
# Helpers (`_clip_copy`, `_clip_paste`, `_snapshot_dir`, etc.) sit at the end
# of this section so the handlers stay close to the dispatcher.
# ===========================================================================


# ----- Session group -------------------------------------------------------

def _cmd_save(arg: str, state: ReplState, session_id: str) -> None:
    """Write the conversation to a markdown file. Default path under memory_dir."""
    from joshv1.memory import memory_dir as _mem_dir
    if arg:
        out_path = Path(arg).expanduser()
    else:
        saves = _mem_dir("~/.josh-memory") / "saves"
        saves.mkdir(parents=True, exist_ok=True)
        out_path = saves / f"{session_id}.md"
    pairs = list(zip(state.user_history, state.assistant_history + [""] * 9))
    body = [f"# Session {session_id}\n\nSaved {datetime.now().isoformat()}\n"]
    for i, (u, a) in enumerate(pairs, 1):
        body.append(f"\n## Turn {i}\n\n### You\n\n{u}\n\n### Josh\n\n{a or '_(no reply captured)_'}")
    out_path.write_text("\n".join(body))
    console.print(f"  [josh.success]saved →[/] [josh.info]{out_path}[/] [dim]({len(pairs)} turn{'s' if len(pairs)!=1 else ''})[/]")
    return None


def _cmd_undo(state: ReplState) -> None:
    """Pop the last exchange. Doesn't actually rewind the agent's server-side
    session — Claude Code doesn't expose that — but trims local history so
    /retry, /save, /copy, etc. see a cleaner state."""
    if not state.user_history:
        console.print("[dim]nothing to undo — history is empty.[/]")
        return None
    u = state.user_history.pop()
    a = state.assistant_history.pop() if state.assistant_history else "(no reply captured)"
    console.print(f"  [josh.success]undid[/] [dim]({len(u)}-char prompt + {len(a)}-char reply)[/]")
    console.print(
        "  [dim]note: only LOCAL history changed. The agent's server-side "
        "session still remembers that exchange. Use /reset to fully reset.[/]"
    )
    return None


async def _cmd_branch(arg: str, agent, state: ReplState, session_id: str):
    """Fork the current session under a new id, optionally with a name."""
    try:
        from claude_agent_sdk import fork_session  # type: ignore
    except ImportError:
        console.print("[josh.error]fork_session not available in your claude_agent_sdk[/]")
        return None
    try:
        result = fork_session(session_id, title=arg or None)  # type: ignore[arg-type]
        new_id = getattr(result, "session_id", arg) or session_id + "_fork"
    except Exception as e:  # noqa: BLE001
        console.print(f"[josh.error]branch failed:[/] {e}")
        return None
    console.print(
        f"  [josh.success]branched[/] [dim]from {session_id[:16]}…[/] "
        f"[josh.success]→[/] [josh.info]{new_id}[/]"
    )
    return ("session", new_id)


async def _cmd_compress(arg: str, agent, state: ReplState):
    """Have the agent summarize-then-replace its context. Stub-real: asks the
    agent itself to do the summarization on the next turn; cannot actually
    rewrite the Claude Code session transcript from outside."""
    focus = arg.strip() or "the whole conversation so far"
    prompt = (
        f"Compress this conversation. Summarize {focus} into a tight set of "
        f"key facts, decisions, open questions, and code-state notes that "
        f"would let a fresh session pick up where we left off. Use bullets. "
        f"Don't re-derive — just preserve. After the summary, ack with "
        f"\"context compressed\" so I know it's done."
    )
    console.print("  [dim]queuing context compression request as next turn…[/]")
    return ("retry", prompt)


def _cmd_rollback(arg: str) -> None:
    """File-checkpoint rollback. STUB — joshv1 doesn't yet have a
    filesystem-snapshot system. Shows what it would do."""
    console.print(
        "  [josh.highlight]/rollback[/] [dim]is a stub.[/] "
        "Would list filesystem checkpoints taken at risk-points (before /yolo "
        "actions, before destructive edits) and restore on selection. "
        "Implement: hook the agent's Write/Edit/Bash tool calls to snapshot "
        "touched files under ~/.josh-memory/checkpoints/ before the edit."
    )
    return None


def _snapshot_dir() -> Path:
    from joshv1.memory import memory_dir as _mem_dir
    d = _mem_dir("~/.josh-memory") / "snapshots"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cmd_snapshot(arg: str, cfg: Config, state: ReplState, current_model: str) -> None:
    """JSON snapshot of cfg + state. Round-trippable via `restore <id>`."""
    sub, _, rest = arg.partition(" ")
    sub = sub.strip().lower() or "create"
    snap_dir = _snapshot_dir()

    if sub == "list":
        snaps = sorted(snap_dir.glob("*.json"))
        if not snaps:
            console.print("[dim]no snapshots yet[/]")
            return None
        console.print("\n[josh.title]Snapshots[/]")
        for s in snaps:
            console.print(f"  [josh.info]{s.stem}[/]  [dim]({s.stat().st_size} bytes)[/]")
        console.print()
        return None

    if sub == "prune":
        snaps = sorted(snap_dir.glob("*.json"))
        keep = 5
        for old in snaps[:-keep] if len(snaps) > keep else []:
            old.unlink()
        console.print(f"  [josh.success]pruned[/] [dim](kept latest {keep})[/]")
        return None

    if sub == "restore":
        if not rest:
            console.print("  [josh.highlight]usage:[/] /snapshot restore <id>")
            return None
        target = snap_dir / f"{rest}.json"
        if not target.is_file():
            console.print(f"[josh.error]no snapshot[/] [dim]{rest}[/]")
            return None
        data = json.loads(target.read_text())
        state.personality = data.get("personality", state.personality)
        state.verbose_level = data.get("verbose_level", state.verbose_level)
        state.busy_mode = data.get("busy_mode", state.busy_mode)
        state.statusbar_enabled = data.get("statusbar_enabled", state.statusbar_enabled)
        console.print(f"  [josh.success]restored[/] [josh.info]{rest}[/] [dim](state flags only — agent session unchanged)[/]")
        return None

    # create (default)
    snap_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    payload = {
        "id": snap_id,
        "saved_at": datetime.now().isoformat(),
        "session_id": state.session_id,
        "model": current_model,
        "personality": state.personality,
        "verbose_level": state.verbose_level,
        "busy_mode": state.busy_mode,
        "statusbar_enabled": state.statusbar_enabled,
        "user_history": state.user_history[-50:],
        "assistant_history": state.assistant_history[-50:],
    }
    (snap_dir / f"{snap_id}.json").write_text(json.dumps(payload, indent=2))
    console.print(f"  [josh.success]snapshot[/] [josh.info]{snap_id}[/] [dim]({snap_dir / (snap_id + '.json')})[/]")
    return None


def _cmd_stop(state: ReplState) -> None:
    """Cancel all running background tasks."""
    if not state.background_tasks:
        console.print("[dim]no background tasks running[/]")
        return None
    n = 0
    for bg_id, bg in list(state.background_tasks.items()):
        if not bg.done:
            bg.task.cancel()
            n += 1
        del state.background_tasks[bg_id]
    console.print(f"  [josh.success]cancelled {n} task(s)[/]")
    return None


async def _cmd_background(arg: str, agent, state: ReplState, session_id: str,
                          stats: SessionStats, current_model: str):
    """Spawn a prompt as a fire-and-forget asyncio task."""
    if not arg:
        console.print("  [josh.highlight]usage:[/] /background <prompt>")
        return None
    if not isinstance(agent, Agent):
        console.print("[dim]/background only works with the Claude backend.[/]")
        return None

    bg_id = f"bg{state.next_bg_id}"
    state.next_bg_id += 1

    async def _run():
        # Fresh agent so the background task doesn't interleave with the
        # foreground turn's tool calls. Same session id so the conversation
        # picks up the result.
        from joshv1.config import AgentSettings as _AS  # noqa: F401  (type hint clarity)
        async with Agent(agent.settings, cwd=agent.settings.workspace_dir) as bg_agent:
            try:
                return await bg_agent.run(arg)
            except Exception as e:  # noqa: BLE001
                return f"[bg error] {e}"

    task = asyncio.create_task(_run())
    bg = BackgroundTask(
        id=bg_id, prompt=arg, started_at=time.monotonic(), task=task,
    )

    def _on_done(t: Any) -> None:
        bg.done = True
        try:
            bg.result = t.result()
        except asyncio.CancelledError:
            bg.error = "cancelled"
        except Exception as e:  # noqa: BLE001
            bg.error = str(e)

    task.add_done_callback(_on_done)
    state.background_tasks[bg_id] = bg
    console.print(
        f"  [josh.success]started[/] [josh.info]{bg_id}[/] "
        f"[dim]({arg[:60]}{'…' if len(arg) > 60 else ''})[/]"
    )
    return None


def _cmd_agents(state: ReplState) -> None:
    """List active background tasks."""
    if not state.background_tasks:
        console.print("[dim]no background tasks[/]")
        return None
    console.print("\n[josh.title]Background tasks[/]")
    for bg_id, bg in state.background_tasks.items():
        elapsed = time.monotonic() - bg.started_at
        status = (
            "[josh.success]done[/]" if bg.done and not bg.error
            else f"[josh.error]err[/] {bg.error}" if bg.error
            else "[josh.highlight]running[/]"
        )
        console.print(f"  [josh.info]{bg_id}[/]  {status}  [dim]{elapsed:.1f}s[/]")
        console.print(f"    [dim]prompt:[/] {bg.prompt[:100]}{'…' if len(bg.prompt) > 100 else ''}")
        if bg.done and bg.result and not bg.error:
            preview = bg.result[:200].replace("\n", " ")
            console.print(f"    [dim]result:[/] {preview}{'…' if len(bg.result) > 200 else ''}")
    console.print()
    return None


def _cmd_queue(arg: str, state: ReplState) -> None:
    """Queue a prompt to run after the current turn finishes."""
    if not arg:
        if not state.queued_prompts:
            console.print("[dim]queue is empty[/]")
            return None
        console.print("\n[josh.title]Queued prompts[/]")
        for i, p in enumerate(state.queued_prompts, 1):
            console.print(f"  [dim]{i}.[/] {p[:120]}{'…' if len(p) > 120 else ''}")
        console.print()
        return None
    state.queued_prompts.append(arg)
    console.print(f"  [josh.success]queued[/] [dim]({len(state.queued_prompts)} pending)[/]")
    return None


def _cmd_steer(arg: str, state: ReplState) -> None:
    """Inject a steer message. STUB-ish — we record it; the tool-call event
    handler (in _stream_one_turn) would need to consume it for true mid-turn
    injection. Right now it's drained the same as /queue at next-turn time."""
    if not arg:
        if not state.steer_prompts:
            console.print("[dim]no steer messages pending[/]")
            return None
        for i, s in enumerate(state.steer_prompts, 1):
            console.print(f"  [dim]{i}.[/] {s}")
        return None
    state.steer_prompts.append(arg)
    state.queued_prompts.append(f"[steer] {arg}")  # also queue so it runs next turn
    console.print(f"  [josh.success]steer queued[/] [dim](mid-turn injection requires a future hook into _stream_one_turn)[/]")
    return None


async def _cmd_status(agent, state: ReplState, current_model: str,
                      stats: SessionStats, session_id: str) -> None:
    ctx = await _ctx_pct(agent)
    items = [
        ("session", session_id),
        ("model", current_model),
        ("turns", str(stats.turns)),
        ("elapsed", f"{stats.session_seconds:.0f}s"),
        ("context", f"{ctx:.0f}%" if ctx is not None else "--"),
        ("user_history", str(len(state.user_history))),
        ("queued", str(len(state.queued_prompts))),
        ("background", str(len(state.background_tasks))),
        ("personality", state.personality),
        ("verbose", state.verbose_level),
        ("busy_mode", state.busy_mode),
        ("statusbar", "on" if state.statusbar_enabled else "off"),
    ]
    tui.render_kv_block(console, "Status", items)
    return None


def _cmd_resume(arg: str):
    """Resume a previously-named session. Re-uses the existing reset path —
    the agent rebuild happens on the next /chat invocation; for now we
    just print instructions since runtime resume isn't supported mid-loop."""
    if not arg:
        console.print("  [josh.highlight]usage:[/] /resume <session_name>")
        console.print("  [dim]for full resume, exit and run:[/] joshv1 chat --session <name>")
        return None
    console.print(
        f"  [dim]queued resume of[/] [josh.info]{arg}[/] "
        f"[dim](exit and re-run with `joshv1 chat --session {arg}` to actually resume)[/]"
    )
    return None


# ----- Info group ----------------------------------------------------------

def _cmd_profile(cfg: Config) -> None:
    items = [
        ("user", os.environ.get("USER", "?")),
        ("home", str(Path.home())),
        ("memory_dir", str(Path(cfg.agent.memory_dir).expanduser())),
        ("workspace_dir", str(Path(cfg.agent.workspace_dir).expanduser())),
        ("config_origin", "config.yaml + env"),
    ]
    tui.render_kv_block(console, "Profile", items)
    return None


async def _cmd_gquota():
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        console.print("[dim]no GEMINI_API_KEY / GOOGLE_API_KEY in env[/]")
        return None
    console.print("  [dim]/gquota is a stub.[/] Google doesn't expose a free-tier quota endpoint;")
    console.print("  [dim]check https://aistudio.google.com/app/usage manually.[/]")
    return None


def _cmd_usage(stats: SessionStats, current_model: str) -> None:
    items = [
        ("model", current_model),
        ("turns", str(stats.turns)),
        ("input tokens", f"{stats.total_input:,}"),
        ("output tokens", f"{stats.total_output:,}"),
        ("total tokens", f"{stats.total_input + stats.total_output:,}"),
        ("cost (equiv)", f"${stats.total_cost:.4f}"),
        ("session time", f"{stats.session_seconds:.0f}s"),
        ("billed via", "Anthropic Max subscription"),
    ]
    tui.render_kv_block(console, "Usage", items)
    return None


def _cmd_insights(arg: str, cfg: Config) -> None:
    """Count saved sessions / snapshots in memory_dir. Very lightweight."""
    from joshv1.memory import memory_dir as _mem_dir
    days = 7
    try:
        days = int(arg) if arg else 7
    except ValueError:
        pass
    mem = _mem_dir(cfg.agent.memory_dir)
    saves = list((mem / "saves").glob("*.md")) if (mem / "saves").is_dir() else []
    snaps = list((mem / "snapshots").glob("*.json")) if (mem / "snapshots").is_dir() else []
    cutoff = time.time() - days * 86400
    recent_saves = [s for s in saves if s.stat().st_mtime > cutoff]
    recent_snaps = [s for s in snaps if s.stat().st_mtime > cutoff]
    items = [
        ("window", f"last {days} days"),
        ("saved conversations", f"{len(recent_saves)} (of {len(saves)} total)"),
        ("snapshots", f"{len(recent_snaps)} (of {len(snaps)} total)"),
        ("memory dir", str(mem)),
    ]
    tui.render_kv_block(console, "Insights", items)
    return None


def _cmd_platforms(cfg: Config) -> None:
    slack_ok = bool(cfg.slack.get("bot_token"))
    discord_ok = bool(cfg.discord.get("bot_token"))
    items = [
        ("Slack",    "[josh.success]configured[/]" if slack_ok else "[dim]no SLACK_BOT_TOKEN[/]"),
        ("Discord",  "[josh.success]configured[/]" if discord_ok else "[dim]no DISCORD_BOT_TOKEN[/]"),
    ]
    tui.render_kv_block(console, "Gateway platforms", items)
    if slack_ok:
        console.print("  [dim]run `joshv1 slack run` to start the Slack gateway.[/]")
    if discord_ok:
        console.print("  [dim]run `joshv1 discord run` to start the Discord gateway.[/]")
    return None


def _clip_copy(text: str) -> tuple[bool, str]:
    """Cross-platform clipboard write. Returns (ok, tool-name)."""
    candidates = [
        (["wl-copy"],          "wl-copy"),
        (["xclip", "-selection", "clipboard"], "xclip"),
        (["xsel", "--clipboard", "--input"],   "xsel"),
        (["pbcopy"],           "pbcopy"),
        (["clip.exe"],         "clip.exe (WSL)"),
    ]
    for argv, name in candidates:
        if shutil.which(argv[0]):
            try:
                subprocess.run(argv, input=text, text=True, check=True, timeout=5)
                return True, name
            except (subprocess.SubprocessError, OSError):
                continue
    return False, ""


def _clip_paste() -> tuple[str | None, str]:
    """Read the clipboard as text. Returns (text-or-None, tool-name)."""
    candidates = [
        (["wl-paste"],           "wl-paste"),
        (["xclip", "-selection", "clipboard", "-o"], "xclip"),
        (["xsel", "--clipboard", "--output"],        "xsel"),
        (["pbpaste"],            "pbpaste"),
        (["powershell.exe", "-NoProfile", "-Command", "Get-Clipboard"], "powershell.exe (WSL)"),
    ]
    for argv, name in candidates:
        if shutil.which(argv[0]):
            try:
                r = subprocess.run(argv, capture_output=True, text=True, check=True, timeout=5)
                return r.stdout, name
            except (subprocess.SubprocessError, OSError):
                continue
    return None, ""


def _cmd_copy(arg: str, state: ReplState) -> None:
    if not state.assistant_history:
        console.print("[dim]no assistant replies yet to copy.[/]")
        return None
    n = 1
    if arg:
        try:
            n = int(arg)
        except ValueError:
            pass
    if not (1 <= n <= len(state.assistant_history)):
        console.print(f"[josh.error]bad index[/] [dim](have {len(state.assistant_history)} replies)[/]")
        return None
    text = state.assistant_history[-n]
    ok, tool = _clip_copy(text)
    if ok:
        console.print(f"  [josh.success]copied {len(text)} chars[/] [dim](via {tool})[/]")
    else:
        console.print(
            "  [josh.error]no clipboard tool found.[/] "
            "[dim]install one of: wl-clipboard, xclip, xsel, pbcopy, clip.exe[/]"
        )
    return None


def _cmd_paste() -> None:
    text, tool = _clip_paste()
    if text is None:
        console.print("  [josh.error]no clipboard tool found.[/]")
        return None
    short = text.strip()
    if not short:
        console.print("[dim]clipboard is empty[/]")
        return None
    preview = short[:200].replace("\n", " ")
    console.print(f"  [josh.success]clipboard:[/] [dim]{preview}{'…' if len(short) > 200 else ''}[/]")
    console.print("  [dim](paste at the prompt — true image-clipboard support is a TODO)[/]")
    return None


def _cmd_debug(cfg: Config, state: ReplState, stats: SessionStats, current_model: str) -> None:
    """Write a debug report under memory_dir/debug/ and print the path."""
    from joshv1.memory import memory_dir as _mem_dir
    out_dir = _mem_dir(cfg.agent.memory_dir) / "debug"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    info = [
        f"# joshv1 debug report — {datetime.now().isoformat()}",
        f"\n## Environment",
        f"python:    {sys.version.split()[0]}",
        f"platform:  {platform.platform()}",
        f"machine:   {platform.machine()}",
        f"\n## Session",
        f"session_id:   {state.session_id}",
        f"model:        {current_model}",
        f"provider:     {cfg.agent.provider}",
        f"effort:       {cfg.agent.effort}",
        f"turns:        {stats.turns}",
        f"input toks:   {stats.total_input}",
        f"output toks:  {stats.total_output}",
        f"\n## State flags",
        f"personality:  {state.personality}",
        f"verbose:      {state.verbose_level}",
        f"busy_mode:    {state.busy_mode}",
        f"statusbar:    {state.statusbar_enabled}",
        f"queued:       {len(state.queued_prompts)}",
        f"background:   {len(state.background_tasks)}",
        f"\n## Ensemble",
        f"mode:         {'on' if cfg.agent.ensemble_mode else 'off'}",
        f"members:      {len(cfg.agent.ensemble) or '(defaults)'}",
        f"\n## Recent user prompts (last 5)",
        *[f"  {i+1}. {p[:120]}" for i, p in enumerate(state.user_history[-5:])],
    ]
    out.write_text("\n".join(info))
    console.print(f"  [josh.success]wrote debug report →[/] [josh.info]{out}[/]")
    return None


# ----- Configuration group -------------------------------------------------

def _cmd_config(cfg: Config, state: ReplState) -> None:
    items = [
        ("provider",         cfg.agent.provider),
        ("model",            cfg.agent.model),
        ("effort",           cfg.agent.effort),
        ("thinking",         cfg.agent.thinking),
        ("permission_mode",  cfg.agent.permission_mode),
        ("workspace_dir",    cfg.agent.workspace_dir),
        ("memory_dir",       cfg.agent.memory_dir),
        ("mcp_servers",      str(len(cfg.agent.mcp_servers))),
        ("skills",           str(cfg.agent.skills)),
        ("ensemble_mode",    "on" if cfg.agent.ensemble_mode else "off"),
        ("ensemble_size",    str(len(cfg.agent.ensemble) or "(defaults)")),
        ("personality",      state.personality),
    ]
    tui.render_kv_block(console, "Configuration", items, key_width=18)
    return None


async def _cmd_personality(arg: str, cfg: Config, agent, state: ReplState):
    if not arg:
        console.print("\n[josh.title]Personalities[/]")
        for name, prompt in PERSONALITIES.items():
            marker = " [josh.success]← active[/]" if name == state.personality else ""
            preview = (prompt or "use default system prompt")[:80]
            console.print(f"  [josh.info]{name:<12}[/]  [dim]{preview}…[/]{marker}")
        console.print()
        return None
    if arg not in PERSONALITIES:
        console.print(f"  [josh.error]unknown personality:[/] {arg}")
        console.print(f"  [dim]valid:[/] {', '.join(PERSONALITIES.keys())}")
        return None
    state.personality = arg
    addendum = PERSONALITIES[arg]
    if addendum:
        # Append to current system prompt rather than replacing — we want the
        # base joshv1 behavior preserved.
        cfg.agent.system_prompt = (cfg.agent.system_prompt.split(
            "\n\n--- PERSONALITY ---\n\n", 1)[0]) + "\n\n--- PERSONALITY ---\n\n" + addendum
    if isinstance(agent, Agent) and agent._client is not None:
        try:
            await agent.reconfigure()
            console.print(f"  [josh.success]personality →[/] [josh.info]{arg}[/]")
        except Exception as e:  # noqa: BLE001
            console.print(f"  [josh.error]reconnect failed:[/] {e}")
    else:
        console.print(f"  [josh.success]personality →[/] [josh.info]{arg}[/] [dim](applies next turn)[/]")
    return None


def _cmd_statusbar(arg: str, state: ReplState) -> None:
    if arg.lower() == "on":
        state.statusbar_enabled = True
    elif arg.lower() == "off":
        state.statusbar_enabled = False
    else:
        state.statusbar_enabled = not state.statusbar_enabled
    console.print(f"  status bar → {'[josh.success]on[/]' if state.statusbar_enabled else '[dim]off[/]'}")
    return None


def _cmd_verbose(state: ReplState) -> None:
    idx = (VERBOSE_LEVELS.index(state.verbose_level) + 1) % len(VERBOSE_LEVELS)
    state.verbose_level = VERBOSE_LEVELS[idx]
    console.print(f"  verbose → [josh.info]{state.verbose_level}[/] [dim]({' → '.join(VERBOSE_LEVELS)})[/]")
    console.print("  [dim]note: actual filtering hookup in _render_event_tui is a follow-up. Flag is stored.[/]")
    return None


async def _cmd_reasoning(arg: str, agent, cfg: Config):
    """Wrapper around /effort plus show/hide thinking-display."""
    if arg in ("show", "hide"):
        cfg.agent.thinking_display = "summarized" if arg == "show" else "omitted"
        if isinstance(agent, Agent) and agent._client is not None:
            try:
                await agent.reconfigure()
            except Exception as e:  # noqa: BLE001
                console.print(f"[josh.error]reconnect failed:[/] {e}")
                return None
        console.print(f"  reasoning display → [josh.info]{cfg.agent.thinking_display}[/]")
        return None
    if not arg:
        items = [
            ("effort",  cfg.agent.effort),
            ("display", cfg.agent.thinking_display),
        ]
        tui.render_kv_block(console, "Reasoning", items)
        return None
    # Else treat the arg as an effort level and delegate.
    if arg in VALID_EFFORTS:
        cfg.agent.effort = arg
        if isinstance(agent, Agent) and agent._client is not None:
            try:
                await agent.reconfigure()
            except Exception as e:  # noqa: BLE001
                console.print(f"[josh.error]reconnect failed:[/] {e}")
                return None
        console.print(f"  effort → [josh.info]{arg}[/]")
        return None
    console.print(f"  [josh.highlight]usage:[/] /reasoning [{'|'.join(VALID_EFFORTS)}|show|hide]")
    return None


def _cmd_skin(arg: str) -> None:
    if not arg:
        themes_list = themes.list_themes()
        active = themes.load_active_palette()
        console.print("\n[josh.title]Available skins[/]")
        for p in themes_list:
            marker = " [josh.success]← active[/]" if p.name == active.name else ""
            console.print(f"  [josh.info]{p.name}[/]  [dim]{themes.render_theme_swatch(p)}[/]{marker}")
        console.print()
        return None
    try:
        themes.get_palette(arg)
    except (KeyError, ValueError):
        console.print(f"  [josh.error]unknown skin:[/] {arg}")
        return None
    themes.save_active_theme(arg)
    console.print(f"  [josh.success]skin →[/] [josh.info]{arg}[/] [dim](restart joshv1 to fully apply)[/]")
    return None


def _cmd_voice(arg: str):
    """Transcribe an audio file with faster-whisper and use as the next prompt."""
    if not arg:
        console.print("  [josh.highlight]usage:[/] /voice <audio-file>")
        return None
    path = Path(arg).expanduser()
    if not path.is_file():
        console.print(f"[josh.error]not a file:[/] {path}")
        return None
    try:
        from joshv1 import voice as voice_mod
    except ImportError:
        console.print("[josh.error]voice module missing[/]")
        return None
    try:
        # voice.py exposes a transcribe-like entry; check there for current API
        text = voice_mod.transcribe(str(path)) if hasattr(voice_mod, "transcribe") else None
    except Exception as e:  # noqa: BLE001
        console.print(f"[josh.error]transcription failed:[/] {e}")
        return None
    if not text:
        console.print(
            "[dim]/voice transcribe() not exposed yet. Use: `joshv1 voice <file>` instead, "
            "or wire `voice.transcribe(path)` into voice.py.[/]"
        )
        return None
    console.print(f"  [josh.success]transcribed[/] [dim]({len(text)} chars)[/]")
    return ("retry", text.strip())


def _cmd_busy(arg: str, state: ReplState) -> None:
    if not arg or arg == "status":
        console.print(f"  busy_mode = [josh.info]{state.busy_mode}[/] [dim](options: {', '.join(BUSY_MODES)})[/]")
        return None
    if arg not in BUSY_MODES:
        console.print(f"  [josh.error]unknown mode:[/] {arg}")
        return None
    state.busy_mode = arg
    console.print(f"  busy_mode → [josh.info]{arg}[/]")
    return None


# ----- Tools & Skills group ------------------------------------------------

def _cmd_toolsets() -> None:
    """List toolset groups. joshv1 has no formal toolset registry; show the
    built-in tool groups from tui.TOOL_GROUPS as a stand-in."""
    if hasattr(tui, "TOOL_GROUPS"):
        console.print("\n[josh.title]Toolsets[/]")
        for group, tools in tui.TOOL_GROUPS.items():
            console.print(f"  [josh.info]{group}[/]  [dim]({len(tools)} tools)[/]")
        console.print()
    else:
        console.print("[dim]no toolset registry yet[/]")
    return None


def _cmd_skills(arg: str) -> None:
    from joshv1 import skills as skills_mod
    sub, _, rest = arg.partition(" ")
    sub = sub.strip().lower() or "list"
    if sub == "list":
        installed = skills_mod.list_installed()
        if not installed:
            console.print("[dim]no skills installed[/]")
            return None
        console.print(f"\n[josh.title]Installed skills[/] [dim]({len(installed)})[/]")
        for s in installed:
            console.print(f"  [josh.info]{s.name}[/]  [dim]{(s.description or '')[:80]}[/]")
        console.print()
        return None
    if sub == "browse":
        items = skills_mod.browse()
        if not items:
            console.print("[dim]marketplace is empty[/]")
            return None
        console.print(f"\n[josh.title]Marketplace[/] [dim]({len(items)} skills)[/]")
        for s in items[:30]:
            console.print(f"  [josh.info]{s.get('name','?')}[/]  [dim]{(s.get('description') or '')[:80]}[/]")
        console.print()
        return None
    if sub == "search" and rest:
        hits = skills_mod.search(rest)
        if not hits:
            console.print(f"[dim]no matches for {rest!r}[/]")
            return None
        for s in hits[:20]:
            console.print(f"  [josh.info]{s.get('name','?')}[/]  [dim]{(s.get('description') or '')[:80]}[/]")
        return None
    if sub == "inspect" and rest:
        body = skills_mod.inspect_skill(rest)
        if not body:
            console.print(f"[dim]no installed skill[/] {rest!r}")
            return None
        console.print(body)
        return None
    if sub == "install" and rest:
        try:
            entry = skills_mod.install(rest)
        except Exception as e:  # noqa: BLE001
            console.print(f"[josh.error]install failed:[/] {e}")
            return None
        console.print(f"  [josh.success]installed[/] [josh.info]{entry.name}[/]")
        return None
    if sub in ("uninstall", "remove") and rest:
        try:
            skills_mod.uninstall(rest)
        except Exception as e:  # noqa: BLE001
            console.print(f"[josh.error]uninstall failed:[/] {e}")
            return None
        console.print(f"  [josh.success]uninstalled[/] [josh.info]{rest}[/]")
        return None
    console.print("  [josh.highlight]usage:[/] /skills [list|browse|search <q>|inspect <name>|install <id>|uninstall <name>]")
    return None


def _cmd_cron(arg: str) -> None:
    """Cron stub — joshv1 doesn't ship a persistent scheduler. Suggests system cron."""
    console.print(
        "  [josh.highlight]/cron[/] [dim]is a stub.[/] joshv1 doesn't have an in-process "
        "scheduler. For recurring tasks, wire up system cron / launchd / "
        "Windows Task Scheduler to invoke `joshv1 run <prompt>`."
    )
    return None


def _cmd_reload() -> None:
    """Re-read .env into the running process."""
    from dotenv import load_dotenv
    home_env = Path.home() / ".env"
    cwd_env = Path.cwd() / ".env"
    n = 0
    for p in (home_env, cwd_env):
        if p.is_file():
            load_dotenv(p, override=True)
            n += 1
    console.print(f"  [josh.success]reloaded {n} .env file(s)[/]")
    return None


async def _cmd_reload_mcp(agent):
    if not isinstance(agent, Agent) or agent._client is None:
        console.print("[dim]/reload-mcp only applies to the Claude backend.[/]")
        return None
    try:
        await agent.reconfigure()
    except Exception as e:  # noqa: BLE001
        console.print(f"[josh.error]reconfigure failed:[/] {e}")
        return None
    console.print(f"  [josh.success]agent reconfigured[/] [dim](MCP servers re-read from cfg)[/]")
    return None


def _cmd_browser(arg: str) -> None:
    """Browser CDP stub. Future: connect a Playwright client to a live Chrome
    via remote-debugging-port and expose page tools."""
    console.print(
        "  [josh.highlight]/browser[/] [dim]is a stub.[/] Would connect to a live Chrome "
        "via CDP (--remote-debugging-port=9222) and expose page tools. "
        "Install playwright + implement BrowserSession in joshv1.browser."
    )
    return None


def _cmd_plugins() -> None:
    """List installed Claude Agent SDK plugins, if any."""
    plugins_dir = Path.home() / ".claude" / "plugins"
    if not plugins_dir.is_dir():
        console.print(f"[dim]no plugins dir at {plugins_dir}[/]")
        return None
    plugs = sorted(p.name for p in plugins_dir.iterdir() if p.is_dir())
    if not plugs:
        console.print("[dim]no plugins installed[/]")
        return None
    console.print(f"\n[josh.title]Plugins[/] [dim]({plugins_dir})[/]")
    for p in plugs:
        console.print(f"  [josh.info]{p}[/]")
    console.print()
    return None


_ENSEMBLE_SUBCOMMANDS = {
    "on", "off", "toggle", "status",
    "list", "models", "add", "remove", "rm", "reset", "run", "retune",
}


async def _retune_ensemble_roles(cfg: Config) -> None:
    """Ask Claude to evaluate every ensemble member and rewrite each role
    to match what the model is actually best at. Persists the new roles.
    """
    import json
    import re
    from dataclasses import replace

    from joshv1 import ensemble as ens_mod
    from joshv1.agent import Agent as _Agent

    members = ens_mod.members_from_settings(cfg.agent)
    if not members:
        console.print("  [dim]ensemble is empty — nothing to retune[/]")
        return

    model_list = "\n".join(f"- {m.provider} / {m.model}" for m in members)
    judge_prompt = (
        "You are configuring an LLM ensemble. For each model below, write a "
        "concise role description (one line, comma-separated specialties, "
        "~5-12 words) reflecting what THAT SPECIFIC MODEL is actually best "
        "at relative to typical peers. Be HONEST: if a model is a generalist, "
        "say so; if its 'specialty' is mostly hype, drop it. Avoid fabricating "
        "strengths to make roles sound impressive. The roles you write will "
        "be injected into each drafter's system prompt verbatim, so make them "
        "useful behavioral cues, not marketing copy.\n\n"
        f"Models:\n{model_list}\n\n"
        "Output ONLY a JSON array with this exact shape — no markdown fences, "
        "no preamble, no trailing text:\n"
        '[{"model": "<exact model id from list>", "role": "<one-line role>"}, ...]'
    )

    console.print(
        f"  [dim]asking Claude to evaluate {len(members)} model(s)...[/]"
    )

    # Use the user's main model if it's Claude; otherwise fall back to opus-4-7
    # — same pattern as the synth step in ensemble.py.
    claude_model = (
        cfg.agent.model if cfg.agent.provider == "claude" else "claude-opus-4-7"
    )
    judge_settings = replace(
        cfg.agent,
        provider="claude",
        model=claude_model,
        system_prompt=(
            "You are a senior ML engineer rating LLMs honestly based on "
            "publicly documented strengths and benchmark behavior. You "
            "produce concise, behavior-shaping role strings — never "
            "marketing copy. You output strictly the JSON requested."
        ),
    )
    try:
        async with _Agent(judge_settings) as judge:
            raw = await judge.run(judge_prompt)
    except Exception as e:  # noqa: BLE001
        console.print(f"  [josh.error]judge call failed:[/] {e}")
        return

    # Strip accidental ```json fences if Claude includes them anyway.
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(),
                     flags=re.MULTILINE).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        console.print(f"  [josh.error]couldn't parse judge response as JSON:[/] {e}")
        console.print(f"  [dim]raw output:[/]\n{raw[:500]}")
        return

    if not isinstance(data, list):
        console.print(f"  [josh.error]judge returned non-list:[/] {type(data).__name__}")
        return

    by_model = {
        entry["model"]: entry["role"]
        for entry in data
        if isinstance(entry, dict) and "model" in entry and "role" in entry
    }

    new_list: list[dict[str, str]] = []
    changed = 0
    for m in members:
        new_role = by_model.get(m.model, m.role)
        if new_role != m.role:
            console.print(f"  [josh.info]{m.model}[/]")
            console.print(f"    [dim]was:[/] {m.role}")
            console.print(f"    [josh.success]now:[/] {new_role}")
            changed += 1
        new_list.append({"provider": m.provider, "model": m.model, "role": new_role})

    if changed == 0:
        console.print("  [dim]no changes — roles already match Claude's evaluation[/]")
        return

    try:
        save_ensemble_members_override(new_list, cfg.agent.memory_dir)
    except Exception as e:  # noqa: BLE001
        console.print(f"  [josh.error]failed to persist new roles:[/] {e}")
        return
    cfg.agent.ensemble = new_list
    console.print(
        f"  [josh.success]updated {changed}/{len(members)} role(s)[/] "
        f"[dim](saved to memory override)[/]"
    )


def _print_ensemble_status(cfg: Config) -> None:
    """Print the ensemble mode + member list. Used by `/ensemble` no-args + status."""
    from joshv1 import ensemble as ens_mod
    members = ens_mod.members_from_settings(cfg.agent)
    mode = "[josh.success]on[/]" if cfg.agent.ensemble_mode else "[dim]off[/]"
    console.print(f"\n[josh.title]Ensemble[/] · mode: {mode}")
    console.print(f"[dim]  {len(members)} member(s):[/]")
    for i, m in enumerate(members, 1):
        console.print(
            f"  [josh.highlight.bold]{i}.[/] "
            f"[josh.info]{m.provider:12s}[/]  "
            f"{m.model:55s}  [dim]{m.role}[/]"
        )
    console.print()
    console.print("[dim]  /ensemble on|off|toggle              toggle auto-routing[/]")
    console.print("[dim]  /ensemble add <provider> <model> [role]   add a model[/]")
    console.print("[dim]  /ensemble remove <model_or_index>    drop a model[/]")
    console.print("[dim]  /ensemble retune                     ask Claude to rewrite every role[/]")
    console.print("[dim]  /ensemble reset                      restore built-in defaults[/]")
    console.print("[dim]  /ensemble run <prompt>               one-shot run[/]")
    console.print()


async def _run_ensemble_once(prompt: str, cfg: Config) -> None:
    """One-shot ensemble run — shared by `/ensemble run …` and auto-routing mode."""
    from joshv1 import ensemble as ens_mod
    members = ens_mod.members_from_settings(cfg.agent)
    console.print(f"  [dim]polling {len(members)} models in parallel...[/]")

    def _on_draft(member, draft):
        if draft.success:
            console.print(
                f"  [josh.success]✓[/] [josh.info]{member.model}[/] "
                f"[dim]({len(draft.text)} chars)[/]"
            )
        else:
            console.print(f"  [josh.error]✗[/] [josh.info]{member.model}[/]")
            console.print(f"      [dim]error: {draft.error!r}[/]")
            if draft.text:
                console.print(f"      [dim]text:  {draft.text[:160]!r}[/]")

    answer, drafts = await ens_mod.run_ensemble(
        prompt, cfg.agent, members=members, on_draft_complete=_on_draft,
    )
    n_ok = sum(1 for d in drafts if d.success)
    console.print(
        f"\n[josh.title]Synthesized answer[/] "
        f"[dim](from {n_ok}/{len(drafts)} drafts)[/]\n"
    )
    tui.render_assistant_markdown(console, answer)
    console.print()


def _ensemble_current_member_dicts(cfg: Config) -> list[dict[str, str]]:
    """Snapshot the *effective* member list as plain dicts, for mutation+persist."""
    from joshv1 import ensemble as ens_mod
    return [
        {"provider": m.provider, "model": m.model, "role": m.role}
        for m in ens_mod.members_from_settings(cfg.agent)
    ]


def _set_ensemble_mode(cfg: Config, enabled: bool) -> None:
    cfg.agent.ensemble_mode = enabled
    try:
        save_active_ensemble_mode(enabled, cfg.agent.memory_dir)
    except Exception as e:  # noqa: BLE001
        console.print(f"  [josh.error]failed to persist mode:[/] {e}")
        return
    label = "[josh.success]ON[/]" if enabled else "[dim]OFF[/]"
    console.print(f"  ensemble mode → {label}")
    if enabled:
        console.print(
            "  [dim]every non-slash prompt will fan out to your ensemble for "
            "drafts in parallel, then your main Claude session uses those "
            "drafts as context to produce the reply. tools, scrollback, and "
            "session memory are preserved.[/]"
        )


async def _handle_ensemble(arg: str, cfg: Config) -> None:
    """Dispatcher for `/ensemble [subcommand] [args...]`."""
    sub, _, rest = arg.partition(" ")
    sub = sub.strip().lower()
    rest = rest.strip()

    # No args: show status + usage hints.
    if not arg.strip():
        _print_ensemble_status(cfg)
        return None

    # Mode toggle.
    if sub == "on":
        _set_ensemble_mode(cfg, True)
        return None
    if sub == "off":
        _set_ensemble_mode(cfg, False)
        return None
    if sub == "toggle":
        _set_ensemble_mode(cfg, not cfg.agent.ensemble_mode)
        return None
    if sub == "status":
        _print_ensemble_status(cfg)
        return None

    # Member management.
    if sub in ("list", "models"):
        _print_ensemble_status(cfg)
        return None

    if sub == "retune":
        await _retune_ensemble_roles(cfg)
        return None

    if sub == "add":
        parts = rest.split(maxsplit=2)
        if len(parts) < 2:
            console.print(
                "  [josh.highlight]usage:[/] "
                "/ensemble add <provider> <model> [role description]"
            )
            return None
        provider, model = parts[0], parts[1]
        role = parts[2] if len(parts) > 2 else ""
        current = _ensemble_current_member_dicts(cfg)
        # Reject duplicates so add isn't silently a no-op.
        if any(m["model"] == model and m["provider"] == provider for m in current):
            console.print(f"  [dim]already in ensemble:[/] {provider} / {model}")
            return None
        current.append({"provider": provider, "model": model, "role": role})
        try:
            save_ensemble_members_override(current, cfg.agent.memory_dir)
        except Exception as e:  # noqa: BLE001
            console.print(f"  [josh.error]failed to save:[/] {e}")
            return None
        cfg.agent.ensemble = current
        console.print(
            f"  [josh.success]added[/] [josh.info]{provider} / {model}[/] "
            f"[dim]({len(current)} members)[/]"
        )
        return None

    if sub in ("remove", "rm"):
        if not rest:
            console.print("  [josh.highlight]usage:[/] /ensemble remove <model_id_or_index>")
            return None
        current = _ensemble_current_member_dicts(cfg)
        if not current:
            console.print("  [dim]ensemble is empty[/]")
            return None
        # Allow either a 1-based index or a model id substring.
        target_idx: int | None = None
        try:
            n = int(rest)
            if 1 <= n <= len(current):
                target_idx = n - 1
        except ValueError:
            for i, m in enumerate(current):
                if m["model"] == rest:
                    target_idx = i
                    break
            if target_idx is None:
                # Loose match on substring as last resort.
                matches = [i for i, m in enumerate(current) if rest in m["model"]]
                if len(matches) == 1:
                    target_idx = matches[0]
                elif len(matches) > 1:
                    console.print(
                        f"  [josh.error]ambiguous:[/] {rest!r} matches "
                        f"{[current[i]['model'] for i in matches]}"
                    )
                    return None
        if target_idx is None:
            console.print(f"  [josh.error]no match for[/] {rest!r}")
            return None
        removed = current.pop(target_idx)
        try:
            save_ensemble_members_override(current, cfg.agent.memory_dir)
        except Exception as e:  # noqa: BLE001
            console.print(f"  [josh.error]failed to save:[/] {e}")
            return None
        cfg.agent.ensemble = current
        console.print(
            f"  [josh.success]removed[/] [josh.info]{removed['model']}[/] "
            f"[dim]({len(current)} members)[/]"
        )
        return None

    if sub == "reset":
        try:
            existed = clear_ensemble_members_override(cfg.agent.memory_dir)
        except Exception as e:  # noqa: BLE001
            console.print(f"  [josh.error]failed to clear override:[/] {e}")
            return None
        cfg.agent.ensemble = []
        if existed:
            console.print("  [josh.success]reset[/] [dim](built-in defaults restored)[/]")
        else:
            console.print("  [dim]no override was set — already on defaults[/]")
        return None

    # `run <prompt>` — explicit one-shot.
    if sub == "run":
        if not rest:
            console.print("  [josh.highlight]usage:[/] /ensemble run <prompt>")
            return None
        await _run_ensemble_once(rest, cfg)
        return None

    # Backward-compat: anything else is treated as the prompt itself, so
    # `/ensemble hi` still runs a one-shot the way it always did.
    await _run_ensemble_once(arg, cfg)
    return None


VALID_PERMISSION_MODES = {"default", "acceptEdits", "plan", "bypassPermissions", "dontAsk", "auto"}


async def _set_perm(agent, mode: str) -> str | None:
    """Switch the agent's permission mode mid-session. Returns 'permission' on success."""
    if mode not in VALID_PERMISSION_MODES:
        console.print(
            f"  [josh.error]unknown mode '{mode}'.[/] "
            f"[dim]Valid: {', '.join(sorted(VALID_PERMISSION_MODES))}.[/]"
        )
        return None
    if not isinstance(agent, Agent):
        console.print("  [dim]permission modes only apply to the Claude backend.[/]")
        return None
    if agent._client is None:
        console.print("[josh.error]Agent not connected.[/]")
        return None
    try:
        await agent._client.set_permission_mode(mode)  # type: ignore[arg-type]
    except Exception as e:  # noqa: BLE001
        console.print(f"[josh.error]Failed to switch permission mode: {e}[/]")
        return None
    desc = {
        "default":           "prompts before destructive actions",
        "plan":              "plan mode — Claude proposes a plan before executing",
        "acceptEdits":       "auto-approve file edits",
        "bypassPermissions": "full trust — no prompts (use carefully)",
        "dontAsk":           "don't ask permission",
        "auto":              "auto-decide",
    }.get(mode, mode)
    console.print(f"  [josh.success]permission mode →[/] [josh.title]{mode}[/] [dim]({desc})[/]")
    return "permission"


async def _ctx_pct(agent) -> float | None:
    """Best-effort context-usage percentage. Returns None if unsupported."""
    if not isinstance(agent, Agent) or agent._client is None:
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


async def _get_context_usage(agent) -> str | None:
    if not isinstance(agent, Agent) or agent._client is None:
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
        console.print("[josh.error]Couldn't locate the joshv1 git checkout.[/]")
        return
    console.print(f"[dim]git pull in {repo}...[/]")
    proc = subprocess.run(
        ["git", "-C", str(repo), "pull"], capture_output=True, text=True, timeout=60
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    console.print(out.strip() or "(no output)")
    if proc.returncode == 0:
        console.print(
            "[josh.success]Update fetched.[/] [dim]Restart joshv1 chat to pick up changes.[/]"
        )
    else:
        console.print(f"[josh.error]git pull failed (exit {proc.returncode}).[/]")


def _find_repo_root() -> Path | None:
    here = Path(__file__).resolve()
    for ancestor in [here, *here.parents]:
        if (ancestor / ".git").is_dir():
            return ancestor
    return None


@main.command()
def update() -> None:
    """`git pull` the latest joshv1 from origin."""
    _run_update()


@main.command()
@click.argument("prompt", nargs=-1)
@click.option("--list", "list_members", is_flag=True,
              help="Print the current ensemble member list and exit.")
@click.pass_context
def ensemble(ctx: click.Context, prompt: tuple[str, ...], list_members: bool) -> None:
    """Fan out a prompt to multiple models in parallel; synthesize one answer."""
    from joshv1 import ensemble as ens_mod

    cfg = load_config(ctx.obj.get("config_path") if ctx.obj else None)

    if list_members:
        members = ens_mod.members_from_settings(cfg.agent)
        source = "config.yaml / memory override" if cfg.agent.ensemble else "built-in default"
        mode = "on" if cfg.agent.ensemble_mode else "off"
        console.print(
            f"\n[josh.title]Ensemble members[/]  [dim]({source}, mode={mode})[/]"
        )
        for m in members:
            console.print(f"  [josh.info]{m.provider:12s}[/]  {m.model:55s}  [dim]{m.role}[/]")
        console.print()
        return

    if prompt:
        message = " ".join(prompt)
    elif not sys.stdin.isatty():
        message = sys.stdin.read().strip()
    else:
        console.print("[josh.error]No prompt given.[/]")
        sys.exit(2)

    members = ens_mod.members_from_settings(cfg.agent)

    async def _go() -> None:
        console.print(
            f"  [dim]polling {len(members)} models in parallel...[/]"
        )

        def _on_draft(member, draft):
            if draft.success:
                console.print(f"  [josh.success]✓[/] [josh.info]{member.model}[/]")
            else:
                console.print(f"  [josh.error]✗[/] [josh.info]{member.model}[/]")
                console.print(f"      [dim]error: {draft.error!r}[/]")
                if draft.text:
                    console.print(f"      [dim]text:  {draft.text[:160]!r}[/]")

        answer, drafts = await ens_mod.run_ensemble(
            message, cfg.agent, members=members, on_draft_complete=_on_draft,
        )
        n_ok = sum(1 for d in drafts if d.success)
        console.print(f"\n[josh.title]Synthesized answer[/] [dim](from {n_ok}/{len(drafts)} drafts)[/]\n")
        console.print(answer)
        console.print()

    asyncio.run(_go())


@main.command("setup-provider")
@click.option("--provider", "provider_name",
              type=click.Choice(["claude", "openrouter", "ollama", "gemini"]),
              default=None,
              help="Skip the interactive picker.")
@click.option("--model", "model_name", default=None,
              help="Model to use. Defaults to a sensible per-provider pick.")
@click.option("--key", "api_key", default=None,
              help="API key for the provider (skips the prompt). Ignored for ollama.")
def setup_provider(provider_name: str | None, model_name: str | None, api_key: str | None) -> None:
    """Interactive setup: writes ~/.config/joshv1/config.yaml + .env for non-Claude providers."""
    from joshv1.providers import PROVIDER_PRESETS

    if provider_name is None:
        console.print("\n[josh.title]Pick a provider[/]")
        choices = ["claude"] + list(PROVIDER_PRESETS.keys())
        for i, name in enumerate(choices, 1):
            if name == "claude":
                label = "Claude (your Max subscription — full tool use)"
            else:
                label = PROVIDER_PRESETS[name].label
            console.print(f"  [josh.highlight]{i}[/]  {name:12s}  [dim]{label}[/]")
        console.print()
        try:
            sel = click.prompt("which", type=click.IntRange(1, len(choices)), default=1)
        except click.Abort:
            console.print("[dim]cancelled.[/]")
            return
        provider_name = choices[sel - 1]

    config_dir = Path.home() / ".config" / "joshv1"
    config_path = config_dir / "config.yaml"

    if provider_name == "claude":
        # Reset to defaults: just remove the provider line if it's there.
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path.write_text("agent:\n  provider: claude\n", encoding="utf-8")
        console.print(f"  [josh.success]wrote[/] [josh.info]{config_path}[/]")
        console.print("  [dim]restart joshv1 to use Claude.[/]")
        return

    preset = PROVIDER_PRESETS[provider_name]
    model = model_name or preset.default_model

    if preset.api_key_env and api_key is None:
        # Skip the prompt if the key is already in ~/.env (or the shell env)
        # — common case when changing just the model.
        existing = _read_env_value(Path.home() / ".env", preset.api_key_env) or os.environ.get(preset.api_key_env)
        if existing:
            console.print(f"  [dim]keeping existing[/] [josh.info]{preset.api_key_env}[/] [dim]from ~/.env[/]")
            api_key = existing
        else:
            console.print(
                f"\n[josh.title]API key[/]  [dim]({preset.api_key_env})[/]\n"
                f"  [dim]Get one at:[/] {_signup_url(provider_name)}"
            )
            try:
                api_key = click.prompt("paste your key", hide_input=True, default="", show_default=False)
            except click.Abort:
                console.print("[dim]cancelled.[/]")
                return
            api_key = api_key.strip()
            if not api_key:
                console.print(f"[josh.error]No key given. Aborting.[/]")
                return

    # Write config.yaml
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        f"agent:\n  provider: {provider_name}\n  model: {model}\n",
        encoding="utf-8",
    )
    console.print(f"  [josh.success]wrote[/] [josh.info]{config_path}[/]")

    # Append API key to ~/.env (or merge in-place if line already exists)
    if preset.api_key_env and api_key:
        env_path = Path.home() / ".env"
        _set_env_line(env_path, preset.api_key_env, api_key)
        console.print(f"  [josh.success]wrote[/] [josh.info]{env_path}[/] [dim](key set)[/]")

    if provider_name == "ollama":
        console.print(
            "  [dim]Make sure Ollama is running locally:[/] "
            "[josh.info]ollama serve[/] [dim]and[/] [josh.info]ollama pull " + model + "[/]"
        )

    console.print("\n  [josh.success]Done.[/] [dim]Restart joshv1 to use[/] "
                  f"[josh.info]{provider_name}[/] [dim]with[/] [josh.info]{model}[/]")


def _signup_url(provider_name: str) -> str:
    return {
        "openrouter": "https://openrouter.ai/keys",
        "gemini":     "https://aistudio.google.com/apikey",
    }.get(provider_name, "")


def _set_env_line(path: Path, key: str, value: str) -> None:
    """Replace KEY=... line in `path` if present, else append. Creates the file if missing."""
    lines: list[str] = []
    if path.is_file():
        lines = path.read_text(encoding="utf-8").splitlines()
    new_line = f"{key}={value}"
    found = False
    for i, line in enumerate(lines):
        if line.startswith(f"{key}="):
            lines[i] = new_line
            found = True
            break
    if not found:
        lines.append(new_line)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _first_nonempty_line(text: str) -> str:
    """First line of `text` that has visible content. Used for one-line error displays."""
    for line in text.splitlines():
        s = line.strip()
        if s:
            return s
    return ""


def _read_env_value(path: Path, key: str) -> str | None:
    """Return the value for KEY=... in the file, or None if not present."""
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(f"{key}="):
            return line.split("=", 1)[1].strip()
    return None


@main.command()
@click.option("--path", "repo_path", default=None,
              help="Repo to audit. Defaults to the joshv1 repo this binary lives in.")
@click.option("--save/--no-save", default=True,
              help="Write the report to ~/.josh-memory/audits/<timestamp>.md.")
@click.pass_context
def audit(ctx: click.Context, repo_path: str | None, save: bool) -> None:
    """Scan joshv1 (or another repo) and report concrete improvements."""
    from joshv1 import audit as audit_mod

    if repo_path:
        repo = Path(repo_path).expanduser().resolve()
    else:
        found = audit_mod.find_hermes_repo()
        if found is None:
            console.print("[josh.error]Couldn't find the joshv1 repo. Pass --path explicitly.[/]")
            sys.exit(2)
        repo = found

    if not repo.is_dir():
        console.print(f"[josh.error]Not a directory: {repo}[/]")
        sys.exit(2)

    cfg = load_config(ctx.obj.get("config_path") if ctx.obj else None)
    asyncio.run(_run_audit(cfg, repo, save))


async def _run_audit(cfg: Config, repo: Path, save: bool) -> None:
    from joshv1 import audit as audit_mod

    console.print(f"  [dim]auditing[/] [josh.info]{repo}[/]")
    transcript: list[str] = []
    async with build_backend(cfg.agent, cwd=repo) as agent:
        async for event in agent.run_stream(audit_mod.AUDIT_PROMPT):
            _render_event_plain(event)
            if isinstance(event, TextDelta):
                transcript.append(event.text)
    console.print()

    if save:
        path = audit_mod.next_report_path(cfg.agent.memory_dir)
        path.write_text("".join(transcript), encoding="utf-8")
        console.print(f"  [josh.success]report saved[/] [josh.info]{path}[/]")


def _print_theme_list() -> None:
    """Pretty list of all themes with inline color swatches."""
    current = themes.load_active_palette().name
    console.print(
        f"\n  [josh.section]Available themes[/]  "
        f"[dim](active:[/] [josh.session]{current}[/][dim])[/]\n"
    )
    for p in themes.list_themes():
        marker = "  [josh.session]← current[/]" if p.name == current else ""
        console.print(
            f"    [josh.info]{p.name:<18}[/]  "
            f"{themes.render_theme_swatch(p)}  "
            f"[dim]{p.label}[/]{marker}"
        )
    console.print(
        "\n  [dim]switch with:[/] [josh.info]joshv1 theme <name>[/] "
        "[dim]or[/] [josh.info]/theme <name>[/] [dim]in chat[/]\n"
    )


@main.command(name="theme")
@click.argument("name", required=False)
def theme_cmd(name: str | None) -> None:
    """List available color themes (with previews) or switch to one.

    Examples:
        joshv1 theme              # list all themes with swatches
        joshv1 theme list         # same — explicit verb
        joshv1 theme synthwave    # switch to synthwave
    """
    if not name or name == "list":
        _print_theme_list()
        return
    if name not in themes.PALETTES:
        console.print(f"  [josh.error]unknown theme:[/] [josh.info]{name}[/]")
        console.print(f"  [dim]available:[/] {', '.join(themes.PALETTES)}")
        sys.exit(1)
    try:
        themes.save_active_theme(name)
    except Exception as e:  # noqa: BLE001
        console.print(f"  [josh.error]failed to save theme:[/] {e}")
        sys.exit(1)
    console.print(
        f"  [josh.success]theme set to[/] [josh.session]{name}[/]  "
        f"{themes.render_theme_swatch(themes.get_palette(name))}\n"
        f"  [dim]Restart joshv1 to see it everywhere.[/]"
    )


# ---------------------------------------------------------------------------
# Daemon: long-running joshv1 you can attach/detach via tmux
# ---------------------------------------------------------------------------

TMUX_SESSION_NAME = "joshv1"


def _tmux_available() -> bool:
    return shutil.which("tmux") is not None


def _tmux_session_exists(name: str) -> bool:
    r = subprocess.run(
        ["tmux", "has-session", "-t", name],
        capture_output=True,
    )
    return r.returncode == 0


def _joshv1_command() -> list[str]:
    """Absolute command to launch the joshv1 CLI inside the daemon session."""
    exe = shutil.which("joshv1")
    if exe:
        return [exe]
    return [sys.executable, "-m", "joshv1"]


@main.command(name="daemon")
def daemon_cmd() -> None:
    """Start joshv1 chat as a persistent tmux session you can attach/detach.

    If a session already exists, attaches to it instead of starting a new one.
    Detach with Ctrl-B then D — the agent keeps running. Re-attach later with
    `joshv1 connect`.
    """
    if not _tmux_available():
        console.print(
            "  [josh.error]tmux not installed.[/] "
            "[dim]Run:[/] [josh.info]sudo apt install -y tmux[/]"
        )
        sys.exit(1)
    if _tmux_session_exists(TMUX_SESSION_NAME):
        console.print(
            f"  [dim]session '{TMUX_SESSION_NAME}' already running; attaching...[/]"
        )
        os.execvp("tmux", ["tmux", "attach", "-t", TMUX_SESSION_NAME])
    cmd = _joshv1_command()
    console.print(
        f"  [josh.success]starting daemon session[/] [josh.info]'{TMUX_SESSION_NAME}'[/]  "
        f"[dim](Ctrl-B then D to detach; reconnect with `joshv1 connect`)[/]"
    )
    # `-s` names the session, then the rest is the command to run inside it.
    os.execvp("tmux", ["tmux", "new-session", "-s", TMUX_SESSION_NAME, *cmd])


@main.command(name="connect")
def connect_cmd() -> None:
    """Attach to a running joshv1 daemon. Ctrl-B then D to detach."""
    if not _tmux_available():
        console.print("  [josh.error]tmux not installed.[/]")
        sys.exit(1)
    if not _tmux_session_exists(TMUX_SESSION_NAME):
        console.print(
            "  [josh.error]no daemon running.[/] "
            "[dim]Start one with:[/] [josh.info]joshv1 daemon[/]"
        )
        sys.exit(1)
    os.execvp("tmux", ["tmux", "attach", "-t", TMUX_SESSION_NAME])


@main.command(name="daemon-stop")
def daemon_stop_cmd() -> None:
    """Kill the running joshv1 daemon tmux session."""
    if not _tmux_available() or not _tmux_session_exists(TMUX_SESSION_NAME):
        console.print("  [dim]no daemon running.[/]")
        return
    subprocess.run(["tmux", "kill-session", "-t", TMUX_SESSION_NAME])
    console.print(f"  [josh.success]stopped daemon session[/] [josh.info]'{TMUX_SESSION_NAME}'[/]")


@main.command(name="daemon-status")
def daemon_status_cmd() -> None:
    """Report whether the joshv1 daemon is running."""
    if not _tmux_available():
        console.print("  [josh.highlight]tmux not installed[/] [dim](needed for daemon mode)[/]")
        return
    if _tmux_session_exists(TMUX_SESSION_NAME):
        console.print(
            f"  [josh.success]running[/] [dim]as tmux session[/] [josh.info]'{TMUX_SESSION_NAME}'[/]\n"
            f"  [dim]attach:[/] [josh.info]joshv1 connect[/]\n"
            f"  [dim]kill:  [/] [josh.info]joshv1 daemon-stop[/]"
        )
    else:
        console.print(
            "  [dim]not running.[/] "
            "[dim]Start:[/] [josh.info]joshv1 daemon[/]"
        )


# ---------------------------------------------------------------------------
# `joshv1 plugin ...` — thin wrapper around `claude plugin` subcommand
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
        console.print("[josh.error]claude CLI not found. Install Claude Code first.[/]")
        return
    try:
        proc = subprocess.run(
            ["claude", "plugin", *args],
            capture_output=True, text=True, timeout=180,
        )
    except subprocess.TimeoutExpired:
        console.print("[josh.error]claude plugin timed out.[/]")
        return
    if proc.stdout:
        console.print(proc.stdout.rstrip())
    if proc.returncode != 0:
        msg = (proc.stderr or "(no stderr)").rstrip()
        console.print(f"[josh.error]exit {proc.returncode}:[/] {msg}")
        return
    console.print(
        "[dim](plugin changes take effect on /new in chat or on next `joshv1` launch)[/]"
    )


@main.group(name="plugin")
def plugin_group() -> None:
    """Install and manage Claude Code plugins.

    Thin wrapper around `claude plugin`. Use plugin@marketplace for a specific
    marketplace, e.g. `joshv1 plugin install github@claude-plugins-official`.
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
    from joshv1 import skills as skillsmod
    entries = skillsmod.list_installed()
    if not entries:
        console.print("[dim]No skills installed.[/]")
        return
    for e in entries:
        line = f"  [josh.info]{e.name:<24}[/]"
        if e.description:
            line += f" [dim]{e.description}[/]"
        if e.author:
            line += f" [dim](by {e.author})[/]"
        console.print(line)


@skill_group.command(name="browse")
def skill_browse() -> None:
    """Show all skills in the curated marketplace."""
    from joshv1 import skills as skillsmod
    items = skillsmod.browse()
    if not items:
        console.print("[dim]Marketplace is empty.[/]")
        return
    for item in items:
        console.print(f"  [josh.title]{item.get('name')}[/]  [dim]{item.get('description', '')}[/]")
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
    from joshv1 import skills as skillsmod
    q = " ".join(query)
    matches = skillsmod.search(q)
    if not matches:
        console.print(f"[dim]No marketplace matches for '{q}'.[/]")
        return
    for item in matches:
        console.print(f"  [josh.info]{item.get('name')}[/]  [dim]{item.get('description', '')}[/]")


@skill_group.command(name="inspect")
@click.argument("name")
def skill_inspect(name: str) -> None:
    """Print a preview of an installed skill's SKILL.md."""
    from joshv1 import skills as skillsmod
    text = skillsmod.inspect_skill(name)
    if text is None:
        console.print(f"[josh.error]not installed:[/] {name}")
        sys.exit(1)
    entry = skillsmod.get_installed(name)
    if entry:
        console.print(f"[josh.title]{entry.name}[/]  [dim]{entry.description}[/]")
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
    from joshv1 import skills as skillsmod
    try:
        entry = skillsmod.install(identifier, name=name)
    except skillsmod.SkillError as e:
        console.print(f"[josh.error]install failed:[/] {e}")
        sys.exit(1)
    console.print(f"[josh.success]installed[/] [josh.info]{entry.name}[/] → [dim]{entry.installed_at}[/]")
    if entry.description:
        console.print(f"[dim]{entry.description}[/]")


@skill_group.command(name="uninstall")
@click.argument("name")
@click.option("--yes", is_flag=True, help="Skip the confirmation prompt.")
def skill_uninstall(name: str, yes: bool) -> None:
    """Remove an installed skill."""
    from joshv1 import skills as skillsmod
    entry = skillsmod.get_installed(name)
    if entry is None:
        console.print(f"[josh.error]not installed:[/] {name}")
        sys.exit(1)
    if not yes:
        console.print(f"  [josh.highlight]about to delete:[/] [josh.info]{entry.installed_at}[/]")
        if not click.confirm("  proceed?", default=False):
            console.print("[dim]cancelled.[/]")
            return
    try:
        skillsmod.uninstall(name)
    except skillsmod.SkillError as e:
        console.print(f"[josh.error]uninstall failed:[/] {e}")
        sys.exit(1)
    console.print(f"[josh.success]uninstalled[/] {name}")


@skill_group.command(name="snapshot")
@click.argument("direction", type=click.Choice(["export", "import"]))
@click.argument("path", type=click.Path())
def skill_snapshot(direction: str, path: str) -> None:
    """Export current skills to JSON, or import a snapshot back."""
    import json

    from joshv1 import skills as skillsmod
    p = Path(path)
    if direction == "export":
        p.write_text(json.dumps(skillsmod.snapshot_export(), indent=2))
        console.print(f"[josh.success]wrote snapshot →[/] [josh.info]{p}[/]")
    else:
        try:
            data = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError) as e:
            console.print(f"[josh.error]bad snapshot file:[/] {e}")
            sys.exit(1)
        installed = skillsmod.snapshot_import(data)
        if installed:
            console.print(f"[josh.success]installed {len(installed)} new skills:[/] {', '.join(installed)}")
        else:
            console.print("[dim]nothing new to install (all already present).[/]")


@main.command(name="index")
@click.argument("directory", type=click.Path(exists=True, file_okay=False))
@click.option("--rebuild", is_flag=True, help="Wipe the index before re-walking.")
def index_cmd(directory: str, rebuild: bool) -> None:
    """Index a directory of text/markdown files for `joshv1 search`."""
    from joshv1.search import index_directory
    n = index_directory(directory, rebuild=rebuild)
    console.print(f"[josh.success]Indexed {n} files[/] from [josh.info]{directory}[/]")


@main.command(name="search")
@click.argument("query", nargs=-1, required=True)
@click.option("--limit", default=10, help="Max hits to return.")
@click.option("--json", "as_json", is_flag=True, help="Machine-readable output.")
def search_cmd(query: tuple[str, ...], limit: int, as_json: bool) -> None:
    """Full-text search over your indexed notes (FTS5)."""
    from joshv1.search import render_hits, render_hits_json, search
    q = " ".join(query)
    try:
        hits = search(q, limit=limit)
    except ValueError as e:
        console.print(f"[josh.error]{e}[/]")
        sys.exit(1)
    if as_json:
        print(render_hits_json(hits))
    else:
        console.print(render_hits(hits))


@main.command(name="voice")
@click.argument("audio_file", type=click.Path(exists=True, dir_okay=False))
@click.option("--model", default="small", help="Whisper model: tiny, base, small, medium, large-v3.")
@click.option("--language", default=None, help="ISO language code; auto-detect if omitted.")
@click.option("--run", "auto_run", is_flag=True, help="Pipe the transcript into `joshv1 run`.")
def voice_cmd(audio_file: str, model: str, language: str | None, auto_run: bool) -> None:
    """Transcribe an audio file using local Whisper, optionally run it as a prompt."""
    from joshv1.voice import transcribe
    try:
        console.print(f"[dim]transcribing {audio_file} with whisper-{model}...[/]")
        text = transcribe(audio_file, model_name=model, language=language)
    except RuntimeError as e:
        console.print(f"[josh.error]{e}[/]")
        sys.exit(1)

    console.print(f"\n[josh.title]Transcript:[/] {text}\n")
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
            console.print(f"[josh.success]Deleted session {delete_id}.[/]")
        except Exception as e:  # noqa: BLE001
            console.print(f"[josh.error]Delete failed: {e}[/]")
        return

    try:
        sessions = list_sessions(limit=limit)
    except Exception as e:  # noqa: BLE001
        console.print(f"[josh.error]Failed to list sessions: {e}[/]")
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
                "set" if slack_set else "unset — only needed for `joshv1 slack`",
            )
        )
        discord_set = bool(cfg.discord.get("bot_token"))
        checks.append(
            (
                "Discord token (optional)",
                discord_set,
                "set" if discord_set else "unset — only needed for `joshv1 discord`",
            )
        )
        # Optional /maps connectivity probes
        try:
            from joshv1 import maps as mapsmod
            nominatim_ok = mapsmod.can_reach("https://nominatim.openstreetmap.org/")
            checks.append((
                "Nominatim reachable (optional)",
                nominatim_ok,
                "ok — /maps geocoding will work" if nominatim_ok
                else "blocked — /maps will fall back to Photon (or agent lookup)",
            ))
            tile_ok = mapsmod.can_reach("https://tile.openstreetmap.org/0/0/0.png")
            checks.append((
                "OSM tiles reachable (optional)",
                tile_ok,
                "ok — /maps will render a braille map" if tile_ok
                else "blocked — /maps will show coords only",
            ))
        except Exception as e:  # noqa: BLE001
            checks.append((
                "Maps connectivity (optional)",
                False,
                f"probe failed: {type(e).__name__}",
            ))
    except Exception as e:  # noqa: BLE001
        checks.append(("Config loaded", False, f"{type(e).__name__}: {e}"))

    required_ok = True
    for name, ok, msg in checks:
        sym = "[josh.success]OK[/]" if ok else "[josh.error]X[/]"
        console.print(f"  {sym}  [bold]{name}[/]  [dim]{msg}[/]")
        if not ok and "optional" not in name:
            required_ok = False

    if required_ok:
        console.print(
            "\n[josh.section]All required checks passed.[/] Just type [josh.info]joshv1[/]."
        )
    else:
        console.print(
            "\n[josh.error.bold]Some required checks failed.[/] Common fixes:\n"
            "  - Install Claude Code: [josh.info]npm install -g @anthropic-ai/claude-code[/]\n"
            "  - Log in to your Max account: [josh.info]claude login[/]"
        )
        sys.exit(1)


@main.command()
@click.pass_context
def slack(ctx: click.Context) -> None:
    """Run the Slack bot (Socket Mode)."""
    from joshv1.platforms.slack import run_slack_bot

    cfg = load_config(ctx.obj.get("config_path") if ctx.obj else None)
    asyncio.run(run_slack_bot(cfg))


@main.command()
@click.pass_context
def discord(ctx: click.Context) -> None:
    """Run the Discord bot."""
    from joshv1.platforms.discord import run_discord_bot

    cfg = load_config(ctx.obj.get("config_path") if ctx.obj else None)
    asyncio.run(run_discord_bot(cfg))
