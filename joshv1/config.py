"""Config: YAML file merged with environment variables.

The agent runs on top of Claude Code (via the Claude Agent SDK), which
authenticates against your local `claude` CLI install — no API key needed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

VALID_EFFORTS: tuple[str, ...] = ("low", "medium", "high", "xhigh", "max")

EFFORT_FILE = "effort"  # under ~/.josh-memory/, set via /effort


def save_active_effort(effort: str, memory_dir_path: str | os.PathLike[str]) -> None:
    """Persist the effort selection so it survives restarts."""
    from joshv1.memory import memory_dir as _mem_dir
    mem_dir = _mem_dir(memory_dir_path)
    mem_dir.mkdir(parents=True, exist_ok=True)
    (mem_dir / EFFORT_FILE).write_text(effort + "\n")


def load_active_effort(memory_dir_path: str | os.PathLike[str]) -> str | None:
    from joshv1.memory import memory_dir as _mem_dir
    f = _mem_dir(memory_dir_path) / EFFORT_FILE
    if not f.is_file():
        return None
    val = f.read_text().strip()
    return val if val in VALID_EFFORTS else None


DEFAULT_SYSTEM_PROMPT = """You are Josh v1, a personal AI agent helping with work tasks.

You have built-in tools for: reading and writing files, running shell commands,
searching/fetching the web, editing text, and spawning sub-agents.

Operating principles:
  - Be concise. Skip filler like "Sure! I'd be happy to help...".
  - For destructive actions (delete files, push commits, send messages),
    confirm intent in plain language before acting.
  - Cite sources (URLs) when answering factual questions from the web.
  - If a task is ambiguous, ask one clear question rather than guessing.

Tool-call discipline (this is strict — the user is watching the trace):
  - Make the FEWEST tool calls that answer the question. One targeted
    search beats five shotgun searches. After 2 web fetches or 2 shell
    commands, stop and answer with what you have — don't keep cross-
    checking just to be thorough.
  - Pick ONE tool per goal. Don't WebFetch a page AND ToolSearch AND
    Bash `gh` for the same lookup. If the first tool answers, you're
    done.
  - Do NOT narrate intent before a tool call. The tool call itself is
    the announcement. No "I'll search…" / "Let me check…" / "I need to
    look through…" preambles. Speak only after results, and only if
    the user needs explanation beyond the tool output itself.
  - If a search returns nothing useful, say so in one short sentence
    and answer with what's known. Do not chain more searches hoping
    something turns up.

Skill use — hard rules:
  - NEVER load `superpowers:brainstorming`, `superpowers:planning`, or any
    other brainstorming / planning / "design first" skill. Not for vague
    prompts, not for "creative building tasks", not for open-ended product
    requests — never. You already know how to ask clarifying questions and
    sketch designs. Use the `AskUserQuestion` tool directly; do NOT wrap
    that in a skill invocation first.
  - Skills are for narrow technical capabilities (e.g. a specific API
    wrapper). If you can't name a concrete capability the skill provides
    that you don't already have, don't load it.

When the user asks you to build, design, or create something (whether
vague like "make a game" or specific like "build me a subscription
tracker"):
  - Do NOT run shell commands to inspect the workspace before asking. The
    workspace is irrelevant until you know what to build. Exception: when
    the user's message explicitly references existing files or repo state
    ("what's in this repo?", "fix the bug in foo.py").
  - Do NOT write a preamble paragraph about scope, capabilities, or
    "let me ask a few questions first to get the shape right". No
    meta-talk. No "happy to help" openers.
  - Open IMMEDIATELY with one `AskUserQuestion` call: a single focused
    question with 3-4 concrete option choices.
  - Keep any text output BEFORE the question to one sentence at most
    (often zero — just call the tool).

Formatting (your output is rendered as Markdown in a styled terminal):
  - When introducing or naming a concept, feature, or product, wrap it in
    **bold** at first mention so it pops mid-sentence. Examples:
      "**Persistent memory** across sessions. A ~/.josh-memory/ ..."
      "**WebSearch** uses Anthropic's server-side search ..."
  - Use **bold** for specific factual values too: names, addresses,
    prices, dates, ratings, key numbers.
  - Use `inline code` for paths, IDs, commands, env vars, tokens.
  - Use ## subheaders when a response has multiple distinct sections.
  - Use bullet lists for 3+ parallel items.
  - Keep paragraphs short — one idea per paragraph.

Persistent memory:
  - A <memory> block is appended below with your notes about this user.
  - Only write to ~/.josh-memory/ when the user EXPLICITLY asks you to
    remember / save / note / record something ("remember that…", "save
    this", "note that I prefer…"). Do NOT auto-update memory just
    because something looks durable or "worth remembering" — that
    behavior creates noise the user has to clean up.
  - When writing, keep entries short and factual. Read the file first if
    it exists, then Edit (don't Write over the whole file).
  - Do not announce that you're saving — just do it. The next session
    will see the update automatically."""


@dataclass
class AgentSettings:
    # "claude" (default, full tool-use via Agent SDK) or "openrouter" / "ollama"
    # / "gemini" (OpenAI-compatible backends, text-only in phase 1).
    provider: str = "claude"
    model: str = "claude-opus-4-7"
    effort: str = "high"                    # low | medium | high | xhigh | max
    thinking: str = "adaptive"              # adaptive | disabled
    thinking_display: str = "summarized"    # summarized | omitted (Opus 4.7 default is omitted)
    permission_mode: str = "default"        # default | acceptEdits | plan | bypassPermissions
    workspace_dir: str = "~/josh-workspace"
    memory_dir: str = "~/.josh-memory"    # persistent USER.md + other notes; auto-loaded into system prompt
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    # MCP servers: name → {type: stdio|sse|http, command/url, args, env}
    mcp_servers: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Skills: "all" loads everything in ~/.claude/skills/, or a list of names
    skills: list[str] | str = "all"
    # Extra directories the agent's Read/Write/Bash tools can touch outside cwd
    add_dirs: list[str] = field(default_factory=list)
    # Ensemble draft pool for `/ensemble` and `joshv1 ensemble`. Each entry is
    # {provider, model, role}. Empty list = use the hardcoded default in
    # joshv1/ensemble.py. OpenRouter free model IDs churn frequently, so
    # configuring this in YAML lets you swap models without a code change.
    ensemble: list[dict[str, str]] = field(default_factory=list)


@dataclass
class Config:
    agent: AgentSettings = field(default_factory=AgentSettings)
    slack: dict[str, str] = field(default_factory=dict)
    discord: dict[str, str] = field(default_factory=dict)


def _merge(data: dict[str, Any] | None, defaults: Any) -> Any:
    if not data:
        return defaults
    for key, val in data.items():
        if hasattr(defaults, key):
            current = getattr(defaults, key)
            if hasattr(current, "__dataclass_fields__"):
                setattr(defaults, key, _merge(val, current))
            else:
                setattr(defaults, key, val)
    return defaults


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    # Walk-up search first (cwd → parents). `joshv1 setup-provider` writes
    # to ~/.env, so also explicitly load that as a fallback so the API keys
    # are picked up no matter where joshv1 is launched from. Existing env
    # vars take precedence over both.
    load_dotenv()
    home_env = Path.home() / ".env"
    if home_env.is_file():
        load_dotenv(home_env, override=False)

    cfg = Config()

    candidates: list[Path] = []
    if path:
        candidates.append(Path(path))
    else:
        candidates.extend(
            [
                Path("config.local.yaml"),
                Path("config.yaml"),
                Path.home() / ".config" / "joshv1" / "config.yaml",
            ]
        )

    for candidate in candidates:
        if candidate.is_file():
            with candidate.open() as f:
                data = yaml.safe_load(f) or {}
            cfg.agent = _merge(data.get("agent"), cfg.agent)
            break

    cfg.slack = {
        "bot_token": os.environ.get("SLACK_BOT_TOKEN", ""),
        "app_token": os.environ.get("SLACK_APP_TOKEN", ""),
        "signing_secret": os.environ.get("SLACK_SIGNING_SECRET", ""),
    }
    cfg.discord = {"bot_token": os.environ.get("DISCORD_BOT_TOKEN", "")}

    saved_effort = load_active_effort(cfg.agent.memory_dir)
    if saved_effort:
        cfg.agent.effort = saved_effort

    return cfg
