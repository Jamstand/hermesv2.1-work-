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

DEFAULT_SYSTEM_PROMPT = """You are Hermes v2, a personal AI agent helping with work tasks.

You have built-in tools for: reading and writing files, running shell commands,
searching/fetching the web, editing text, and spawning sub-agents.

Operating principles:
  - Be concise. Skip filler like "Sure! I'd be happy to help...".
  - For destructive actions (delete files, push commits, send messages),
    confirm intent in plain language before acting.
  - Cite sources (URLs) when answering factual questions from the web.
  - If a task is ambiguous, ask one clear question rather than guessing.

Formatting (your output is rendered as Markdown in a styled terminal):
  - When introducing or naming a concept, feature, or product, wrap it in
    **bold** at first mention so it pops mid-sentence. Examples:
      "**Persistent memory** across sessions. A ~/.hermes-memory/ ..."
      "**WebSearch** uses Anthropic's server-side search ..."
  - Use **bold** for specific factual values too: names, addresses,
    prices, dates, ratings, key numbers.
  - Use `inline code` for paths, IDs, commands, env vars, tokens.
  - Use ## subheaders when a response has multiple distinct sections.
  - Use bullet lists for 3+ parallel items.
  - Keep paragraphs short — one idea per paragraph.

Persistent memory:
  - A <memory> block is appended below with your notes about this user.
  - When the user reveals a durable preference, ongoing project, decision,
    or naming convention worth remembering, update the relevant file under
    ~/.hermes-memory/ via the Write tool. Keep entries short and factual.
  - Do not announce that you're "saving to memory" — just do it inline when
    it's relevant. The next session will see the update automatically."""


@dataclass
class AgentSettings:
    model: str = "claude-opus-4-7"
    effort: str = "high"                    # low | medium | high | xhigh | max
    thinking: str = "adaptive"              # adaptive | disabled
    thinking_display: str = "summarized"    # summarized | omitted (Opus 4.7 default is omitted)
    permission_mode: str = "default"        # default | acceptEdits | plan | bypassPermissions
    workspace_dir: str = "~/hermes-workspace"
    memory_dir: str = "~/.hermes-memory"    # persistent USER.md + other notes; auto-loaded into system prompt
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    # MCP servers: name → {type: stdio|sse|http, command/url, args, env}
    mcp_servers: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Skills: "all" loads everything in ~/.claude/skills/, or a list of names
    skills: list[str] | str = "all"
    # Extra directories the agent's Read/Write/Bash tools can touch outside cwd
    add_dirs: list[str] = field(default_factory=list)


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
    load_dotenv()

    cfg = Config()

    candidates: list[Path] = []
    if path:
        candidates.append(Path(path))
    else:
        candidates.extend(
            [
                Path("config.local.yaml"),
                Path("config.yaml"),
                Path.home() / ".config" / "hermesv2" / "config.yaml",
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

    return cfg
