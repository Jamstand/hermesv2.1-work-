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
searching/fetching the web, and editing text. Use them to actually do work
rather than describing what to do.

Operating principles:
  - Be concise. Skip filler like "Sure! I'd be happy to help...".
  - For destructive actions (delete files, push commits, send messages),
    confirm intent in plain language before acting.
  - Cite sources (URLs) when answering factual questions from the web.
  - If a task is ambiguous, ask one clear question rather than guessing."""


@dataclass
class AgentSettings:
    model: str = "claude-opus-4-7"
    effort: str = "high"                    # low | medium | high | xhigh | max
    thinking: str = "adaptive"              # adaptive | disabled
    thinking_display: str = "summarized"    # summarized | omitted (Opus 4.7 default is omitted)
    permission_mode: str = "default"        # default | acceptEdits | plan | bypassPermissions
    workspace_dir: str = "~/hermes-workspace"
    system_prompt: str = DEFAULT_SYSTEM_PROMPT


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
