"""Config: YAML file merged with environment variables.

The config file (`config.yaml` next to the cwd, by default) defines defaults;
secrets and host-specific settings come from environment variables (or a `.env`
file loaded by `python-dotenv`).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

DEFAULT_SYSTEM_PROMPT = """You are Hermes v2, a personal AI agent helping with work tasks.

Capabilities you have via tools:
  - Read, write, list, and search files on the local machine.
  - Run shell commands carefully.
  - Fetch URLs and search the web.
  - Send email and read the inbox (when SMTP/IMAP are configured).

Operating principles:
  - Be concise. Skip filler ("Sure! I'd be happy to help...").
  - Use tools to actually do work, don't just describe what to do.
  - For destructive actions (delete files, send email, push commits),
    confirm intent in plain language before acting.
  - Cite sources (URLs) when answering factual questions from the web."""


@dataclass
class AgentSettings:
    model: str = "claude-opus-4-7"
    effort: str = "high"
    thinking: str = "adaptive"          # "adaptive" or "disabled"
    thinking_display: str = "summarized"  # "summarized" or "omitted"
    max_tokens: int = 16000
    system_prompt: str = DEFAULT_SYSTEM_PROMPT


@dataclass
class ShellToolSettings:
    enabled: bool = True
    timeout_seconds: int = 60
    blocked_patterns: list[str] = field(default_factory=list)


@dataclass
class FilesToolSettings:
    enabled: bool = True
    allowed_roots: list[str] = field(default_factory=list)


@dataclass
class ToolSettings:
    shell: ShellToolSettings = field(default_factory=ShellToolSettings)
    files: FilesToolSettings = field(default_factory=FilesToolSettings)
    web: bool = True
    email: bool = False


@dataclass
class Config:
    agent: AgentSettings = field(default_factory=AgentSettings)
    tools: ToolSettings = field(default_factory=ToolSettings)
    # Secrets pulled from environment, not file
    anthropic_api_key: str | None = None
    smtp: dict[str, str] = field(default_factory=dict)
    imap: dict[str, str] = field(default_factory=dict)
    slack: dict[str, str] = field(default_factory=dict)
    discord: dict[str, str] = field(default_factory=dict)


def _merge(data: dict[str, Any] | None, defaults: Any) -> Any:
    """Apply YAML overrides on top of a dataclass instance."""
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
    """Load config from YAML + env. Missing file is OK — defaults are used."""
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
            cfg.tools = _merge(data.get("tools"), cfg.tools)
            break

    cfg.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
    cfg.smtp = {
        "host": os.environ.get("SMTP_HOST", ""),
        "port": os.environ.get("SMTP_PORT", "587"),
        "user": os.environ.get("SMTP_USER", ""),
        "password": os.environ.get("SMTP_PASSWORD", ""),
        "from": os.environ.get("SMTP_FROM", ""),
    }
    cfg.imap = {
        "host": os.environ.get("IMAP_HOST", ""),
        "port": os.environ.get("IMAP_PORT", "993"),
        "user": os.environ.get("IMAP_USER", ""),
        "password": os.environ.get("IMAP_PASSWORD", ""),
    }
    cfg.slack = {
        "bot_token": os.environ.get("SLACK_BOT_TOKEN", ""),
        "app_token": os.environ.get("SLACK_APP_TOKEN", ""),
        "signing_secret": os.environ.get("SLACK_SIGNING_SECRET", ""),
    }
    cfg.discord = {"bot_token": os.environ.get("DISCORD_BOT_TOKEN", "")}

    return cfg
