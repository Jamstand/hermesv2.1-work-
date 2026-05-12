"""Core agent loop on top of the Claude Agent SDK.

Authenticates via your local `claude` CLI installation (OAuth) instead of an
API key — so billing goes through your Claude Max subscription, not the
Anthropic API.

The Agent SDK runs the agent loop inside the local `claude` subprocess. We
just forward user messages, stream back blocks/events, and yield them as
typed events the CLI / Slack / Discord layers render uniformly.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)

from hermesv2.config import AgentSettings

DEFAULT_TOOLS = [
    # Files
    "Read",
    "Write",
    "Edit",
    "NotebookEdit",
    # Search
    "Grep",
    "Glob",
    # Shell
    "Bash",
    "BashOutput",
    "KillShell",
    # Web
    "WebFetch",
    "WebSearch",
    # Agents
    "Task",        # spawn sub-agents
    "TodoWrite",   # internal task tracking for multi-step work
    # Skills + Plans
    "Skill",       # invoke skills from ~/.claude/skills
    "ExitPlanMode",
    "SlashCommand",
]


# --- Events ------------------------------------------------------------------


@dataclass
class TextDelta:
    text: str


@dataclass
class ThinkingDelta:
    text: str


@dataclass
class ToolCall:
    name: str
    input: dict[str, Any]


@dataclass
class ToolResult:
    name: str
    output: str
    is_error: bool


@dataclass
class TurnDone:
    stop_reason: str
    final_text: str
    cost_usd: float | None
    usage: dict[str, Any]


Event = TextDelta | ThinkingDelta | ToolCall | ToolResult | TurnDone


# --- The agent ---------------------------------------------------------------


class Agent:
    """Long-lived agent backed by a single ClaudeSDKClient.

    Multiple conversation threads are multiplexed via the `session_id` argument
    to `run` / `run_stream`. The Slack and Discord adapters use this to keep
    per-user history without spawning a new subprocess per user.
    """

    def __init__(
        self,
        settings: AgentSettings,
        cwd: str | Path | None = None,
        resume_session_id: str | None = None,
    ):
        self.settings = settings
        self.cwd = Path(cwd).expanduser() if cwd else None
        self.resume_session_id = resume_session_id
        self._client: ClaudeSDKClient | None = None
        self._reset_counter = 0

    def _build_options(self) -> ClaudeAgentOptions:
        if self.settings.thinking == "adaptive":
            thinking: dict[str, Any] = {
                "type": "adaptive",
                "display": self.settings.thinking_display,
            }
        else:
            thinking = {"type": "disabled"}

        kwargs: dict[str, Any] = dict(
            system_prompt=self.settings.system_prompt,
            allowed_tools=DEFAULT_TOOLS,
            permission_mode=self.settings.permission_mode,
            cwd=str(self.cwd) if self.cwd else None,
            model=self.settings.model,
            effort=self.settings.effort,
            thinking=thinking,
            include_partial_messages=True,
        )
        if self.resume_session_id:
            kwargs["resume"] = self.resume_session_id
        if self.settings.mcp_servers:
            kwargs["mcp_servers"] = self.settings.mcp_servers
        if self.settings.skills:
            kwargs["skills"] = self.settings.skills
        if self.settings.add_dirs:
            kwargs["add_dirs"] = self.settings.add_dirs

        return ClaudeAgentOptions(**kwargs)

    async def connect(self) -> None:
        if self.cwd:
            self.cwd.mkdir(parents=True, exist_ok=True)
        # We auth via the local `claude` CLI's OAuth (Max). If a stray
        # ANTHROPIC_API_KEY is set in the env (e.g. leftover from .env), the
        # subprocess would try to use it and fail with "Invalid API key".
        os.environ.pop("ANTHROPIC_API_KEY", None)
        self._client = ClaudeSDKClient(options=self._build_options())
        await self._client.connect()

    async def disconnect(self) -> None:
        if self._client is not None:
            await self._client.disconnect()
            self._client = None

    async def __aenter__(self) -> Agent:
        await self.connect()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.disconnect()

    def reset_session(self, base: str = "default") -> str:
        """Bump the session counter so the next call starts a fresh history."""
        self._reset_counter += 1
        return f"{base}-{self._reset_counter}"

    async def run(self, user_message: str, session_id: str = "default") -> str:
        parts: list[str] = []
        async for ev in self.run_stream(user_message, session_id=session_id):
            if isinstance(ev, TurnDone):
                return ev.final_text
            if isinstance(ev, TextDelta):
                parts.append(ev.text)
        return "".join(parts)

    async def run_stream(
        self, user_message: str, session_id: str = "default"
    ) -> AsyncIterator[Event]:
        if self._client is None:
            raise RuntimeError(
                "Agent not connected. Use `async with Agent(...) as agent:` "
                "or call `await agent.connect()` first."
            )

        await self._client.query(user_message, session_id=session_id)

        final_text_parts: list[str] = []
        tool_use_names: dict[str, str] = {}  # tool_use_id -> name

        async for msg in self._client.receive_response():
            if isinstance(msg, AssistantMessage):
                async for ev in self._handle_assistant(msg, final_text_parts, tool_use_names):
                    yield ev
            elif isinstance(msg, ResultMessage):
                yield TurnDone(
                    stop_reason=msg.stop_reason or "end_turn",
                    final_text="".join(final_text_parts),
                    cost_usd=msg.total_cost_usd,
                    usage=msg.usage or {},
                )
                return
            elif isinstance(msg, SystemMessage):
                continue
            # StreamEvent / RateLimitEvent: ignore for now; could surface later.

    async def _handle_assistant(
        self,
        msg: AssistantMessage,
        final_text_parts: list[str],
        tool_use_names: dict[str, str],
    ) -> AsyncIterator[Event]:
        for block in msg.content:
            if isinstance(block, TextBlock):
                final_text_parts.append(block.text)
                yield TextDelta(block.text)
            elif isinstance(block, ThinkingBlock):
                yield ThinkingDelta(block.thinking)
            elif isinstance(block, ToolUseBlock):
                tool_use_names[block.id] = block.name
                yield ToolCall(name=block.name, input=block.input or {})
            elif isinstance(block, ToolResultBlock):
                yield ToolResult(
                    name=tool_use_names.get(block.tool_use_id, block.tool_use_id),
                    output=_stringify_result(block.content),
                    is_error=bool(block.is_error),
                )


def _stringify_result(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)
