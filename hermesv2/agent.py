"""Core agent loop on top of the Anthropic SDK.

Wraps `client.messages.stream()` in a tool-use loop. Emits typed events so the
CLI / Slack / Discord layers can render progress without each one re-implementing
the loop. Uses adaptive thinking and caches the system prompt + tool list across
turns.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

import anthropic

from hermesv2.config import AgentSettings

# --- Tool plumbing -----------------------------------------------------------


@dataclass
class Tool:
    """Custom (client-side) tool. Handler runs on the user's machine."""

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], str]

    def to_api(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


# --- Agent events ------------------------------------------------------------


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
    usage: dict[str, int]


Event = TextDelta | ThinkingDelta | ToolCall | ToolResult | TurnDone


@dataclass
class AgentConfig:
    """Convenience alias so external callers can `from hermesv2 import AgentConfig`."""

    settings: AgentSettings
    tools: list[Tool] = field(default_factory=list)
    server_tools: list[dict[str, Any]] = field(default_factory=list)


# --- The agent ---------------------------------------------------------------


class Agent:
    def __init__(
        self,
        settings: AgentSettings,
        tools: list[Tool] | None = None,
        server_tools: list[dict[str, Any]] | None = None,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        self.settings = settings
        self.tools: dict[str, Tool] = {t.name: t for t in (tools or [])}
        self.server_tools = server_tools or []
        self.client = client or anthropic.Anthropic()
        self.conversation: list[dict[str, Any]] = []

    def reset(self) -> None:
        self.conversation = []

    def _build_params(self) -> dict[str, Any]:
        system_block = [
            {
                "type": "text",
                "text": self.settings.system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ]
        params: dict[str, Any] = {
            "model": self.settings.model,
            "max_tokens": self.settings.max_tokens,
            "system": system_block,
            "messages": self.conversation,
            "tools": [t.to_api() for t in self.tools.values()] + self.server_tools,
        }
        if self.settings.thinking == "adaptive":
            params["thinking"] = {
                "type": "adaptive",
                "display": self.settings.thinking_display,
            }
            params["output_config"] = {"effort": self.settings.effort}
        elif self.settings.thinking == "disabled":
            params["thinking"] = {"type": "disabled"}
        return params

    def run(self, user_message: str, max_iterations: int = 10) -> str:
        """One-shot: send message, run tool loop, return final text."""
        text_parts: list[str] = []
        for event in self.run_stream(user_message, max_iterations=max_iterations):
            if isinstance(event, TurnDone):
                text_parts.append(event.final_text)
        return "".join(text_parts)

    def run_stream(
        self, user_message: str, max_iterations: int = 10
    ) -> Iterator[Event]:
        """Stream events as the agent works through tools to a final answer."""
        self.conversation.append({"role": "user", "content": user_message})

        for _ in range(max_iterations):
            with self.client.messages.stream(**self._build_params()) as stream:
                for sse in stream:
                    if sse.type != "content_block_delta":
                        continue
                    delta = sse.delta
                    if delta.type == "text_delta":
                        yield TextDelta(delta.text)
                    elif delta.type == "thinking_delta":
                        yield ThinkingDelta(delta.thinking)
                response = stream.get_final_message()

            self.conversation.append(
                {"role": "assistant", "content": response.content}
            )

            tool_uses = [b for b in response.content if b.type == "tool_use"]

            # Server-side tool hit its internal cap; re-send to continue.
            if response.stop_reason == "pause_turn":
                continue

            if not tool_uses or response.stop_reason == "end_turn":
                final_text = "".join(
                    b.text for b in response.content if b.type == "text"
                )
                yield TurnDone(
                    stop_reason=response.stop_reason or "end_turn",
                    final_text=final_text,
                    usage=_usage_dict(response.usage),
                )
                return

            tool_results: list[dict[str, Any]] = []
            for tu in tool_uses:
                yield ToolCall(name=tu.name, input=tu.input or {})
                output, is_error = self._dispatch_tool(tu.name, tu.input or {})
                yield ToolResult(name=tu.name, output=output, is_error=is_error)
                result_block: dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": output,
                }
                if is_error:
                    result_block["is_error"] = True
                tool_results.append(result_block)

            self.conversation.append({"role": "user", "content": tool_results})

        yield TurnDone(stop_reason="max_iterations", final_text="", usage={})

    def _dispatch_tool(
        self, name: str, tool_input: dict[str, Any]
    ) -> tuple[str, bool]:
        tool = self.tools.get(name)
        if tool is None:
            return f"Unknown tool: {name}", True
        try:
            return tool.handler(tool_input), False
        except Exception as e:  # noqa: BLE001 — surface any handler crash to the model
            return f"{type(e).__name__}: {e}", True


def _usage_dict(usage: Any) -> dict[str, int]:
    return {
        "input_tokens": getattr(usage, "input_tokens", 0),
        "output_tokens": getattr(usage, "output_tokens", 0),
        "cache_creation_input_tokens": getattr(
            usage, "cache_creation_input_tokens", 0
        ),
        "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0),
    }
