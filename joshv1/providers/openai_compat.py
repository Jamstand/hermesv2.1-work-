"""OpenAI-chat-completion provider. One impl, three free endpoints.

All three target backends speak the OpenAI chat completion protocol:
  - OpenRouter  → https://openrouter.ai/api/v1            (key: OPENROUTER_API_KEY)
  - Ollama      → http://localhost:11434/v1               (no auth, local)
  - Gemini      → https://generativelanguage.googleapis.com/v1beta/openai/
                                                          (key: GEMINI_API_KEY)

Phase 1 surface: text streaming only. Yields TextDelta + TurnDone, never
ToolCall / ToolResult. Phase 2 will add a tool loop that mirrors what the
Claude Agent SDK does internally for Claude.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI, APIStatusError, APIConnectionError

from joshv1.agent import Event, TextDelta, TurnDone
from joshv1.config import AgentSettings
from joshv1.memory import ensure_user_md, memory_dir, memory_system_block


@dataclass(frozen=True)
class ProviderPreset:
    """Connection details for a known OpenAI-compatible backend."""
    base_url: str
    api_key_env: str | None          # None for no-auth (Ollama)
    default_model: str
    label: str                        # human-readable name shown in /model picker


PROVIDER_PRESETS: dict[str, ProviderPreset] = {
    "openrouter": ProviderPreset(
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        # OpenRouter's free-tier lineup churns — see https://openrouter.ai/models?max_price=0.
        # Llama 4 Maverick is the current free flagship (Meta, multimodal, beats
        # Gemini 2.0 Flash in benchmarks).
        default_model="meta-llama/llama-4-maverick:free",
        label="OpenRouter (free tier)",
    ),
    "ollama": ProviderPreset(
        base_url=os.environ.get("OLLAMA_HOST", "http://localhost:11434") + "/v1",
        api_key_env=None,
        default_model="llama3.1",
        label="Ollama (local)",
    ),
    "gemini": ProviderPreset(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key_env="GEMINI_API_KEY",
        default_model="gemini-2.0-flash",
        label="Google Gemini",
    ),
}


class OpenAICompatProvider:
    """Mirrors the `Agent` interface (connect / disconnect / run_stream)
    so the chat loop can swap one in for the other without caring."""

    def __init__(
        self,
        settings: AgentSettings,
        provider_name: str,
        cwd: str | Path | None = None,
    ):
        if provider_name not in PROVIDER_PRESETS:
            raise ValueError(
                f"Unknown provider: {provider_name}. "
                f"Known: {', '.join(PROVIDER_PRESETS)}"
            )
        self.settings = settings
        self.provider_name = provider_name
        self.preset = PROVIDER_PRESETS[provider_name]
        self.cwd = Path(cwd).expanduser() if cwd else None
        self._client: AsyncOpenAI | None = None
        # per-session message history. The Claude SDK keeps this inside its
        # subprocess; we have to do it ourselves.
        self._sessions: dict[str, list[dict[str, Any]]] = {}

    # ------- lifecycle -------------------------------------------------------

    async def connect(self) -> None:
        if self.cwd:
            self.cwd.mkdir(parents=True, exist_ok=True)

        api_key: str | None = None
        if self.preset.api_key_env:
            api_key = os.environ.get(self.preset.api_key_env)
            if not api_key:
                raise RuntimeError(
                    f"{self.preset.label} requires {self.preset.api_key_env} "
                    f"in your environment (.env file or shell)."
                )
        else:
            # Ollama doesn't auth but the openai SDK still wants *something*.
            api_key = "ollama"

        self._client = AsyncOpenAI(base_url=self.preset.base_url, api_key=api_key)

    async def disconnect(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    async def __aenter__(self) -> OpenAICompatProvider:
        await self.connect()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.disconnect()

    async def reconfigure(self) -> None:
        await self.disconnect()
        await self.connect()

    def reset_session(self, base: str = "default") -> str:
        # mirror Agent.reset_session — different session_id means fresh history
        suffix = sum(1 for k in self._sessions if k.startswith(base))
        new_id = f"{base}-{suffix + 1}"
        self._sessions.pop(base, None)
        return new_id

    # ------- main entry point ------------------------------------------------

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
                "Provider not connected. Use `async with provider:` "
                "or call `await provider.connect()` first."
            )

        history = self._sessions.setdefault(session_id, [self._system_message()])
        history.append({"role": "user", "content": user_message})

        text_parts: list[str] = []
        try:
            stream = await self._client.chat.completions.create(
                model=self.settings.model,
                messages=history,
                stream=True,
            )
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta.content:
                    text_parts.append(delta.content)
                    yield TextDelta(delta.content)
        except APIStatusError as e:
            msg = self._friendly_api_error(e)
            text_parts.append(msg)
            yield TextDelta(msg)
            # Pop the user message so the failed turn doesn't poison history.
            history.pop()
        except APIConnectionError as e:
            msg = (
                f"\n[connection error: {e}]\n"
                f"Can't reach {self.preset.base_url}. "
                + ("Is Ollama running? Try `ollama serve`." if self.provider_name == "ollama" else "Check your network.")
            )
            text_parts.append(msg)
            yield TextDelta(msg)
            history.pop()

        final_text = "".join(text_parts)
        history.append({"role": "assistant", "content": final_text})

        yield TurnDone(
            stop_reason="end_turn",
            final_text=final_text,
            cost_usd=None,
            usage={},
        )

    # ------- helpers ---------------------------------------------------------

    def _system_message(self) -> dict[str, Any]:
        mem_dir = memory_dir(self.settings.memory_dir)
        ensure_user_md(mem_dir)
        return {
            "role": "system",
            "content": self.settings.system_prompt + memory_system_block(mem_dir),
        }

    def _friendly_api_error(self, e: APIStatusError) -> str:
        """Turn a raw 404/401/etc. into a one-paragraph hint."""
        status = getattr(e, "status_code", None)
        # Try several places the API message might live, falling back to
        # str(e) (which always contains the raw response body for openai SDK).
        api_msg = ""
        body = getattr(e, "body", None)
        if isinstance(body, dict):
            inner = body.get("error")
            if isinstance(inner, dict):
                api_msg = inner.get("message", "") or ""
            elif isinstance(inner, str):
                api_msg = inner
        if not api_msg:
            api_msg = str(e)
        haystack = api_msg.lower()

        if status == 404 and ("endpoints" in haystack or "not found" in haystack or "no such model" in haystack):
            # OpenRouter shape: "No endpoints found for <model>."
            url = "https://openrouter.ai/models?max_price=0" if self.provider_name == "openrouter" else ""
            extra = f" See {url} for current free models." if url else ""
            return (
                f"\n[model not available: {self.settings.model}]\n"
                f"{self.preset.label} has no endpoint for this model — it may have been "
                f"retired or renamed.{extra}\n"
                f"Switch with: joshv1 setup-provider --provider {self.provider_name} --model <name>\n"
            )
        if status == 401:
            return (
                f"\n[auth failed (401)]\n"
                f"{self.preset.label} rejected the API key. Re-run:\n"
                f"  joshv1 setup-provider --provider {self.provider_name} --key <new key>\n"
            )
        if status == 429:
            return (
                f"\n[rate limited (429)]\n"
                f"{self.preset.label} throttled this request. "
                f"Wait a minute, then retry — or try a different free model.\n"
            )
        return f"\n[{self.preset.label} error {status}: {api_msg or e}]\n"
