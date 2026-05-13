"""Pluggable LLM backends beyond Claude.

`Agent` (in joshv1/agent.py) talks to Claude via the Claude Agent SDK.
Everything else — OpenRouter free tier, local Ollama, Gemini direct —
goes through `OpenAICompatProvider`. All three speak the OpenAI chat
completion protocol; they differ only in base URL and auth header.

Phase 1 (current): text-only chat. No tool execution yet, so non-Claude
providers can answer questions but can't Read/Write/Bash. Phase 2 will
add a local tool loop so tool-use parity is achievable.

Provider selection happens in two places:
  - `cfg.agent.provider`  — default ("claude" preserves existing behavior)
  - `/model openrouter:<name>` slash command — switches mid-session
"""

from joshv1.providers.openai_compat import OpenAICompatProvider, PROVIDER_PRESETS

__all__ = ["OpenAICompatProvider", "PROVIDER_PRESETS"]
