"""Ensemble: query multiple models in parallel, synthesize one answer.

Pattern: user prompt → fan out to N draft models in parallel → collect
drafts → ONE judge model (Claude) merges them into a single final answer
that draws on each draft's strengths.

Phase A scope (this module):
- Fixed 3-member ensemble: Claude + Hermes 3 405B + DeepSeek V3.
- Synthesizer is always Claude on whatever model the user's main session
  uses (falls back to claude-opus-4-7 if their main backend isn't Claude).
- Members run in parallel via asyncio.gather; if a member fails (rate
  limit, model retired, network error), it's dropped silently.
- If only one member succeeds, return its answer raw — no synthesis needed.
- If zero succeed, return an error summary listing each failure reason.

Phase B (later): configurable members in YAML, per-prompt routing by
prompt-classification, expose draft transcripts for transparency.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, replace

from joshv1.agent import Agent
from joshv1.config import AgentSettings
from joshv1.providers import OpenAICompatProvider


@dataclass
class EnsembleMember:
    """One model in the draft pool."""
    provider: str          # "claude" or an OpenAI-compat provider name
    model: str
    role: str              # one-line description used in the synth prompt


DEFAULT_ENSEMBLE: list[EnsembleMember] = [
    EnsembleMember(
        provider="claude",
        model="claude-opus-4-7",
        role="reasoning, code review, careful analysis",
    ),
    EnsembleMember(
        provider="openrouter",
        # Gemini 3.1 Flash Lite: speed-optimized + multimodal (audio/video/PDF).
        # Fills the two biggest gaps Opus has — latency and non-text input.
        # PAID tier (no `:free` suffix); needs an OpenRouter balance.
        model="google/gemini-3.1-flash-lite",
        role="fast drafts, multimodal input (audio/video/PDF), low-latency generalist",
    ),
    EnsembleMember(
        provider="openrouter",
        # Qwen3-Coder is a coding-specialized model — math removed from the
        # role since the Coder variant isn't math-RL'd (that's Qwen3-Math).
        model="qwen/qwen3-coder:free",
        role="code generation, technical detail",
    ),
    EnsembleMember(
        provider="openrouter",
        # DeepSeek V4 Pro: 1.6T MoE, math-RL'd, cost-efficient at scale.
        # Different lab from Anthropic / Google / Alibaba for breadth.
        # PAID tier (no `:free` suffix); ~40× cheaper per token than Opus.
        model="deepseek/deepseek-v4-pro",
        role="math, quantitative reasoning, cost-efficient MoE perspective",
    ),
]


DRAFT_SYSTEM_PROMPT = """\
You are one of several expert models being polled in parallel for the same
user question. You are particularly strong at: {role}.

Answer the question directly and well — no preamble, no "I'd be happy to
help," no meta-commentary about other models. Lean into your strength
without restricting yourself to only it. If the question doesn't suit your
strength, still answer with what you've got.
"""


SYNTHESIZER_SYSTEM_PROMPT = """\
You are synthesizing several expert drafts into ONE final answer to a user
question. Produce a single coherent response — do NOT enumerate the drafts
or say "Draft 1 says X, Draft 2 says Y." Pull the best phrasings, facts,
and code from each draft. If drafts contradict, pick the right one (or
the right blend) and commit to it. Drop hedging and any meta-commentary
about the drafts themselves. Match the format the user asked for. Just
the answer — no "Here is the synthesized response:" preamble.
"""


@dataclass
class Draft:
    label: str             # display name, usually the model id
    role: str
    text: str
    success: bool
    error: str | None = None


# Errors that surface from OpenAICompatProvider are inlined as text starting
# with one of these markers (vs. raised exceptions). Used to detect "soft"
# failures that produced output but no real answer.
_ERROR_MARKERS = (
    "[connection error",
    "[rate limited",
    "[auth failed",
    "[model not available",
    "[OpenRouter",
    "[Ollama",
    "[Google Gemini",
)


def _looks_like_error(text: str) -> bool:
    stripped = text.lstrip()
    return any(stripped.startswith(m) for m in _ERROR_MARKERS)


async def _query_one(
    member: EnsembleMember, prompt: str, base: AgentSettings,
) -> Draft:
    """Run one ensemble member. Returns a Draft (success=False on any error)."""
    settings = replace(
        base,
        provider=member.provider,
        model=member.model,
        system_prompt=DRAFT_SYSTEM_PROMPT.format(role=member.role),
    )
    backend = (
        Agent(settings) if member.provider == "claude"
        else OpenAICompatProvider(settings, provider_name=member.provider)
    )
    # Track which phase fails so the error message actually identifies the cause.
    phase = "init"
    try:
        phase = "connect"
        await backend.connect()
        try:
            phase = "run"
            result = await backend.run(prompt)
        finally:
            phase = "disconnect"
            try:
                await backend.disconnect()
            except Exception:  # noqa: BLE001
                pass
        if not result:
            return Draft(label=member.model, role=member.role, text="",
                         success=False, error="empty response from model")
        if _looks_like_error(result):
            return Draft(label=member.model, role=member.role, text="",
                         success=False, error=result.strip())
        return Draft(label=member.model, role=member.role, text=result, success=True)
    except Exception as e:  # noqa: BLE001
        return Draft(label=member.model, role=member.role, text="",
                     success=False,
                     error=f"[{phase}] {type(e).__name__}: {e or '<no message>'}")


async def _synthesize(prompt: str, drafts: list[Draft], base: AgentSettings) -> str:
    """Have Claude merge the successful drafts into one answer."""
    drafts_block = "\n\n".join(
        f"--- DRAFT {i+1}: {d.label} ({d.role}) ---\n{d.text}"
        for i, d in enumerate(drafts)
    )
    synth_user = f"USER QUESTION:\n{prompt}\n\nDRAFTS:\n{drafts_block}"

    claude_model = base.model if base.provider == "claude" else "claude-opus-4-7"
    judge_settings = replace(
        base,
        provider="claude",
        model=claude_model,
        system_prompt=SYNTHESIZER_SYSTEM_PROMPT,
    )
    async with Agent(judge_settings) as judge:
        return await judge.run(synth_user)


async def collect_drafts(
    prompt: str,
    base_settings: AgentSettings,
    members: list[EnsembleMember] | None = None,
    on_draft_complete: Callable[[EnsembleMember, Draft], None] | None = None,
) -> list[Draft]:
    """Query all ensemble members in parallel; return raw drafts (no synthesis).

    Used by the auto-routing path that feeds drafts into the main Claude session
    as context instead of synthesizing one merged answer.
    """
    if members is None:
        members = members_from_settings(base_settings)

    async def _with_callback(m: EnsembleMember) -> Draft:
        d = await _query_one(m, prompt, base_settings)
        if on_draft_complete is not None:
            on_draft_complete(m, d)
        return d

    return list(await asyncio.gather(*(_with_callback(m) for m in members)))


async def run_ensemble(
    prompt: str,
    base_settings: AgentSettings,
    members: list[EnsembleMember] | None = None,
    on_draft_complete: Callable[[EnsembleMember, Draft], None] | None = None,
) -> tuple[str, list[Draft]]:
    """Query all ensemble members in parallel, return synthesized answer + drafts.

    Returns: (final_answer, all_drafts) — `all_drafts` includes failures so the
    UI can show what went wrong, but `final_answer` is already a usable string.
    """
    drafts = await collect_drafts(prompt, base_settings, members, on_draft_complete)
    successful = [d for d in drafts if d.success]

    if not successful:
        summary = "[ensemble: all models failed]\n" + "\n".join(
            f"  - {d.label}: {d.error}" for d in drafts
        )
        return summary, drafts

    if len(successful) == 1:
        return successful[0].text, drafts

    final = await _synthesize(prompt, successful, base_settings)
    return final, drafts


def format_drafts_as_context(
    user_message: str, successful_drafts: list[Draft],
) -> str:
    """Wrap successful drafts into an augmented prompt for the main Claude session.

    The main session sees: original user message + drafts appendix telling it
    to use them as context but reply in its own voice with full tool access.
    Returns the augmented user message string.
    """
    drafts_block = "\n\n".join(
        f"--- DRAFT {i+1}: {d.label} ({d.role}) ---\n{d.text}"
        for i, d in enumerate(successful_drafts)
    )
    return (
        f"{user_message}\n\n"
        "---\n"
        "[ENSEMBLE CONTEXT — system-injected, not from the user]\n"
        f"Several peer expert models drafted answers to my message above in "
        f"parallel. They have no tools, no memory of this conversation, and "
        f"no awareness of each other. Their drafts:\n\n"
        f"{drafts_block}\n\n"
        "Use these as input. You retain full tool access, this session's "
        "scrollback, and your normal voice. Pull the best phrasings/facts/"
        "code; correct mistakes; verify with tools if a draft makes a "
        "claim you can check. Do NOT enumerate the drafts (\"Draft 1 says…\"), "
        "do NOT mention that other models were consulted, do NOT prefix with "
        "\"Synthesized answer:\". Just answer the user's message naturally, "
        "as if the drafts never existed but happened to inform you.\n"
    )


def members_from_settings(settings: AgentSettings) -> list[EnsembleMember]:
    """Resolve ensemble members from YAML config, falling back to the default pool.

    YAML shape (under `agent:`):
        ensemble:
          - provider: claude
            model: claude-opus-4-7
            role: reasoning, code review
          - provider: openrouter
            model: nousresearch/hermes-3-llama-3.1-405b:free
            role: creative writing
    """
    if not settings.ensemble:
        return list(DEFAULT_ENSEMBLE)
    out: list[EnsembleMember] = []
    for entry in settings.ensemble:
        try:
            out.append(EnsembleMember(
                provider=entry["provider"],
                model=entry["model"],
                role=entry.get("role", ""),
            ))
        except (KeyError, TypeError):
            # Skip malformed entries silently — better than crashing the whole
            # ensemble because of one bad YAML line.
            continue
    return out or list(DEFAULT_ENSEMBLE)
