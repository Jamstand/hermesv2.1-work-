# Hermes v2

A personal AI agent for work, in the spirit of [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent) — built from scratch on the Claude API.

> This is **not** a port of the upstream `hermes-agent` codebase. It's a small, focused Python agent that covers the same shape (chat platform integrations, local automation, email/calendar) without trying to match every feature.

## What it does

- **Local CLI / automation** — interactive REPL and one-shot mode that can read/write files, run shell commands, and search the web.
- **Slack bot** — DMs and @mentions, per-user conversation threads.
- **Discord bot** — DMs and @mentions, same behavior.
- **Email** — send via SMTP and read inbox headers via IMAP, when configured.
- **Web** — uses Claude's server-side `web_search` and `web_fetch` tools (no API keys to manage).

Under the hood:

- Claude Opus 4.7 by default, with **adaptive thinking** (`effort: high`).
- **Prompt caching** on the system prompt — repeated requests hit the cache.
- Streaming responses with visible reasoning summaries in the CLI.
- Custom tools execute on your machine; web tools run server-side at Anthropic.

## Install

```sh
git clone <this-repo>
cd hermesv2.1-work-
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"          # or .[slack] / .[discord] / just .
cp .env.example .env             # fill in ANTHROPIC_API_KEY
cp config.example.yaml config.yaml   # tweak optional knobs
```

## Use

```sh
# Interactive chat
hermesv2 chat

# One-shot
hermesv2 run "summarize the top 5 files in this directory"
echo "what's the weather in SF?" | hermesv2 run

# Slack bot (needs SLACK_BOT_TOKEN + SLACK_APP_TOKEN — Socket Mode)
hermesv2 slack

# Discord bot (needs DISCORD_BOT_TOKEN + Message Content intent)
hermesv2 discord
```

In the chat REPL:
- `/reset` — clear history
- `exit` or Ctrl-D — quit

## Tools that ship by default

| Tool          | Where it runs | Notes |
| ------------- | ------------- | ----- |
| `run_bash`    | Local         | Regex denylist + timeout. Confirms destructive ops in-band. |
| `read_file`   | Local         | UTF-8, up to 200KB. |
| `write_file`  | Local         | Creates parent dirs. |
| `list_dir`    | Local         | One level deep. |
| `send_email`  | Local (SMTP)  | Off by default; flip `tools.email` and set `SMTP_*` env vars. |
| `read_inbox`  | Local (IMAP)  | Same. |
| `web_search`  | Server-side   | Built-in Claude tool. |
| `web_fetch`   | Server-side   | Built-in Claude tool. |

Sandboxing: `tools.files.allowed_roots` in `config.yaml` restricts file ops to specific paths. `tools.shell.blocked_patterns` adds regex denies for bash commands.

## Adding a tool

In `hermesv2/tools/` write a small builder:

```python
from hermesv2.agent import Tool

def build_my_tools(settings) -> list[Tool]:
    def handler(args: dict) -> str:
        return f"got {args['x']}"

    return [
        Tool(
            name="my_thing",
            description="Does the thing.",
            input_schema={
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": ["x"],
            },
            handler=handler,
        )
    ]
```

Then wire it into `hermesv2/tools/__init__.py:build_tools`.

## What's intentionally missing

The upstream `hermes-agent` framework boasts 20+ platform integrations, multi-agent kanban, voice cloning, video analysis, etc. This is a personal-scope rebuild — it has Slack, Discord, email, and local file/shell. Add what you actually use.

Calendar (CalDAV / Google Calendar) is a likely next addition; the scaffolding is there to drop a tool in.

## Layout

```
hermesv2/
├── agent.py            # core loop: streaming + tool dispatch + caching
├── config.py           # YAML + env config
├── cli.py              # click-based CLI
├── tools/
│   ├── shell.py
│   ├── files.py
│   └── email_tools.py
└── platforms/
    ├── slack.py
    └── discord.py
```
