# Josh v1

A personal AI agent for work, built on the **Claude Agent SDK** so it uses your **Claude Max (or Pro) subscription** via OAuth — no API key, no per-token billing.

> Inspired by the shape of [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent). This is a from-scratch personal-use rebuild, not a port.

## How it bills

- The Agent SDK runs the agent loop inside your local `claude` CLI (the same CLI you use for Claude Code).
- When `claude` is logged in via `claude login`, it authenticates with your Claude account's OAuth tokens.
- Usage counts against your Max plan's quota. No `ANTHROPIC_API_KEY` needed.

## What it does

- **Local CLI / automation** — interactive REPL (`joshv1 chat`) and one-shot (`joshv1 run`).
- **Slack bot** — DMs and @mentions, per-user conversation threads.
- **Discord bot** — DMs and @mentions, same behavior.
- **Built-in tools** — Claude Code's Read, Write, Edit, Bash, Grep, Glob, WebFetch, WebSearch all come for free.

## Install

### 1. Prerequisites

You need Node.js + Claude Code installed and logged in:

```sh
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo bash -
sudo apt install -y nodejs
sudo npm install -g @anthropic-ai/claude-code
claude login
```

Then Python 3.10+ and pip:

```sh
sudo apt install -y python3-venv python3-pip python-is-python3 git
```

### 2. Install joshv1

```sh
git clone https://github.com/jamstand/joshv1-work-.git joshv1
cd joshv1
git checkout claude/josh-v1-personal-ECfgW
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"
cp .env.example .env             # only edit if you want Slack/Discord
cp config.example.yaml config.yaml   # optional — defaults are sensible
```

### 3. Verify

```sh
joshv1 doctor
```

Should show green checks for Claude CLI installed, runnable, auth state, config loaded, and workspace writable.

## Use

```sh
joshv1                                # interactive REPL (default)
joshv1 chat                           # explicit
joshv1 chat --session work            # named, resumable session
joshv1 run "list this directory"      # one-shot
echo "summarize this" | cat README.md - | joshv1 run
joshv1 sessions                       # list saved sessions
joshv1 sessions --delete <id>         # delete one
joshv1 update                         # git pull
joshv1 doctor                         # diagnose (now actually pings claude)

joshv1 index ~/notes                  # FTS5-index a directory of text/md files
joshv1 search "deploy"                # search the indexed corpus

joshv1 voice recording.m4a            # transcribe an audio file (needs [voice] extra)
joshv1 voice recording.m4a --run      # transcribe then run as a prompt

joshv1 slack                          # needs SLACK_BOT_TOKEN + SLACK_APP_TOKEN
joshv1 discord                        # needs DISCORD_BOT_TOKEN
```

## Persistent memory

Joshv1 reads every `*.md` file under `~/.josh-memory/` at the start of each turn and appends them to the system prompt inside a `<memory>` block. `USER.md` is auto-created on first run and the agent updates it via the Write tool when it learns durable facts about you (preferences, ongoing projects, decisions). This is the Honcho-style user-modeling layer: persistent context, zero round-trips, all local.

Edit any file in `~/.josh-memory/` to seed context manually.

## Notes search (FTS5)

```sh
joshv1 index ~/notes               # walks recursively for *.md *.txt *.markdown *.org *.rst
joshv1 search "kubernetes ingress"
joshv1 search "kubernetes ingress" --json   # pipe-friendly
```

Backed by SQLite FTS5 (stdlib, no extra deps). Returns top BM25-ranked snippets. For semantic ranking, pipe the JSON output into `joshv1 run "rank these by relevance to <topic>: ..."`.

## Voice input

```sh
pip install -e ".[voice]"           # one-time: pulls faster-whisper
joshv1 voice meeting.m4a          # prints transcript
joshv1 voice meeting.m4a --run    # transcribe then send to agent
```

WSL note: live mic capture is fragile because of PulseAudio passthrough. Easier path: record on Windows with the built-in Voice Recorder app (saves `.m4a` to `~/Documents/Sound Recordings/`), then `joshv1 voice /mnt/c/Users/.../recording.m4a`.

In the chat REPL, type `/help` for the full slash-command list:

| Command | Effect |
|---|---|
| `/help` | show commands |
| `/exit`, `/quit` | quit |
| `/reset` | clear conversation (new internal session id) |
| `/clear` | clear screen |
| `/context` | current context-window usage |
| `/stats` | turns + tokens + elapsed for this session |
| `/sessions` | list saved Claude Code sessions |
| `/tools` | list built-in tools |
| `/model <name>` | switch model mid-session |
| `/update` | git pull the install |

## Sessions

Claude Code persists sessions automatically. Resume across restarts with:

```sh
joshv1 chat --session work-2026
```

The first time, this creates a session named `work-2026`. Next time you launch with the same name, it picks up where you left off. List all with `joshv1 sessions`.

## MCP servers

To plug in Model Context Protocol servers (filesystem, GitHub, Slack, Linear, etc.), add them to `config.yaml`:

```yaml
agent:
  mcp_servers:
    filesystem:
      type: stdio
      command: npx
      args: ["-y", "@modelcontextprotocol/server-filesystem", "/home/josh/projects"]
    github:
      type: stdio
      command: npx
      args: ["-y", "@modelcontextprotocol/server-github"]
      env:
        GITHUB_PERSONAL_ACCESS_TOKEN: ${GITHUB_PAT}
```

These run as subprocesses and expose their tools to the agent transparently.

## Skills

Claude Code skills (markdown files in `~/.claude/skills/`) load automatically. To load only specific ones:

```yaml
agent:
  skills: [code-review, sql-explainer]
```

## Configuration

`config.yaml`:

```yaml
agent:
  model: claude-opus-4-7
  effort: high                  # low | medium | high | xhigh | max
  thinking: adaptive            # adaptive | disabled
  permission_mode: default      # default | acceptEdits | plan | bypassPermissions
  workspace_dir: ~/josh-workspace
  system_prompt: |
    You are Josh v1 ...
```

**`permission_mode`** matters most:
- `default` — prompts before destructive actions (good for the CLI)
- `acceptEdits` — auto-approves file edits (good for bots)
- `bypassPermissions` — never prompts (use carefully)

## Behind a corporate proxy

If you're behind a TLS-intercepting proxy (BlueCoat / Zscaler / Netskope), Node.js (which runs `claude`) needs the proxy's CA cert too:

```sh
# Install the corp CA in WSL's system store, then:
export NODE_EXTRA_CA_CERTS=/etc/ssl/certs/ca-certificates.crt
echo 'export NODE_EXTRA_CA_CERTS=/etc/ssl/certs/ca-certificates.crt' >> ~/.bashrc
```

## What you don't get

- **No API key path.** The point of this rewrite was to use Max instead of API billing. If you need pay-as-you-go API access, use the prior commit (`b1cbee1`) before the SDK switch.
- **No per-token streaming.** The Agent SDK streams per-content-block, not per-token. Output appears in chunks, not letter-by-letter. Acceptable tradeoff for free-via-Max.
- **No history persistence.** Conversation history lives in memory; restart wipes it.

## Layout

```
joshv1/
├── agent.py              # ClaudeSDKClient wrapper + event types
├── cli.py                # click commands: chat / run / doctor / slack / discord
├── config.py             # YAML + env config
└── platforms/
    ├── slack.py
    └── discord.py
```
