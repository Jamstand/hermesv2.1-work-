# Hermes v2

A personal AI agent for work, built on the **Claude Agent SDK** so it uses your **Claude Max (or Pro) subscription** via OAuth — no API key, no per-token billing.

> Inspired by the shape of [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent). This is a from-scratch personal-use rebuild, not a port.

## How it bills

- The Agent SDK runs the agent loop inside your local `claude` CLI (the same CLI you use for Claude Code).
- When `claude` is logged in via `claude login`, it authenticates with your Claude account's OAuth tokens.
- Usage counts against your Max plan's quota. No `ANTHROPIC_API_KEY` needed.

## What it does

- **Local CLI / automation** — interactive REPL (`hermesv2 chat`) and one-shot (`hermesv2 run`).
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

### 2. Install hermesv2

```sh
git clone https://github.com/jamstand/hermesv2.1-work-.git hermesv2
cd hermesv2
git checkout claude/hermes-v2-personal-ECfgW
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"
cp .env.example .env             # only edit if you want Slack/Discord
cp config.example.yaml config.yaml   # optional — defaults are sensible
```

### 3. Verify

```sh
hermesv2 doctor
```

Should show green checks for Claude CLI installed, runnable, auth state, config loaded, and workspace writable.

## Use

```sh
hermesv2                                # interactive REPL (default)
hermesv2 chat                           # explicit
hermesv2 chat --session work            # named, resumable session
hermesv2 run "list this directory"      # one-shot
echo "summarize this" | cat README.md - | hermesv2 run
hermesv2 sessions                       # list saved sessions
hermesv2 sessions --delete <id>         # delete one
hermesv2 update                         # git pull
hermesv2 doctor                         # diagnose (now actually pings claude)

hermesv2 slack                          # needs SLACK_BOT_TOKEN + SLACK_APP_TOKEN
hermesv2 discord                        # needs DISCORD_BOT_TOKEN
```

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
hermesv2 chat --session work-2026
```

The first time, this creates a session named `work-2026`. Next time you launch with the same name, it picks up where you left off. List all with `hermesv2 sessions`.

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
  workspace_dir: ~/hermes-workspace
  system_prompt: |
    You are Hermes v2 ...
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
hermesv2/
├── agent.py              # ClaudeSDKClient wrapper + event types
├── cli.py                # click commands: chat / run / doctor / slack / discord
├── config.py             # YAML + env config
└── platforms/
    ├── slack.py
    └── discord.py
```
