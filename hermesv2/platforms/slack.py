"""Slack bot adapter using Socket Mode.

Requires `pip install 'hermesv2[slack]'`. Needs a Slack app with:
  - Socket Mode enabled
  - Bot Token scopes: chat:write, app_mentions:read, im:history, im:write, im:read
  - Event subscriptions: app_mention, message.im

Set SLACK_BOT_TOKEN (xoxb-...) and SLACK_APP_TOKEN (xapp-...) in your .env.

Each Slack user gets their own conversation thread so concurrent DMs don't
clobber each other's history.
"""

from __future__ import annotations

import sys
from threading import Lock

from hermesv2.agent import Agent
from hermesv2.config import Config
from hermesv2.tools import build_tools


def run_slack_bot(config: Config) -> None:
    try:
        from slack_bolt import App
        from slack_bolt.adapter.socket_mode import SocketModeHandler
    except ImportError:
        print(
            "Slack support requires the slack extras: pip install 'hermesv2[slack]'",
            file=sys.stderr,
        )
        sys.exit(1)

    bot_token = config.slack.get("bot_token")
    app_token = config.slack.get("app_token")
    if not bot_token or not app_token:
        print(
            "SLACK_BOT_TOKEN and SLACK_APP_TOKEN must be set in the environment.",
            file=sys.stderr,
        )
        sys.exit(1)

    app = App(token=bot_token)
    custom_tools, server_tools = build_tools(config)

    # One Agent per user (per-user conversation history).
    agents: dict[str, Agent] = {}
    agents_lock = Lock()

    def _agent_for(user_id: str) -> Agent:
        with agents_lock:
            if user_id not in agents:
                agents[user_id] = Agent(
                    settings=config.agent,
                    tools=custom_tools,
                    server_tools=server_tools,
                )
            return agents[user_id]

    def _strip_mention(text: str, bot_user_id: str | None) -> str:
        if not bot_user_id:
            return text.strip()
        return text.replace(f"<@{bot_user_id}>", "").strip()

    def _handle(event: dict, say) -> None:
        user_id = event.get("user")
        if not user_id or event.get("bot_id"):
            return
        text = _strip_mention(event.get("text", ""), app.client.auth_test()["user_id"])
        if not text:
            return

        agent = _agent_for(user_id)
        thread_ts = event.get("thread_ts") or event.get("ts")
        try:
            reply = agent.run(text)
        except Exception as e:  # noqa: BLE001
            reply = f":warning: {type(e).__name__}: {e}"
        say(text=reply or "(no response)", thread_ts=thread_ts)

    @app.event("app_mention")
    def on_mention(event, say):  # type: ignore[no-untyped-def]
        _handle(event, say)

    @app.event("message")
    def on_message(event, say):  # type: ignore[no-untyped-def]
        # Only respond to DMs here; channel messages go through app_mention.
        if event.get("channel_type") == "im":
            _handle(event, say)

    print(f"Slack bot starting (model={config.agent.model})...")
    SocketModeHandler(app, app_token).start()
