"""Slack bot adapter using slack-bolt async + Socket Mode.

Requires `pip install 'joshv1[slack]'`. Needs a Slack app with:
  - Socket Mode enabled
  - Bot Token scopes: chat:write, app_mentions:read, im:history, im:write, im:read
  - Event subscriptions: app_mention, message.im

Set SLACK_BOT_TOKEN (xoxb-...) and SLACK_APP_TOKEN (xapp-...) in your .env.

Per-Slack-user conversation history is kept by multiplexing the agent's
session_id, so concurrent DMs don't clobber each other.
"""

from __future__ import annotations

import sys

from joshv1.agent import Agent
from joshv1.config import Config


async def run_slack_bot(config: Config) -> None:
    try:
        from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
        from slack_bolt.async_app import AsyncApp
    except ImportError:
        print(
            "Slack support requires the slack extras: pip install 'joshv1[slack]'",
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

    app = AsyncApp(token=bot_token)
    bot_info = await app.client.auth_test()
    bot_user_id = bot_info["user_id"]

    agent = Agent(config.agent, cwd=config.agent.workspace_dir)
    await agent.connect()

    def _strip_mention(text: str) -> str:
        return text.replace(f"<@{bot_user_id}>", "").strip()

    async def _respond(event: dict, say) -> None:
        user_id = event.get("user")
        if not user_id or event.get("bot_id"):
            return
        text = _strip_mention(event.get("text", ""))
        if not text:
            return
        thread_ts = event.get("thread_ts") or event.get("ts")
        try:
            reply = await agent.run(text, session_id=f"slack-{user_id}")
        except Exception as e:  # noqa: BLE001
            reply = f":warning: {type(e).__name__}: {e}"
        await say(text=reply or "(no response)", thread_ts=thread_ts)

    @app.event("app_mention")
    async def _on_mention(event, say):  # type: ignore[no-untyped-def]
        await _respond(event, say)

    @app.event("message")
    async def _on_message(event, say):  # type: ignore[no-untyped-def]
        if event.get("channel_type") == "im":
            await _respond(event, say)

    print(f"Slack bot starting (model={config.agent.model})...")
    try:
        await AsyncSocketModeHandler(app, app_token).start_async()
    finally:
        await agent.disconnect()
