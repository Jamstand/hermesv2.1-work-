"""Discord bot adapter using discord.py.

Requires `pip install 'hermesv2[discord]'`. Needs a bot token from the Discord
Developer Portal with the MESSAGE_CONTENT privileged intent enabled.

The bot responds to direct messages and to @mentions in channels. Per-user
history is kept by multiplexing the agent's session_id, so different users
don't share conversation state.
"""

from __future__ import annotations

import sys

from hermesv2.agent import Agent
from hermesv2.config import Config


async def run_discord_bot(config: Config) -> None:
    try:
        import discord
    except ImportError:
        print(
            "Discord support requires the discord extras: pip install 'hermesv2[discord]'",
            file=sys.stderr,
        )
        sys.exit(1)

    token = config.discord.get("bot_token")
    if not token:
        print("DISCORD_BOT_TOKEN must be set in the environment.", file=sys.stderr)
        sys.exit(1)

    agent = Agent(config.agent, cwd=config.agent.workspace_dir)
    await agent.connect()

    intents = discord.Intents.default()
    intents.message_content = True
    intents.dm_messages = True
    client = discord.Client(intents=intents)

    @client.event
    async def on_ready() -> None:
        print(f"Discord bot online as {client.user} (model={config.agent.model})")

    @client.event
    async def on_message(message: discord.Message) -> None:
        if message.author == client.user or message.author.bot:
            return

        is_dm = isinstance(message.channel, discord.DMChannel)
        is_mention = client.user in message.mentions
        if not (is_dm or is_mention):
            return

        text = message.content
        if client.user is not None:
            text = text.replace(f"<@{client.user.id}>", "").strip()
        if not text:
            return

        async with message.channel.typing():
            try:
                reply = await agent.run(text, session_id=f"discord-{message.author.id}")
            except Exception as e:  # noqa: BLE001
                reply = f":warning: {type(e).__name__}: {e}"

        for chunk in _chunk(reply or "(no response)", 1900):
            await message.channel.send(chunk)

    try:
        await client.start(token)
    finally:
        await agent.disconnect()


def _chunk(text: str, size: int) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)] or [""]
