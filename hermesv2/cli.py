"""Hermes v2 CLI: `hermesv2 chat`, `hermesv2 run`, `hermesv2 slack`, `hermesv2 discord`."""

from __future__ import annotations

import sys

import click
from rich.console import Console

from hermesv2.agent import (
    Agent,
    TextDelta,
    ThinkingDelta,
    ToolCall,
    ToolResult,
    TurnDone,
)
from hermesv2.config import load_config
from hermesv2.tools import build_tools

console = Console()


def _make_agent(config_path: str | None) -> Agent:
    cfg = load_config(config_path)
    if not cfg.anthropic_api_key:
        console.print(
            "[red]ANTHROPIC_API_KEY is not set. Add it to .env or export it.[/]"
        )
        sys.exit(1)
    tools, server_tools = build_tools(cfg)
    return Agent(settings=cfg.agent, tools=tools, server_tools=server_tools)


def _render_event(event: object) -> None:
    if isinstance(event, TextDelta):
        console.print(event.text, end="", soft_wrap=True, highlight=False)
    elif isinstance(event, ThinkingDelta):
        console.print(
            f"[dim italic]{event.text}[/]", end="", soft_wrap=True, highlight=False
        )
    elif isinstance(event, ToolCall):
        console.print(f"\n[cyan]→ {event.name}[/] [dim]{event.input}[/]")
    elif isinstance(event, ToolResult):
        color = "red" if event.is_error else "green"
        preview = event.output if len(event.output) < 400 else event.output[:400] + "..."
        console.print(f"[{color}]← {event.name}[/] [dim]{preview}[/]")
    elif isinstance(event, TurnDone) and event.usage:
        cached = event.usage.get("cache_read_input_tokens", 0)
        console.print(
            f"\n[dim](stop={event.stop_reason}, "
            f"in={event.usage.get('input_tokens', 0)}, "
            f"out={event.usage.get('output_tokens', 0)}, "
            f"cached={cached})[/]"
        )


@click.group()
@click.version_option()
def main() -> None:
    """Hermes v2 — personal agent on top of Claude."""


@main.command()
@click.option("--config", "config_path", default=None, help="Path to config YAML.")
@click.argument("prompt", nargs=-1)
def run(config_path: str | None, prompt: tuple[str, ...]) -> None:
    """Run a single prompt and exit. Reads from stdin if no prompt is given."""
    if prompt:
        message = " ".join(prompt)
    elif not sys.stdin.isatty():
        message = sys.stdin.read().strip()
    else:
        console.print("[red]No prompt given. Pass one as args or pipe via stdin.[/]")
        sys.exit(2)

    agent = _make_agent(config_path)
    for event in agent.run_stream(message):
        _render_event(event)
    console.print()


@main.command()
@click.option("--config", "config_path", default=None, help="Path to config YAML.")
def chat(config_path: str | None) -> None:
    """Interactive REPL. Ctrl-D or 'exit' to quit; '/reset' clears history."""
    agent = _make_agent(config_path)
    console.print(
        f"[bold green]Hermes v2[/] — model [cyan]{agent.settings.model}[/], "
        f"effort [cyan]{agent.settings.effort}[/]. Type /reset to clear history."
    )

    while True:
        try:
            line = console.input("\n[bold blue]you›[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nbye.")
            return
        if not line:
            continue
        if line in ("exit", "quit", "/exit", "/quit"):
            return
        if line == "/reset":
            agent.reset()
            console.print("[dim](history cleared)[/]")
            continue

        console.print("[bold magenta]hermes›[/] ", end="")
        for event in agent.run_stream(line):
            _render_event(event)
        console.print()


@main.command()
@click.option("--config", "config_path", default=None, help="Path to config YAML.")
def slack(config_path: str | None) -> None:
    """Run the Slack bot (Socket Mode)."""
    from hermesv2.platforms.slack import run_slack_bot

    cfg = load_config(config_path)
    run_slack_bot(cfg)


@main.command()
@click.option("--config", "config_path", default=None, help="Path to config YAML.")
def discord(config_path: str | None) -> None:
    """Run the Discord bot."""
    from hermesv2.platforms.discord import run_discord_bot

    cfg = load_config(config_path)
    run_discord_bot(cfg)


# --- Plaid -------------------------------------------------------------------


def _plaid_client(config_path: str | None):
    cfg = load_config(config_path)
    if not cfg.plaid.get("client_id") or not cfg.plaid.get("secret"):
        console.print("[red]PLAID_CLIENT_ID / PLAID_SECRET are not set.[/]")
        sys.exit(1)
    from hermesv2.integrations.plaid_client import PlaidClient, PlaidNotInstalled

    try:
        client = PlaidClient(
            cfg.tools.plaid,
            client_id=cfg.plaid["client_id"],
            secret=cfg.plaid["secret"],
            env=cfg.plaid.get("env") or cfg.tools.plaid.env,
        )
    except PlaidNotInstalled as e:
        console.print(f"[red]{e}[/]")
        sys.exit(1)
    return cfg, client


@main.group()
def plaid() -> None:
    """Manage Plaid bank/card linking and transaction sync."""


@plaid.command("link")
@click.option("--config", "config_path", default=None, help="Path to config YAML.")
@click.option("--no-browser", is_flag=True, help="Don't auto-open the browser.")
def plaid_link(config_path: str | None, no_browser: bool) -> None:
    """One-time interactive flow to link a bank/card via Plaid Link."""
    from pathlib import Path

    from hermesv2.integrations.plaid_link_server import run_link_flow

    cfg, client = _plaid_client(config_path)
    items_path = Path(cfg.tools.plaid.items_path).expanduser()
    console.print(
        f"[cyan]Plaid env:[/] {client.env_name} — opening Link "
        f"on http://127.0.0.1:{cfg.tools.plaid.link_port}"
    )
    item = run_link_flow(
        client,
        items_path=items_path,
        port=cfg.tools.plaid.link_port,
        open_browser=not no_browser,
    )
    console.print(
        f"[green]Linked {item.get('institution_name', '?')}[/] "
        f"(item_id={item['item_id'][:8]}). Token saved to {items_path}."
    )


@plaid.command("sync")
@click.option("--config", "config_path", default=None, help="Path to config YAML.")
def plaid_sync(config_path: str | None) -> None:
    """Pull new transactions from every linked institution."""
    from hermesv2.tools.plaid_tools import build_plaid_tools

    cfg, _ = _plaid_client(config_path)
    tools = {
        t.name: t
        for t in build_plaid_tools(
            cfg.tools.plaid, cfg.plaid, cfg.tools.subscriptions.store_path
        )
    }
    console.print(tools["plaid_sync_transactions"].handler({}))


@plaid.command("recurring")
@click.option("--config", "config_path", default=None, help="Path to config YAML.")
@click.option("--min-amount", default=0.0, type=float)
def plaid_recurring(config_path: str | None, min_amount: float) -> None:
    """Show recurring outflows Plaid detected from real transactions."""
    from hermesv2.tools.plaid_tools import build_plaid_tools

    cfg, _ = _plaid_client(config_path)
    tools = {
        t.name: t
        for t in build_plaid_tools(
            cfg.tools.plaid, cfg.plaid, cfg.tools.subscriptions.store_path
        )
    }
    console.print(
        tools["plaid_recurring_subscriptions"].handler({"min_amount": min_amount})
    )
