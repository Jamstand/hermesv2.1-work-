"""Tool registry. Each builder takes the Config and returns a list of Tools."""

from __future__ import annotations

from typing import Any

from hermesv2.agent import Tool
from hermesv2.config import Config
from hermesv2.tools.email_tools import build_email_tools
from hermesv2.tools.files import build_file_tools
from hermesv2.tools.shell import build_shell_tools
from hermesv2.tools.subscriptions import build_subscription_tools


def build_tools(config: Config) -> tuple[list[Tool], list[dict[str, Any]]]:
    """Return (custom_tools, server_tools) based on config flags."""
    custom: list[Tool] = []
    server: list[dict[str, Any]] = []

    if config.tools.shell.enabled:
        custom.extend(build_shell_tools(config.tools.shell))
    if config.tools.files.enabled:
        custom.extend(build_file_tools(config.tools.files))
    if config.tools.subscriptions.enabled:
        custom.extend(build_subscription_tools(config.tools.subscriptions))
    if config.tools.email and config.smtp.get("host"):
        custom.extend(build_email_tools(config.smtp, config.imap))
    if config.tools.web:
        server.extend(
            [
                {"type": "web_search_20260209", "name": "web_search"},
                {"type": "web_fetch_20260209", "name": "web_fetch"},
            ]
        )

    return custom, server
