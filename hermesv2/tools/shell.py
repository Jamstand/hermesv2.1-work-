"""Shell execution tool with a regex denylist and a wall-clock timeout."""

from __future__ import annotations

import re
import subprocess
from typing import Any

from hermesv2.agent import Tool
from hermesv2.config import ShellToolSettings


def build_shell_tools(settings: ShellToolSettings) -> list[Tool]:
    compiled = [re.compile(p) for p in settings.blocked_patterns]

    def run_bash(args: dict[str, Any]) -> str:
        command = str(args.get("command", "")).strip()
        if not command:
            return "Error: command is required"

        for pat in compiled:
            if pat.search(command):
                return (
                    f"Refused: command matches blocked pattern /{pat.pattern}/. "
                    "Ask the user before running it."
                )

        try:
            proc = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=settings.timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return f"Timed out after {settings.timeout_seconds}s."

        out = proc.stdout or ""
        err = proc.stderr or ""
        body = (
            f"exit={proc.returncode}\n"
            f"--- stdout ---\n{out}\n"
            f"--- stderr ---\n{err}"
        )
        if len(body) > 8000:
            body = body[:8000] + "\n[...truncated]"
        return body

    return [
        Tool(
            name="run_bash",
            description=(
                "Execute a bash command on the local machine and return exit code, "
                "stdout, and stderr. Output is truncated at 8KB. Use this for git, "
                "ls, grep, sed, package managers, and one-off scripts. For destructive "
                "operations (rm, force-push, dropping data), confirm with the user first."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to run.",
                    }
                },
                "required": ["command"],
            },
            handler=run_bash,
        )
    ]
