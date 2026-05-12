"""File read / write / list tools with optional allowed-roots sandbox."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hermesv2.agent import Tool
from hermesv2.config import FilesToolSettings

MAX_READ_BYTES = 200_000


def _check_path(path: str, allowed_roots: list[str]) -> Path:
    p = Path(path).expanduser().resolve()
    if not allowed_roots:
        return p
    for root in allowed_roots:
        root_p = Path(root).expanduser().resolve()
        try:
            p.relative_to(root_p)
            return p
        except ValueError:
            continue
    raise PermissionError(
        f"Path {p} is outside allowed roots: {allowed_roots}"
    )


def build_file_tools(settings: FilesToolSettings) -> list[Tool]:
    roots = settings.allowed_roots

    def read_file(args: dict[str, Any]) -> str:
        path = _check_path(str(args["path"]), roots)
        if not path.is_file():
            return f"Not a file: {path}"
        data = path.read_bytes()
        if len(data) > MAX_READ_BYTES:
            return (
                f"File is {len(data)} bytes (limit {MAX_READ_BYTES}). "
                "Read a slice with `run_bash` using sed/head/tail."
            )
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            return f"<binary file, {len(data)} bytes>"

    def write_file(args: dict[str, Any]) -> str:
        path = _check_path(str(args["path"]), roots)
        content = str(args.get("content", ""))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"

    def list_dir(args: dict[str, Any]) -> str:
        path = _check_path(str(args.get("path", ".")), roots)
        if not path.is_dir():
            return f"Not a directory: {path}"
        entries = []
        for entry in sorted(path.iterdir()):
            kind = "d" if entry.is_dir() else "f"
            size = entry.stat().st_size if entry.is_file() else 0
            entries.append(f"{kind} {size:>10} {entry.name}")
        return "\n".join(entries) if entries else "(empty)"

    return [
        Tool(
            name="read_file",
            description=(
                "Read a text file from disk and return its contents. Returns an "
                f"explanatory message if the file exceeds {MAX_READ_BYTES} bytes "
                "or isn't UTF-8 text."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or ~-relative path to the file.",
                    }
                },
                "required": ["path"],
            },
            handler=read_file,
        ),
        Tool(
            name="write_file",
            description=(
                "Overwrite a file on disk with the given content. Creates parent "
                "directories as needed. Confirm with the user before overwriting "
                "files containing important work."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
            handler=write_file,
        ),
        Tool(
            name="list_dir",
            description="List the immediate contents of a directory.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path. Defaults to the current working directory.",
                    }
                },
            },
            handler=list_dir,
        ),
    ]
