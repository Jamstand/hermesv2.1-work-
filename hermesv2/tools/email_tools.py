"""Email tools: SMTP send + IMAP inbox read.

Both use the stdlib (`smtplib`, `imaplib`, `email`). For Gmail you'll need an
app-specific password — regular passwords won't work with 2FA accounts.
"""

from __future__ import annotations

import email
import imaplib
import smtplib
from email.message import EmailMessage
from email.utils import parseaddr
from typing import Any

from hermesv2.agent import Tool


def build_email_tools(
    smtp: dict[str, str], imap: dict[str, str]
) -> list[Tool]:
    tools: list[Tool] = []

    if smtp.get("host") and smtp.get("user"):
        tools.append(_build_send_email(smtp))
    if imap.get("host") and imap.get("user"):
        tools.append(_build_read_inbox(imap))

    return tools


def _build_send_email(smtp: dict[str, str]) -> Tool:
    def send_email(args: dict[str, Any]) -> str:
        to_addr = str(args["to"]).strip()
        subject = str(args.get("subject", "")).strip()
        body = str(args.get("body", ""))

        if "@" not in parseaddr(to_addr)[1]:
            return f"Invalid recipient: {to_addr!r}"

        msg = EmailMessage()
        msg["From"] = smtp.get("from") or smtp["user"]
        msg["To"] = to_addr
        msg["Subject"] = subject
        msg.set_content(body)

        port = int(smtp.get("port") or 587)
        with smtplib.SMTP(smtp["host"], port, timeout=30) as server:
            server.starttls()
            server.login(smtp["user"], smtp["password"])
            server.send_message(msg)
        return f"Sent to {to_addr} ({len(body)} bytes)."

    return Tool(
        name="send_email",
        description=(
            "Send an email via the configured SMTP account. Confirm intent "
            "with the user before sending — emails can't be unsent."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "Recipient email address."},
                "subject": {"type": "string"},
                "body": {"type": "string", "description": "Plain-text body."},
            },
            "required": ["to", "subject", "body"],
        },
        handler=send_email,
    )


def _build_read_inbox(imap: dict[str, str]) -> Tool:
    def read_inbox(args: dict[str, Any]) -> str:
        limit = int(args.get("limit", 10))
        port = int(imap.get("port") or 993)
        with imaplib.IMAP4_SSL(imap["host"], port) as m:
            m.login(imap["user"], imap["password"])
            m.select("INBOX")
            typ, data = m.search(None, "ALL")
            if typ != "OK":
                return f"IMAP search failed: {typ}"
            ids = data[0].split()[-limit:]
            lines = []
            for msg_id in reversed(ids):
                typ, raw = m.fetch(msg_id, "(RFC822.HEADER)")
                if typ != "OK" or not raw or not raw[0]:
                    continue
                header_bytes = raw[0][1]
                if not isinstance(header_bytes, bytes):
                    continue
                msg = email.message_from_bytes(header_bytes)
                lines.append(
                    f"#{msg_id.decode()} | {msg.get('From', '?')} | "
                    f"{msg.get('Subject', '(no subject)')} | "
                    f"{msg.get('Date', '?')}"
                )
            return "\n".join(lines) if lines else "(empty inbox)"

    return Tool(
        name="read_inbox",
        description="List recent inbox messages (headers only) via IMAP.",
        input_schema={
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Number of most recent messages to return. Default 10.",
                }
            },
        },
        handler=read_inbox,
    )
