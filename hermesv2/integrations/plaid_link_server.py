"""Local one-shot Plaid Link server for the CLI.

Plaid Link is browser-based, so we serve a tiny HTML page on localhost that
loads the Plaid Link JS, opens the modal with our `link_token`, and posts the
resulting `public_token` back to a `/callback` endpoint. The endpoint exchanges
it for a long-lived access_token and persists it. The server then shuts down.
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

from hermesv2.integrations.plaid_client import PlaidClient


def _items_load(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text() or "[]")
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def _items_save(path: Path, items: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(items, indent=2))
    with contextlib.suppress(OSError):
        os.chmod(path, 0o600)


_INDEX_HTML = """<!doctype html>
<html><head><title>Hermes — Plaid Link</title>
<script src="https://cdn.plaid.com/link/v2/stable/link-initialize.js"></script>
<style>body{font-family:system-ui;padding:2rem;max-width:40rem}</style>
</head><body>
<h1>Link a bank to Hermes</h1>
<p id="status">Opening Plaid Link…</p>
<script>
const handler = Plaid.create({
  token: "__LINK_TOKEN__",
  onSuccess: (public_token, metadata) => {
    document.getElementById("status").textContent = "Linked. Exchanging token…";
    fetch("/callback", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({public_token, institution: metadata.institution})
    }).then(r => r.json()).then(j => {
      document.getElementById("status").textContent =
        j.ok ? `Saved ${j.institution}. You can close this tab.` : `Error: ${j.error}`;
    });
  },
  onExit: (err) => {
    document.getElementById("status").textContent = err
      ? `Exited: ${err.error_message || err.error_code}` : "Exited.";
  }
});
handler.open();
</script>
</body></html>
"""


def run_link_flow(
    plaid_client: PlaidClient,
    items_path: Path,
    port: int = 8765,
    open_browser: bool = True,
    timeout_seconds: int = 600,
) -> dict[str, Any]:
    """Block until the user finishes Link, return the new item dict."""
    link_token = plaid_client.create_link_token()
    page = _INDEX_HTML.replace("__LINK_TOKEN__", link_token)

    result: dict[str, Any] = {}
    done = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:  # silence access log
            return

        def do_GET(self) -> None:
            if self.path != "/":
                self.send_response(404)
                self.end_headers()
                return
            body = page.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            if self.path != "/callback":
                self.send_response(404)
                self.end_headers()
                return
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            public_token = payload.get("public_token")
            institution = (payload.get("institution") or {}).get("name") or ""
            try:
                access_token, item_id = plaid_client.exchange_public_token(
                    public_token
                )
                if not institution:
                    institution = plaid_client.get_institution_name(access_token)
                items = _items_load(items_path)
                items = [i for i in items if i.get("item_id") != item_id]
                items.append(
                    {
                        "item_id": item_id,
                        "access_token": access_token,
                        "institution_name": institution,
                        "env": plaid_client.env_name,
                        "cursor": "",
                        "linked_at": int(time.time()),
                    }
                )
                _items_save(items_path, items)
                result.update(items[-1])
                resp = {"ok": True, "institution": institution}
            except Exception as e:  # noqa: BLE001
                resp = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            body = json.dumps(resp).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            if resp.get("ok"):
                done.set()

    server = HTTPServer(("127.0.0.1", port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{port}"
        if open_browser:
            webbrowser.open(url)
        if not done.wait(timeout=timeout_seconds):
            raise TimeoutError(
                f"Link flow timed out after {timeout_seconds}s. "
                f"Open {url} manually if the browser didn't launch."
            )
    finally:
        server.shutdown()
        server.server_close()

    return result
