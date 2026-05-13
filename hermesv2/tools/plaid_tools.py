"""Plaid-backed billing tools.

Exposes agent tools to inspect linked institutions, sync transactions, list
recurring subscription streams Plaid detects from real transactions, and
reconcile those against the manual subscription tracker.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from datetime import date
from pathlib import Path
from typing import Any

from hermesv2.agent import Tool
from hermesv2.config import PlaidToolSettings
from hermesv2.integrations.plaid_client import (
    PlaidClient,
    PlaidNotInstalled,
    RecurringStream,
)
from hermesv2.tools.subscriptions import load_subscriptions

FREQ_TO_MONTHLY = {
    "WEEKLY": 4.345,
    "BIWEEKLY": 2.1725,
    "SEMI_MONTHLY": 2.0,
    "MONTHLY": 1.0,
    "QUARTERLY": 1 / 3,
    "SEMI_ANNUALLY": 1 / 6,
    "ANNUALLY": 1 / 12,
    "UNKNOWN": 1.0,
}


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


def _txns_load(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text() or "{}")
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _txns_save(path: Path, cache: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2))


def _client_or_error(
    settings: PlaidToolSettings, creds: dict[str, str]
) -> PlaidClient:
    return PlaidClient(
        settings,
        client_id=creds.get("client_id", ""),
        secret=creds.get("secret", ""),
        env=creds.get("env") or settings.env,
    )


def _format_stream(s: RecurringStream) -> str:
    flag = "" if s.is_active else " [inactive]"
    next_date = s.predicted_next_date or "—"
    return (
        f"{s.merchant or s.description}{flag} — "
        f"{s.last_amount:.2f} {s.currency}/{s.frequency.lower()} "
        f"| next: {next_date} | id: {s.stream_id[:8]}"
    )


def _name_match(plaid_name: str, manual_name: str) -> bool:
    a = plaid_name.lower().strip()
    b = manual_name.lower().strip()
    if not a or not b:
        return False
    return a in b or b in a


def build_plaid_tools(
    settings: PlaidToolSettings,
    creds: dict[str, str],
    subscriptions_store_path: str,
) -> list[Tool]:
    items_path = Path(settings.items_path).expanduser()
    cache_path = Path(settings.cache_path).expanduser()

    def _wrap(fn):
        def runner(args: dict[str, Any]) -> str:
            try:
                return fn(args)
            except PlaidNotInstalled as e:
                return f"Plaid unavailable: {e}"
            except Exception as e:  # noqa: BLE001 — surface for the agent
                return f"Plaid error: {type(e).__name__}: {e}"

        return runner

    def list_items(args: dict[str, Any]) -> str:
        items = _items_load(items_path)
        if not items:
            return (
                "No linked institutions. Run `hermesv2 plaid link` to connect "
                "a bank or card."
            )
        lines = []
        for it in items:
            cur = "(no sync yet)" if not it.get("cursor") else f"cursor={it['cursor'][:12]}…"
            lines.append(
                f"{it.get('institution_name', '?')} [{it.get('env', '?')}] "
                f"id={it['item_id'][:8]} {cur}"
            )
        return "\n".join(lines)

    def sync_transactions(args: dict[str, Any]) -> str:
        client = _client_or_error(settings, creds)
        items = _items_load(items_path)
        if not items:
            return "No linked institutions; nothing to sync."

        cache = _txns_load(cache_path)
        summary = []
        for it in items:
            result = client.sync_transactions(it["access_token"], it.get("cursor") or None)
            for txn in result.added_transactions:
                txn["institution"] = it.get("institution_name", "")
                cache[txn["transaction_id"]] = txn
            it["cursor"] = result.next_cursor
            it["last_synced_at"] = int(time.time())
            summary.append(
                f"{it.get('institution_name', '?')}: added={result.added}, "
                f"modified={result.modified}, removed={result.removed}"
            )
        _items_save(items_path, items)
        _txns_save(cache_path, cache)
        return "\n".join(summary)

    def recent_transactions(args: dict[str, Any]) -> str:
        days = int(args.get("days", 30))
        limit = int(args.get("limit", 25))
        cache = _txns_load(cache_path)
        cutoff = date.today().toordinal() - days
        rows: list[dict[str, Any]] = []
        for txn in cache.values():
            d = txn.get("date")
            if not d:
                continue
            try:
                if date.fromisoformat(d).toordinal() < cutoff:
                    continue
            except ValueError:
                continue
            rows.append(txn)
        rows.sort(key=lambda t: t.get("date", ""), reverse=True)
        rows = rows[:limit]
        if not rows:
            return f"(no cached transactions in the last {days} days; run plaid_sync_transactions first)"
        lines = []
        for t in rows:
            sign = "-" if t["amount"] > 0 else "+"
            lines.append(
                f"{t['date']} {sign}{abs(t['amount']):.2f} {t['iso_currency_code']} "
                f"{t['merchant_name'] or t['name']} [{t.get('institution', '?')}]"
            )
        return "\n".join(lines)

    def recurring_subscriptions(args: dict[str, Any]) -> str:
        active_only = bool(args.get("active_only", True))
        min_amount = float(args.get("min_amount", 0))
        client = _client_or_error(settings, creds)
        items = _items_load(items_path)
        if not items:
            return "No linked institutions; nothing to inspect."

        all_streams: list[tuple[str, RecurringStream]] = []
        for it in items:
            for s in client.recurring_streams(it["access_token"]):
                if active_only and not s.is_active:
                    continue
                if abs(s.last_amount) < min_amount:
                    continue
                all_streams.append((it.get("institution_name", "?"), s))
        if not all_streams:
            return "(no recurring streams found)"
        all_streams.sort(key=lambda pair: -abs(pair[1].last_amount))
        lines = ["Recurring outflows detected by Plaid:"]
        for inst, s in all_streams:
            lines.append(f"  [{inst}] {_format_stream(s)}")

        monthly = sum(
            abs(s.last_amount) * FREQ_TO_MONTHLY.get(s.frequency, 1.0)
            for _, s in all_streams
        )
        lines.append(f"Estimated total: {monthly:.2f}/mo (≈ {monthly * 12:.2f}/yr)")
        return "\n".join(lines)

    def reconcile(args: dict[str, Any]) -> str:
        client = _client_or_error(settings, creds)
        items = _items_load(items_path)
        if not items:
            return "No linked institutions; link one with `hermesv2 plaid link` first."

        plaid_streams: list[RecurringStream] = []
        for it in items:
            plaid_streams.extend(
                s
                for s in client.recurring_streams(it["access_token"])
                if s.is_active
            )

        manual = [
            s for s in load_subscriptions(subscriptions_store_path) if s.get("active", True)
        ]

        matched: list[str] = []
        manual_only: list[str] = []
        plaid_only: list[RecurringStream] = list(plaid_streams)

        for m in manual:
            hit = next(
                (s for s in plaid_only if _name_match(s.merchant or s.description, m["name"])),
                None,
            )
            if hit:
                plaid_only.remove(hit)
                diff = abs(hit.last_amount) - float(m["cost"])
                note = ""
                if abs(diff) >= 0.01:
                    note = f" (Δ {diff:+.2f} vs tracker)"
                matched.append(
                    f"  {m['name']} ↔ {hit.merchant}: "
                    f"{hit.last_amount:.2f} {hit.currency}/{hit.frequency.lower()}{note}"
                )
            else:
                manual_only.append(
                    f"  {m['name']} ({m['cost']:.2f} {m['currency']}/{m['billing_cycle']})"
                )

        lines = [f"Manual: {len(manual)} active, Plaid: {len(plaid_streams)} active recurring"]
        lines.append(f"Matched ({len(matched)}):")
        lines.extend(matched or ["  (none)"])
        lines.append(f"Tracked but not seen on card ({len(manual_only)}):")
        lines.extend(manual_only or ["  (none)"])
        lines.append(f"Untracked recurring charges ({len(plaid_only)}):")
        if plaid_only:
            lines.extend(f"  {_format_stream(s)}" for s in plaid_only)
        else:
            lines.append("  (none)")
        return "\n".join(lines)

    def remove_item(args: dict[str, Any]) -> str:
        ident = str(args["identifier"]).lower().strip()
        items = _items_load(items_path)
        match = next(
            (
                i
                for i in items
                if i["item_id"] == ident or i.get("institution_name", "").lower() == ident
            ),
            None,
        )
        if match is None:
            return f"No linked item matching {args['identifier']!r}."
        kept = [i for i in items if i["item_id"] != match["item_id"]]
        _items_save(items_path, kept)
        return (
            f"Removed local access_token for {match.get('institution_name', '?')}. "
            "Note: this only forgets the token here; revoke server-side via Plaid Dashboard."
        )

    return [
        Tool(
            name="plaid_list_items",
            description="List bank/card institutions linked via Plaid.",
            input_schema={"type": "object", "properties": {}},
            handler=_wrap(list_items),
        ),
        Tool(
            name="plaid_sync_transactions",
            description=(
                "Pull new/modified/removed transactions from Plaid for every "
                "linked item, update the local cache, and return a per-item summary."
            ),
            input_schema={"type": "object", "properties": {}},
            handler=_wrap(sync_transactions),
        ),
        Tool(
            name="plaid_recent_transactions",
            description=(
                "List cached Plaid transactions from the last N days (default 30). "
                "Run plaid_sync_transactions first to refresh the cache."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "days": {"type": "integer", "description": "Lookback window. Default 30."},
                    "limit": {"type": "integer", "description": "Max rows. Default 25."},
                },
            },
            handler=_wrap(recent_transactions),
        ),
        Tool(
            name="plaid_recurring_subscriptions",
            description=(
                "List recurring outflows Plaid detected from real card/bank "
                "transactions (Netflix, gym, SaaS, etc.). Use this to see what "
                "you're actually being billed for."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "min_amount": {
                        "type": "number",
                        "description": "Hide streams below this amount. Default 0.",
                    },
                    "active_only": {
                        "type": "boolean",
                        "description": "Hide streams Plaid marks inactive. Default true.",
                    },
                },
            },
            handler=_wrap(recurring_subscriptions),
        ),
        Tool(
            name="plaid_reconcile_subscriptions",
            description=(
                "Compare the manual subscription tracker against Plaid's "
                "real recurring charges. Reports matches (with cost diffs), "
                "tracker entries with no recent charge, and untracked recurring "
                "charges Plaid found."
            ),
            input_schema={"type": "object", "properties": {}},
            handler=_wrap(reconcile),
        ),
        Tool(
            name="plaid_remove_item",
            description=(
                "Forget a linked institution locally (deletes its access_token "
                "from the items file). Confirm with the user before calling. "
                "Does NOT revoke server-side — do that in the Plaid Dashboard."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "identifier": {
                        "type": "string",
                        "description": "Institution name or item_id.",
                    }
                },
                "required": ["identifier"],
            },
            handler=_wrap(remove_item),
        ),
    ]
