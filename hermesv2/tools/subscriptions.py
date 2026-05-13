"""Subscription manager: track recurring charges (streaming, SaaS, etc.) in a
local JSON store. Exposes tools the agent can call to add, list, update, remove,
and summarize subscriptions, plus surface upcoming renewals.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from hermesv2.agent import Tool
from hermesv2.config import SubscriptionsToolSettings

CYCLES = {"weekly": 7, "monthly": 30, "quarterly": 91, "yearly": 365}


@dataclass
class Subscription:
    id: str
    name: str
    cost: float
    currency: str = "USD"
    billing_cycle: str = "monthly"
    next_renewal: str | None = None  # ISO date YYYY-MM-DD
    category: str = ""
    notes: str = ""
    active: bool = True
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


def _load(store_path: Path) -> list[dict[str, Any]]:
    if not store_path.is_file():
        return []
    try:
        data = json.loads(store_path.read_text() or "[]")
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def load_subscriptions(store_path: str | Path) -> list[dict[str, Any]]:
    """Public reader for other modules (e.g. Plaid reconciliation)."""
    return _load(Path(str(store_path)).expanduser())


def _save(store_path: Path, subs: list[dict[str, Any]]) -> None:
    store_path.parent.mkdir(parents=True, exist_ok=True)
    store_path.write_text(json.dumps(subs, indent=2))


def _find(subs: list[dict[str, Any]], identifier: str) -> dict[str, Any] | None:
    ident = identifier.strip().lower()
    for s in subs:
        if s["id"] == identifier or s["name"].lower() == ident:
            return s
    return None


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _monthly_equivalent(cost: float, cycle: str) -> float:
    days = CYCLES.get(cycle, 30)
    return cost * (30 / days)


def _format_row(s: dict[str, Any]) -> str:
    flag = "" if s.get("active", True) else " [inactive]"
    renewal = s.get("next_renewal") or "—"
    return (
        f"{s['name']}{flag} — {s['cost']:.2f} {s['currency']}/{s['billing_cycle']}"
        f" | next: {renewal} | id: {s['id'][:8]}"
    )


def build_subscription_tools(settings: SubscriptionsToolSettings) -> list[Tool]:
    store_path = Path(settings.store_path).expanduser()

    def add_subscription(args: dict[str, Any]) -> str:
        name = str(args["name"]).strip()
        if not name:
            return "Error: name is required."
        cycle = str(args.get("billing_cycle", "monthly")).lower()
        if cycle not in CYCLES:
            return f"Error: billing_cycle must be one of {sorted(CYCLES)}."
        renewal = args.get("next_renewal")
        if renewal and not _parse_date(str(renewal)):
            return "Error: next_renewal must be ISO date (YYYY-MM-DD)."

        subs = _load(store_path)
        if _find(subs, name):
            return f"Subscription {name!r} already exists. Use update_subscription instead."

        sub = Subscription(
            id=uuid.uuid4().hex,
            name=name,
            cost=float(args["cost"]),
            currency=str(args.get("currency", "USD")).upper(),
            billing_cycle=cycle,
            next_renewal=str(renewal) if renewal else None,
            category=str(args.get("category", "")),
            notes=str(args.get("notes", "")),
        )
        subs.append(asdict(sub))
        _save(store_path, subs)
        return f"Added {name} ({sub.cost:.2f} {sub.currency}/{cycle}). id={sub.id[:8]}"

    def list_subscriptions(args: dict[str, Any]) -> str:
        subs = _load(store_path)
        category = str(args.get("category", "")).strip().lower()
        include_inactive = bool(args.get("include_inactive", False))

        rows = [
            s
            for s in subs
            if (include_inactive or s.get("active", True))
            and (not category or s.get("category", "").lower() == category)
        ]
        if not rows:
            return "(no matching subscriptions)"
        rows.sort(key=lambda s: s["name"].lower())
        return "\n".join(_format_row(s) for s in rows)

    def update_subscription(args: dict[str, Any]) -> str:
        subs = _load(store_path)
        sub = _find(subs, str(args["identifier"]))
        if sub is None:
            return f"No subscription matching {args['identifier']!r}."

        for field_name in ("name", "currency", "category", "notes"):
            if field_name in args:
                sub[field_name] = str(args[field_name])
        if "cost" in args:
            sub["cost"] = float(args["cost"])
        if "billing_cycle" in args:
            cycle = str(args["billing_cycle"]).lower()
            if cycle not in CYCLES:
                return f"Error: billing_cycle must be one of {sorted(CYCLES)}."
            sub["billing_cycle"] = cycle
        if "next_renewal" in args:
            value = args["next_renewal"]
            if value and not _parse_date(str(value)):
                return "Error: next_renewal must be ISO date (YYYY-MM-DD)."
            sub["next_renewal"] = str(value) if value else None
        if "active" in args:
            sub["active"] = bool(args["active"])

        _save(store_path, subs)
        return f"Updated {sub['name']}: {_format_row(sub)}"

    def remove_subscription(args: dict[str, Any]) -> str:
        subs = _load(store_path)
        sub = _find(subs, str(args["identifier"]))
        if sub is None:
            return f"No subscription matching {args['identifier']!r}."
        subs = [s for s in subs if s["id"] != sub["id"]]
        _save(store_path, subs)
        return f"Removed {sub['name']}."

    def subscription_summary(args: dict[str, Any]) -> str:
        subs = [s for s in _load(store_path) if s.get("active", True)]
        if not subs:
            return "(no active subscriptions)"

        by_currency: dict[str, float] = {}
        by_category: dict[str, float] = {}
        for s in subs:
            monthly = _monthly_equivalent(s["cost"], s["billing_cycle"])
            cur = s.get("currency", "USD")
            by_currency[cur] = by_currency.get(cur, 0.0) + monthly
            cat = s.get("category") or "uncategorized"
            by_category[cat] = by_category.get(cat, 0.0) + monthly

        lines = [f"Active subscriptions: {len(subs)}", "Monthly equivalent:"]
        for cur, total in sorted(by_currency.items()):
            lines.append(f"  {cur}: {total:.2f}/mo (≈ {total * 12:.2f}/yr)")
        lines.append("By category (monthly):")
        for cat, total in sorted(by_category.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {cat}: {total:.2f}")
        return "\n".join(lines)

    def upcoming_renewals(args: dict[str, Any]) -> str:
        days = int(args.get("days", 30))
        today = date.today()
        cutoff = today + timedelta(days=days)

        upcoming: list[tuple[date, dict[str, Any]]] = []
        for s in _load(store_path):
            if not s.get("active", True):
                continue
            renewal = _parse_date(s.get("next_renewal"))
            if renewal and today <= renewal <= cutoff:
                upcoming.append((renewal, s))
        if not upcoming:
            return f"(no renewals in the next {days} days)"

        upcoming.sort(key=lambda pair: pair[0])
        lines = [f"Renewals in the next {days} days:"]
        for renewal, s in upcoming:
            delta = (renewal - today).days
            when = "today" if delta == 0 else f"in {delta}d"
            lines.append(
                f"  {renewal.isoformat()} ({when}) — {s['name']}: "
                f"{s['cost']:.2f} {s['currency']}"
            )
        return "\n".join(lines)

    return [
        Tool(
            name="add_subscription",
            description=(
                "Add a recurring subscription (Netflix, SaaS, magazines, etc.) "
                "to the local store. Use update_subscription to change an "
                "existing one instead of re-adding."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "cost": {"type": "number", "description": "Price per billing cycle."},
                    "currency": {"type": "string", "description": "ISO code, default USD."},
                    "billing_cycle": {
                        "type": "string",
                        "enum": sorted(CYCLES),
                        "description": "Default monthly.",
                    },
                    "next_renewal": {
                        "type": "string",
                        "description": "ISO date YYYY-MM-DD of the next charge.",
                    },
                    "category": {"type": "string"},
                    "notes": {"type": "string"},
                },
                "required": ["name", "cost"],
            },
            handler=add_subscription,
        ),
        Tool(
            name="list_subscriptions",
            description=(
                "List tracked subscriptions. Filter by category, and pass "
                "include_inactive=true to also see paused/cancelled ones."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "category": {"type": "string"},
                    "include_inactive": {"type": "boolean"},
                },
            },
            handler=list_subscriptions,
        ),
        Tool(
            name="update_subscription",
            description=(
                "Update fields on an existing subscription, matched by name or id. "
                "Only provided fields change; set active=false to mark cancelled."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "identifier": {
                        "type": "string",
                        "description": "Subscription name or id.",
                    },
                    "name": {"type": "string"},
                    "cost": {"type": "number"},
                    "currency": {"type": "string"},
                    "billing_cycle": {"type": "string", "enum": sorted(CYCLES)},
                    "next_renewal": {"type": "string"},
                    "category": {"type": "string"},
                    "notes": {"type": "string"},
                    "active": {"type": "boolean"},
                },
                "required": ["identifier"],
            },
            handler=update_subscription,
        ),
        Tool(
            name="remove_subscription",
            description=(
                "Permanently delete a subscription. Confirm with the user before "
                "calling; prefer update_subscription with active=false to keep history."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "identifier": {"type": "string", "description": "Name or id."},
                },
                "required": ["identifier"],
            },
            handler=remove_subscription,
        ),
        Tool(
            name="subscription_summary",
            description=(
                "Summarize active subscriptions: count and monthly-equivalent "
                "totals by currency and by category."
            ),
            input_schema={"type": "object", "properties": {}},
            handler=subscription_summary,
        ),
        Tool(
            name="upcoming_renewals",
            description="List active subscriptions renewing within N days (default 30).",
            input_schema={
                "type": "object",
                "properties": {
                    "days": {"type": "integer", "description": "Window in days. Default 30."}
                },
            },
            handler=upcoming_renewals,
        ),
    ]
