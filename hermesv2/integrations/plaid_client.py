"""Thin wrapper around plaid-python.

Lazy-imports the SDK so the rest of Hermes runs without `pip install
plaid-python`. The wrapper exposes only the operations Hermes needs:
Link token creation, public-token exchange, transaction sync (cursor-based),
and the recurring-transactions stream feed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from hermesv2.config import PlaidToolSettings


class PlaidNotInstalled(RuntimeError):
    pass


def _require_sdk() -> Any:
    try:
        import plaid  # noqa: F401
        from plaid.api import plaid_api
        from plaid.api_client import ApiClient
        from plaid.configuration import Configuration
    except ImportError as e:
        raise PlaidNotInstalled(
            "plaid-python is not installed. Run: pip install -e \".[plaid]\""
        ) from e
    return plaid, plaid_api, ApiClient, Configuration


_HOSTS = {
    "sandbox": "https://sandbox.plaid.com",
    "development": "https://development.plaid.com",
    "production": "https://production.plaid.com",
}


@dataclass
class RecurringStream:
    stream_id: str
    merchant: str
    description: str
    average_amount: float
    last_amount: float
    currency: str
    frequency: str
    last_date: str | None
    predicted_next_date: str | None
    is_active: bool
    is_user_modified: bool


@dataclass
class SyncResult:
    added: int
    modified: int
    removed: int
    next_cursor: str
    added_transactions: list[dict[str, Any]]


class PlaidClient:
    def __init__(
        self,
        settings: PlaidToolSettings,
        client_id: str,
        secret: str,
        env: str | None = None,
    ) -> None:
        if not client_id or not secret:
            raise PlaidNotInstalled(
                "PLAID_CLIENT_ID and PLAID_SECRET must be set in the environment."
            )
        plaid, plaid_api, ApiClient, Configuration = _require_sdk()

        self.settings = settings
        env_name = (env or settings.env or "sandbox").lower()
        host = _HOSTS.get(env_name)
        if not host:
            raise ValueError(
                f"Unknown PLAID_ENV {env_name!r}; expected one of {sorted(_HOSTS)}."
            )
        self.env_name = env_name

        cfg = Configuration(
            host=host, api_key={"clientId": client_id, "secret": secret}
        )
        self._client = plaid_api.PlaidApi(ApiClient(cfg))

    # --- Link flow --------------------------------------------------------

    def create_link_token(self, client_user_id: str = "hermesv2-user") -> str:
        from plaid.model.country_code import CountryCode
        from plaid.model.link_token_create_request import LinkTokenCreateRequest
        from plaid.model.link_token_create_request_user import (
            LinkTokenCreateRequestUser,
        )
        from plaid.model.products import Products

        req = LinkTokenCreateRequest(
            products=[Products(p) for p in self.settings.products],
            client_name="Hermes v2",
            country_codes=[CountryCode(c) for c in self.settings.country_codes],
            language="en",
            user=LinkTokenCreateRequestUser(client_user_id=client_user_id),
        )
        resp = self._client.link_token_create(req)
        return resp["link_token"]

    def exchange_public_token(self, public_token: str) -> tuple[str, str]:
        from plaid.model.item_public_token_exchange_request import (
            ItemPublicTokenExchangeRequest,
        )

        resp = self._client.item_public_token_exchange(
            ItemPublicTokenExchangeRequest(public_token=public_token)
        )
        return resp["access_token"], resp["item_id"]

    def get_institution_name(self, access_token: str) -> str:
        from plaid.model.country_code import CountryCode
        from plaid.model.institutions_get_by_id_request import (
            InstitutionsGetByIdRequest,
        )
        from plaid.model.item_get_request import ItemGetRequest

        item_resp = self._client.item_get(ItemGetRequest(access_token=access_token))
        institution_id = item_resp["item"].get("institution_id")
        if not institution_id:
            return "(unknown institution)"
        inst_resp = self._client.institutions_get_by_id(
            InstitutionsGetByIdRequest(
                institution_id=institution_id,
                country_codes=[CountryCode(c) for c in self.settings.country_codes],
            )
        )
        return inst_resp["institution"].get("name") or "(unknown institution)"

    # --- Sync + recurring -------------------------------------------------

    def sync_transactions(
        self, access_token: str, cursor: str | None
    ) -> SyncResult:
        from plaid.model.transactions_sync_request import TransactionsSyncRequest

        added: list[dict[str, Any]] = []
        modified_ct = 0
        removed_ct = 0
        cur = cursor or ""
        while True:
            resp = self._client.transactions_sync(
                TransactionsSyncRequest(access_token=access_token, cursor=cur)
            )
            added.extend(_normalize_txn(t) for t in resp["added"])
            modified_ct += len(resp["modified"])
            removed_ct += len(resp["removed"])
            cur = resp["next_cursor"]
            if not resp["has_more"]:
                break
        return SyncResult(
            added=len(added),
            modified=modified_ct,
            removed=removed_ct,
            next_cursor=cur,
            added_transactions=added,
        )

    def recurring_streams(self, access_token: str) -> list[RecurringStream]:
        from plaid.model.accounts_get_request import AccountsGetRequest
        from plaid.model.transactions_recurring_get_request import (
            TransactionsRecurringGetRequest,
        )

        # Plaid requires account_ids; fetch them first.
        accounts = self._client.accounts_get(
            AccountsGetRequest(access_token=access_token)
        )
        account_ids = [a["account_id"] for a in accounts["accounts"]]
        if not account_ids:
            return []

        resp = self._client.transactions_recurring_get(
            TransactionsRecurringGetRequest(
                access_token=access_token, account_ids=account_ids
            )
        )
        # Outflows = money leaving the account (subscriptions / charges).
        return [_normalize_stream(s) for s in resp["outflow_streams"]]


# --- helpers ----------------------------------------------------------------


def _normalize_txn(t: Any) -> dict[str, Any]:
    return {
        "transaction_id": t["transaction_id"],
        "account_id": t["account_id"],
        "amount": float(t["amount"]),
        "iso_currency_code": t.get("iso_currency_code") or "USD",
        "merchant_name": t.get("merchant_name") or t.get("name") or "",
        "name": t.get("name") or "",
        "date": _date_str(t.get("date")),
        "pending": bool(t.get("pending", False)),
    }


def _normalize_stream(s: Any) -> RecurringStream:
    avg = s.get("average_amount") or {}
    last = s.get("last_amount") or {}
    return RecurringStream(
        stream_id=s["stream_id"],
        merchant=(s.get("merchant_name") or s.get("description") or "").strip(),
        description=s.get("description", ""),
        average_amount=float(avg.get("amount") or 0.0),
        last_amount=float(last.get("amount") or 0.0),
        currency=avg.get("iso_currency_code") or last.get("iso_currency_code") or "USD",
        frequency=str(s.get("frequency", "UNKNOWN")),
        last_date=_date_str(s.get("last_date")),
        predicted_next_date=_date_str(s.get("predicted_next_date")),
        is_active=bool(s.get("is_active", True)),
        is_user_modified=bool(s.get("is_user_modified", False)),
    )


def _date_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value.isoformat()
    return str(value)
