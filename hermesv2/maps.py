"""Inline terminal-map renderer for the /maps slash command.

Geocodes a query via OpenStreetMap Nominatim, fetches a single 256×256
PNG tile from tile.openstreetmap.org, and renders it as braille dots in
the terminal — one braille character = 2×4 pixels.

No API keys required; both endpoints are free with usage limits intended
for personal use. Falls back to a coords-only display when Pillow isn't
installed or the network is unreachable.

Dependencies:
    httpx (already in core)
    Pillow (gated behind the [maps] extra)
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Any

import httpx

USER_AGENT = "hermesv2/0.4 (+https://github.com/jamstand/hermesv2.1-work-)"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
PHOTON_URL    = "https://photon.komoot.io/api/"
TILE_URL      = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"


@dataclass
class GeocodeResult:
    display_name: str
    lat: float
    lon: float
    bbox: tuple[float, float, float, float] | None = None  # south, north, west, east


@dataclass
class GeocodeAttempt:
    """Outcome of one provider's geocode attempt."""
    provider: str
    result: GeocodeResult | None = None
    reason: str = "ok"     # ok | no_result | timeout | http_error | network_error
    detail: str = ""


def _geocode_nominatim(query: str, timeout: float = 12.0) -> GeocodeAttempt:
    try:
        r = httpx.get(
            NOMINATIM_URL,
            params={"format": "json", "q": query, "limit": 1, "addressdetails": 0},
            headers={"User-Agent": USER_AGENT},
            timeout=timeout,
        )
    except httpx.TimeoutException as e:
        return GeocodeAttempt("nominatim", reason="timeout", detail=str(e))
    except httpx.RequestError as e:
        return GeocodeAttempt("nominatim", reason="network_error", detail=str(e))
    if r.status_code != 200:
        return GeocodeAttempt("nominatim", reason="http_error", detail=f"HTTP {r.status_code}")
    try:
        data = r.json()
    except ValueError as e:
        return GeocodeAttempt("nominatim", reason="http_error", detail=f"bad JSON: {e}")
    if not data:
        return GeocodeAttempt("nominatim", reason="no_result")

    item = data[0]
    bbox: tuple[float, float, float, float] | None = None
    if isinstance(item.get("boundingbox"), list) and len(item["boundingbox"]) == 4:
        try:
            s, n, w, e = (float(x) for x in item["boundingbox"])
            bbox = (s, n, w, e)
        except (TypeError, ValueError):
            pass
    return GeocodeAttempt(
        "nominatim",
        result=GeocodeResult(
            display_name=item.get("display_name", query),
            lat=float(item["lat"]),
            lon=float(item["lon"]),
            bbox=bbox,
        ),
    )


def _geocode_photon(query: str, timeout: float = 12.0) -> GeocodeAttempt:
    try:
        r = httpx.get(
            PHOTON_URL,
            params={"q": query, "limit": 1},
            headers={"User-Agent": USER_AGENT},
            timeout=timeout,
        )
    except httpx.TimeoutException as e:
        return GeocodeAttempt("photon", reason="timeout", detail=str(e))
    except httpx.RequestError as e:
        return GeocodeAttempt("photon", reason="network_error", detail=str(e))
    if r.status_code != 200:
        return GeocodeAttempt("photon", reason="http_error", detail=f"HTTP {r.status_code}")
    try:
        data = r.json()
    except ValueError as e:
        return GeocodeAttempt("photon", reason="http_error", detail=f"bad JSON: {e}")
    feats = data.get("features") or []
    if not feats:
        return GeocodeAttempt("photon", reason="no_result")
    feat = feats[0]
    coords = (feat.get("geometry") or {}).get("coordinates")  # [lon, lat]
    if not coords or len(coords) < 2:
        return GeocodeAttempt("photon", reason="no_result")
    props = feat.get("properties") or {}
    parts = [props.get("name"), props.get("street"), props.get("city"),
             props.get("state"), props.get("country")]
    display_name = ", ".join(p for p in parts if p) or query
    return GeocodeAttempt(
        "photon",
        result=GeocodeResult(
            display_name=display_name,
            lat=float(coords[1]),
            lon=float(coords[0]),
        ),
    )


def geocode(query: str) -> tuple[GeocodeResult | None, list[GeocodeAttempt]]:
    """Try Nominatim, then Photon. Returns (result_or_None, list_of_attempts).

    Callers can inspect the attempts list to surface a useful error
    (e.g. "no_result everywhere" vs "network_error" → likely proxy block).
    """
    attempts: list[GeocodeAttempt] = []
    for fn in (_geocode_nominatim, _geocode_photon):
        att = fn(query)
        attempts.append(att)
        if att.result is not None:
            return att.result, attempts
    return None, attempts


def latlon_to_tile(lat: float, lon: float, zoom: int) -> tuple[int, int]:
    """Slippy-map tile xy from lat/lon."""
    lat_rad = math.radians(lat)
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def fetch_tile(x: int, y: int, z: int, timeout: float = 12.0) -> bytes | None:
    try:
        r = httpx.get(
            TILE_URL.format(z=z, x=x, y=y),
            headers={"User-Agent": USER_AGENT},
            timeout=timeout,
        )
    except httpx.RequestError:
        return None
    if r.status_code == 200 and r.content:
        return r.content
    return None


# Braille dot positions in a 2×4 cell, mapped to bit masks of U+2800.
#   1 (0x01)  4 (0x08)
#   2 (0x02)  5 (0x10)
#   3 (0x04)  6 (0x20)
#   7 (0x40)  8 (0x80)
_BRAILLE_OFFSETS = (
    (0, 0, 0x01), (0, 1, 0x02), (0, 2, 0x04), (0, 3, 0x40),
    (1, 0, 0x08), (1, 1, 0x10), (1, 2, 0x20), (1, 3, 0x80),
)


def png_to_braille(
    png_bytes: bytes,
    cols: int = 60,
    rows: int = 18,
    threshold: int = 190,
) -> str | None:
    """Render a PNG as a `rows × cols` braille string.

    OSM tiles have light cream backgrounds and darker features (roads,
    building outlines, labels). `threshold` is the grayscale cutoff below
    which a pixel is treated as "lit" in the braille char.

    Returns None if Pillow isn't available.
    """
    try:
        from PIL import Image
    except ImportError:
        return None
    img = Image.open(io.BytesIO(png_bytes)).convert("L")
    target_w = cols * 2
    target_h = rows * 4
    img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
    px = img.load()

    lines: list[str] = []
    for cy in range(rows):
        chars: list[str] = []
        for cx in range(cols):
            base = 0x2800
            for dx, dy, bit in _BRAILLE_OFFSETS:
                v = px[cx * 2 + dx, cy * 4 + dy]
                if v < threshold:
                    base |= bit
            chars.append(chr(base))
        lines.append("".join(chars))
    return "\n".join(lines)


def render_map(query: str, zoom: int = 14, cols: int = 60, rows: int = 18) -> dict[str, Any]:
    """All-in-one: geocode → tile fetch → braille render.

    Returns a dict with keys: place, lat, lon, braille, error. Any may be None.
    `error` is a single human-readable line when something went wrong.
    """
    result: dict[str, Any] = {
        "place": None, "lat": None, "lon": None, "braille": None, "error": None,
    }
    geo, attempts = geocode(query)
    if geo is None:
        # Pick the most informative attempt to surface.
        reasons = [a.reason for a in attempts]
        if all(r == "no_result" for r in reasons):
            result["error"] = f"no place found matching '{query}'"
        elif any(r in ("network_error", "timeout") for r in reasons):
            providers = ", ".join(a.provider for a in attempts)
            result["error"] = (
                f"geocoding blocked by your network ({providers} unreachable). "
                "Browser + agent lookup below still work."
            )
        else:
            details = "; ".join(f"{a.provider}: {a.reason}" for a in attempts)
            result["error"] = f"geocoding failed ({details})"
        return result

    result["place"] = geo.display_name
    result["lat"] = geo.lat
    result["lon"] = geo.lon

    tx, ty = latlon_to_tile(geo.lat, geo.lon, zoom)
    png = fetch_tile(tx, ty, zoom)
    if png is None:
        result["error"] = "map tile fetch failed (network or tile.openstreetmap.org blocked)"
        return result

    braille = png_to_braille(png, cols=cols, rows=rows)
    if braille is None:
        result["error"] = "Pillow not installed; run `pip install 'hermesv2[maps]'`"
        return result

    result["braille"] = braille
    return result


def can_reach(url: str, timeout: float = 5.0) -> bool:
    """Cheap network probe used by `hermesv2 doctor`."""
    try:
        r = httpx.head(url, headers={"User-Agent": USER_AGENT}, timeout=timeout, follow_redirects=True)
    except httpx.RequestError:
        return False
    return r.status_code < 500
