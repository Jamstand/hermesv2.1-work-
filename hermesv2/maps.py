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
TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"


@dataclass
class GeocodeResult:
    display_name: str
    lat: float
    lon: float
    bbox: tuple[float, float, float, float] | None = None  # south, north, west, east


def geocode(query: str, timeout: float = 12.0) -> GeocodeResult | None:
    """Return the first Nominatim match for `query`, or None on failure."""
    try:
        r = httpx.get(
            NOMINATIM_URL,
            params={"format": "json", "q": query, "limit": 1, "addressdetails": 0},
            headers={"User-Agent": USER_AGENT},
            timeout=timeout,
        )
    except httpx.RequestError:
        return None
    if r.status_code != 200:
        return None
    try:
        data = r.json()
    except ValueError:
        return None
    if not data:
        return None
    item = data[0]
    bbox: tuple[float, float, float, float] | None = None
    if isinstance(item.get("boundingbox"), list) and len(item["boundingbox"]) == 4:
        try:
            s, n, w, e = (float(x) for x in item["boundingbox"])
            bbox = (s, n, w, e)
        except (TypeError, ValueError):
            pass
    return GeocodeResult(
        display_name=item.get("display_name", query),
        lat=float(item["lat"]),
        lon=float(item["lon"]),
        bbox=bbox,
    )


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
    """
    result: dict[str, Any] = {
        "place": None, "lat": None, "lon": None, "braille": None, "error": None,
    }
    geo = geocode(query)
    if geo is None:
        result["error"] = "geocoding failed (no result, or network blocked)"
        return result
    result["place"] = geo.display_name
    result["lat"] = geo.lat
    result["lon"] = geo.lon

    tx, ty = latlon_to_tile(geo.lat, geo.lon, zoom)
    png = fetch_tile(tx, ty, zoom)
    if png is None:
        result["error"] = "map tile fetch failed"
        return result

    braille = png_to_braille(png, cols=cols, rows=rows)
    if braille is None:
        result["error"] = "Pillow not installed; run `pip install 'hermesv2[maps]'`"
        return result

    result["braille"] = braille
    return result
