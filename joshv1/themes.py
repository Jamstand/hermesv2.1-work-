"""Themes for joshv1.

Each theme is a palette of semantic color slots. They map to Rich style
names (`josh.*` and `markdown.*`) via build_theme(), so swapping a theme
swaps every color in the TUI without touching the code that emits styled
text.

Built-in palettes are listed in PALETTES. The active palette is persisted
in ~/.josh-memory/theme so the choice survives restarts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rich.theme import Theme


@dataclass(frozen=True)
class Palette:
    """A complete theme palette.

    Slot meanings (mapped to josh.* style names downstream):
      primary    — banner, panel border, panel title, primary labels
      secondary  — prompt chevron, josh-label, spinner, **bold** highlights
      warm       — logo, italic emphasis
      highlight  — session ID, current marker, slash-command names
      success    — section headers ("Available Tools"), ok states
      info       — model name, config values, tool-call labels, links
      info2      — session marker in status bar, h3 sub-headers
      error      — errors
      text       — default body text
      dim        — subtle / metadata text
      bg         — primary dark background
      bg_alt     — darker bg used for dropdown menus + inline code
    """

    name: str
    label: str
    primary: str
    secondary: str
    warm: str
    highlight: str
    success: str
    info: str
    info2: str
    error: str
    text: str
    dim: str
    bg: str
    bg_alt: str
    code_accent: str = ""  # optional override for inline code background


PALETTES: dict[str, Palette] = {
    "catppuccin-mocha": Palette(
        name="catppuccin-mocha",
        label="Catppuccin Mocha · warm pastels on dark navy",
        primary="#cba6f7", secondary="#f5c2e7", warm="#fab387",
        highlight="#f9e2af", success="#a6e3a1", info="#89dceb", info2="#b4befe",
        error="#f38ba8", text="#cdd6f4", dim="#a6adc8",
        bg="#1e1e2e", bg_alt="#181825",
    ),
    "tokyo-night": Palette(
        name="tokyo-night",
        label="Tokyo Night · cool blues, purple highlights",
        primary="#7aa2f7", secondary="#bb9af7", warm="#ff9e64",
        highlight="#e0af68", success="#9ece6a", info="#7dcfff", info2="#b4f9f8",
        error="#f7768e", text="#c0caf5", dim="#565f89",
        bg="#1a1b26", bg_alt="#15161e",
    ),
    "dracula": Palette(
        name="dracula",
        label="Dracula · purple/pink/cyan on midnight",
        primary="#bd93f9", secondary="#ff79c6", warm="#ffb86c",
        highlight="#f1fa8c", success="#50fa7b", info="#8be9fd", info2="#bd93f9",
        error="#ff5555", text="#f8f8f2", dim="#6272a4",
        bg="#282a36", bg_alt="#1f2029",
    ),
    "nord": Palette(
        name="nord",
        label="Nord · cool teal/blue/gray, minimalist",
        primary="#88c0d0", secondary="#b48ead", warm="#d08770",
        highlight="#ebcb8b", success="#a3be8c", info="#81a1c1", info2="#8fbcbb",
        error="#bf616a", text="#eceff4", dim="#4c566a",
        bg="#2e3440", bg_alt="#272c39",
    ),
    "gruvbox": Palette(
        name="gruvbox",
        label="Gruvbox Dark · warm earth tones, retro",
        primary="#d3869b", secondary="#fb4934", warm="#fe8019",
        highlight="#fabd2f", success="#b8bb26", info="#83a598", info2="#8ec07c",
        error="#cc241d", text="#ebdbb2", dim="#928374",
        bg="#282828", bg_alt="#1d2021",
    ),
    "rose-pine": Palette(
        name="rose-pine",
        label="Rose Pine · muted rose, gold, teal",
        primary="#c4a7e7", secondary="#eb6f92", warm="#f6c177",
        highlight="#ebbcba", success="#9ccfd8", info="#31748f", info2="#9ccfd8",
        error="#eb6f92", text="#e0def4", dim="#6e6a86",
        bg="#191724", bg_alt="#1f1d2e",
    ),
    "monokai": Palette(
        name="monokai",
        label="Monokai · classic vibrant, pink/purple/green",
        primary="#f92672", secondary="#ae81ff", warm="#fd971f",
        highlight="#e6db74", success="#a6e22e", info="#66d9ef", info2="#ae81ff",
        error="#f92672", text="#f8f8f2", dim="#75715e",
        bg="#272822", bg_alt="#1e1f1c",
    ),
    "solarized-dark": Palette(
        name="solarized-dark",
        label="Solarized Dark · calm cyan/blue on deep teal",
        primary="#268bd2", secondary="#d33682", warm="#cb4b16",
        highlight="#b58900", success="#859900", info="#2aa198", info2="#6c71c4",
        error="#dc322f", text="#93a1a1", dim="#586e75",
        bg="#002b36", bg_alt="#073642",
    ),
    "everforest": Palette(
        name="everforest",
        label="Everforest · natural greens and earth tones",
        primary="#a7c080", secondary="#d699b6", warm="#e69875",
        highlight="#dbbc7f", success="#83c092", info="#7fbbb3", info2="#d699b6",
        error="#e67e80", text="#d3c6aa", dim="#859289",
        bg="#2d353b", bg_alt="#232a2e",
    ),
    "synthwave": Palette(
        name="synthwave",
        label="Synthwave · neon retro pink/cyan/yellow",
        primary="#ff7edb", secondary="#36f9f6", warm="#fede5d",
        highlight="#fede5d", success="#72f1b8", info="#36f9f6", info2="#b893ce",
        error="#fe4450", text="#f2f3f7", dim="#5a5475",
        bg="#1a1727", bg_alt="#14111c",
    ),
}

DEFAULT_THEME = "catppuccin-mocha"
_THEME_FILE = Path.home() / ".josh-memory" / "theme"


def list_themes() -> list[Palette]:
    return list(PALETTES.values())


def get_palette(name: str) -> Palette:
    return PALETTES.get(name, PALETTES[DEFAULT_THEME])


def load_active_palette() -> Palette:
    """Read the persisted theme name, falling back to the default."""
    try:
        name = _THEME_FILE.read_text().strip()
    except OSError:
        return PALETTES[DEFAULT_THEME]
    return get_palette(name)


def save_active_theme(name: str) -> None:
    """Persist the user's theme choice to disk."""
    if name not in PALETTES:
        raise ValueError(f"unknown theme: {name}")
    _THEME_FILE.parent.mkdir(parents=True, exist_ok=True)
    _THEME_FILE.write_text(name)


def build_theme(p: Palette) -> Theme:
    """Map a palette to a Rich Theme with both josh.* and markdown.* styles."""
    return Theme({
        # Josh-specific semantic styles. tui.py + cli.py use these names
        # instead of hex codes, so a theme swap re-paints the entire UI.
        "josh.banner":         f"bold {p.primary}",
        "josh.title":          f"bold {p.primary}",
        "josh.label.you":      f"bold {p.primary}",
        "josh.label.josh":     f"bold {p.secondary}",
        "josh.chevron":        f"bold {p.secondary}",
        "josh.bar":            f"bold {p.primary}",
        "josh.spinner":        f"bold {p.secondary}",
        "josh.logo":           p.warm,
        "josh.session":        f"bold {p.highlight}",
        "josh.session.marker": f"bold {p.info2}",
        "josh.section":        f"bold {p.success}",
        "josh.success":        p.success,
        "josh.info":           p.info,
        "josh.info2":          p.info2,
        "josh.warm":           p.warm,
        "josh.secondary":      p.secondary,
        "josh.highlight":      p.highlight,
        "josh.highlight.bold": f"bold {p.highlight}",
        "josh.error":          p.error,
        "josh.error.bold":     f"bold {p.error}",
        "josh.text":           p.text,
        "josh.dim":            p.dim,
        "josh.tool":           p.info,
        "josh.tool.error":     p.error,
        "josh.tool.ok":        p.success,
        "josh.border":         p.primary,
        # Rich Markdown overrides — these are what makes **Persistent memory**
        # mid-sentence pop as bold pink/secondary.
        "markdown.h1":          f"bold {p.primary}",
        "markdown.h2":          f"bold {p.primary}",
        "markdown.h3":          f"bold {p.info2}",
        "markdown.h4":          f"bold {p.info2}",
        "markdown.strong":      f"bold {p.secondary}",
        "markdown.emph":        f"italic {p.warm}",
        "markdown.code":        f"bold {p.success} on {p.bg_alt}",
        "markdown.link":        f"underline {p.info}",
        "markdown.link_url":    f"dim {p.info}",
        "markdown.item.bullet": f"bold {p.info2}",
        "markdown.item.number": f"bold {p.info2}",
        "markdown.block_quote": f"italic {p.dim}",
        "markdown.hr":          p.primary,
    })


def build_dropdown_style_dict(p: Palette) -> dict[str, str]:
    """Hex-only dict consumed by prompt_toolkit's Style.from_dict() in cli.py."""
    return {
        "completion-menu.completion":               f"bg:{p.bg_alt} {p.info}",
        "completion-menu.completion.current":       f"bg:{p.primary} {p.bg} bold",
        "completion-menu.meta.completion":          f"bg:{p.bg_alt} {p.dim}",
        "completion-menu.meta.completion.current":  f"bg:{p.primary} {p.bg_alt}",
        "completion-menu.multi-column-meta":        f"bg:{p.bg_alt} {p.dim}",
        "scrollbar.background":                     f"bg:{p.bg_alt}",
        "scrollbar.button":                         f"bg:{p.primary}",
        "slash":                                    p.secondary,
        "cmd-name":                                 f"{p.info} bold",
    }


def render_theme_swatch(p: Palette) -> str:
    """Return a Rich-markup color swatch showing the palette's six main slots.

    Each ██ block is styled inline with one of the palette's hex codes so a
    `console.print(swatch)` reveals what the theme looks like without having
    to switch into it first.
    """
    blocks = [
        ("██", p.primary),     # the banner / title accent
        ("██", p.secondary),   # **bold** highlights, chevrons
        ("██", p.warm),        # logo, italic
        ("██", p.highlight),   # session ID, current marker
        ("██", p.success),     # section headers, ok states
        ("██", p.info),        # model name, links
    ]
    return " ".join(f"[{hex_}]{block}[/]" for block, hex_ in blocks)


def prompt_html(p: Palette) -> str:
    """The HTML for the chat prompt line: ▎ you ❱"""
    return (
        f'<b><style fg="{p.primary}">▎</style></b> '
        f'<b><style fg="{p.primary}">you</style></b> '
        f'<b><style fg="{p.secondary}">❱</style></b> '
    )


def picker_prompt_html(p: Palette, label: str) -> str:
    """HTML for the inline model-picker prompt."""
    return f'  <style fg="{p.primary}">{label}</style> '
