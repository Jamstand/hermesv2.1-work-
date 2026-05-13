"""Skills management for Josh v1.

Skills are markdown-file packages Claude Code loads from ~/.claude/skills/.
Each skill is a directory containing at least a SKILL.md file with YAML
frontmatter (`name`, `description`, optional `author`/`tags`/`version`).

This module implements:
  - listing installed skills
  - inspecting a single skill (prints SKILL.md preview)
  - installing from a git URL OR a name in the curated marketplace
  - uninstalling
  - browsing/searching the curated marketplace
  - snapshotting which skills are enabled in your config to/from JSON

What it deliberately does NOT do:
  - automatic version tracking / `update`  -> use `git pull` inside the skill dir
  - security `audit`                       -> would be theater without a real scanner
  - `publish` to a registry                -> use `gh pr create` against the marketplace
  - plugin support                         -> Claude Code has its own plugin install path
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SKILLS_DIR = Path.home() / ".claude" / "skills"
MARKETPLACE_PATH = Path(__file__).parent / "marketplace.json"


@dataclass
class SkillEntry:
    name: str
    description: str = ""
    author: str = ""
    version: str = ""
    tags: list[str] = field(default_factory=list)
    source: str = ""        # git URL the skill was installed from, or path
    installed_at: str = ""  # local install path

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Marketplace
# ---------------------------------------------------------------------------


def load_marketplace() -> list[dict[str, Any]]:
    """Load the curated marketplace manifest (skills/marketplace.json)."""
    if not MARKETPLACE_PATH.is_file():
        return []
    try:
        return json.loads(MARKETPLACE_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return []


def browse() -> list[dict[str, Any]]:
    return load_marketplace()


def search(query: str) -> list[dict[str, Any]]:
    q = query.strip().lower()
    if not q:
        return load_marketplace()
    matches: list[dict[str, Any]] = []
    for item in load_marketplace():
        hay = " ".join([
            item.get("name", ""),
            item.get("description", ""),
            item.get("author", ""),
            " ".join(item.get("tags", [])),
        ]).lower()
        if q in hay:
            matches.append(item)
    return matches


# ---------------------------------------------------------------------------
# Installed-skills lookup
# ---------------------------------------------------------------------------


def list_installed() -> list[SkillEntry]:
    if not SKILLS_DIR.is_dir():
        return []
    entries: list[SkillEntry] = []
    for d in sorted(SKILLS_DIR.iterdir()):
        if not d.is_dir():
            continue
        md = d / "SKILL.md"
        if not md.is_file():
            continue
        entries.append(_parse_skill_dir(d))
    return entries


def get_installed(name: str) -> SkillEntry | None:
    target = SKILLS_DIR / name
    if not target.is_dir() or not (target / "SKILL.md").is_file():
        return None
    return _parse_skill_dir(target)


def inspect_skill(name: str, max_chars: int = 4000) -> str | None:
    """Return the full SKILL.md text (truncated) for a preview, or None."""
    md = SKILLS_DIR / name / "SKILL.md"
    if not md.is_file():
        return None
    try:
        text = md.read_text(errors="ignore")
    except OSError:
        return None
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n... [truncated]"
    return text


def _parse_skill_dir(d: Path) -> SkillEntry:
    md = d / "SKILL.md"
    try:
        head = md.read_text(errors="ignore")[:2000]
    except OSError:
        head = ""
    desc, author, version, tags = "", "", "", []
    fm = re.search(r"^---\s*\n(.*?)\n---", head, re.DOTALL | re.MULTILINE)
    if fm:
        for line in fm.group(1).splitlines():
            key, sep, val = line.partition(":")
            if not sep:
                continue
            k = key.strip().lower()
            v = val.strip().strip('"').strip("'")
            if k == "description":
                desc = v
            elif k == "author":
                author = v
            elif k == "version":
                version = v
            elif k == "tags":
                tags = [t.strip() for t in v.strip("[]").split(",") if t.strip()]
    # Source: git remote if present
    source = ""
    git_dir = d / ".git"
    if git_dir.is_dir():
        try:
            r = subprocess.run(
                ["git", "-C", str(d), "remote", "get-url", "origin"],
                capture_output=True, text=True, timeout=5,
            )
            if r.returncode == 0:
                source = r.stdout.strip()
        except Exception:  # noqa: BLE001
            pass
    return SkillEntry(
        name=d.name,
        description=desc,
        author=author,
        version=version,
        tags=tags,
        source=source,
        installed_at=str(d),
    )


# ---------------------------------------------------------------------------
# Install / uninstall
# ---------------------------------------------------------------------------


class SkillError(Exception):
    pass


def resolve_identifier(identifier: str) -> dict[str, Any]:
    """Resolve a skill identifier to {url, subdir, default_name}.

    - 'scheme://...' or 'git@...' → that URL, no subdir
    - 'owner/repo'                → https://github.com/owner/repo, no subdir
    - plain name                  → looked up in the marketplace (may have subdir)
    """
    if "://" in identifier or identifier.startswith("git@"):
        return {"url": identifier, "subdir": None, "default_name": None}
    if "/" in identifier and not identifier.startswith("/"):
        return {
            "url": f"https://github.com/{identifier}",
            "subdir": None,
            "default_name": identifier.split("/")[-1],
        }
    for item in load_marketplace():
        if item.get("name") == identifier:
            if not item.get("url"):
                raise SkillError(f"marketplace entry '{identifier}' has no url")
            return {
                "url": item["url"],
                "subdir": item.get("subdir"),
                "default_name": item["name"],
            }
    raise SkillError(
        f"'{identifier}' is not a git URL and not in the marketplace. "
        "Use a full URL, 'owner/repo' shorthand, or `joshv1 skill browse` to see what's available."
    )


def install(identifier: str, name: str | None = None) -> SkillEntry:
    """Clone a skill into ~/.claude/skills/<name>. Raises SkillError on failure.

    If the marketplace entry has a `subdir`, only that subdir is moved into
    the skills dir (the rest of the cloned repo is discarded).
    """
    import tempfile

    resolved = resolve_identifier(identifier)
    url, subdir, default_name = resolved["url"], resolved["subdir"], resolved["default_name"]

    if name is None:
        name = default_name or (subdir if subdir else _basename_from_url(url))

    SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    target = SKILLS_DIR / name
    if target.exists():
        raise SkillError(f"already installed at {target}")

    with tempfile.TemporaryDirectory(prefix="joshv1-skill-") as tmp:
        tmp_path = Path(tmp) / "clone"
        result = subprocess.run(
            ["git", "clone", "--depth=1", url, str(tmp_path)],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            raise SkillError(f"git clone failed: {(result.stderr or result.stdout).strip()}")

        source = tmp_path / subdir if subdir else tmp_path
        if not source.is_dir():
            raise SkillError(f"subdir '{subdir}' not found in cloned repo")
        if not (source / "SKILL.md").is_file():
            raise SkillError("source doesn't contain SKILL.md — not a valid skill")

        shutil.move(str(source), str(target))

    return _parse_skill_dir(target)


def _basename_from_url(url: str) -> str:
    base = url.rstrip("/").split("/")[-1]
    return base[:-4] if base.endswith(".git") else base


def uninstall(name: str) -> None:
    target = SKILLS_DIR / name
    if not target.is_dir():
        raise SkillError(f"not installed: {name}")
    shutil.rmtree(target)


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


def snapshot_export() -> dict[str, Any]:
    """Export a JSON-serializable snapshot of installed skills + their sources."""
    return {
        "version": 1,
        "skills": [e.to_dict() for e in list_installed()],
    }


def snapshot_import(snapshot: dict[str, Any], skip_existing: bool = True) -> list[str]:
    """Install any skills from a snapshot that aren't already installed.

    Returns the list of names that were freshly installed.
    """
    installed_now: list[str] = []
    for entry in snapshot.get("skills", []):
        name = entry.get("name")
        source = entry.get("source") or ""
        if not name or not source:
            continue
        if skip_existing and (SKILLS_DIR / name).exists():
            continue
        try:
            install(source, name=name)
            installed_now.append(name)
        except SkillError:
            continue
    return installed_now
