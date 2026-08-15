#!/usr/bin/env python3
"""
Read-only knowledge-vault adapter for the advisor chatbots.

Points the advisor at a folder of markdown notes (an Obsidian vault, a docs
directory, anything) so it can ground its advice in your real context instead
of generic frameworks.

Configuration is entirely by environment variable, so this file stays generic
and contains no personal paths or data:

    ADVISOR_VAULT_PATH   Absolute path to the vault root. Integration is off
                         when unset.
    ADVISOR_VAULT_CORE   Optional. Comma-separated vault-relative paths that
                         are always loaded into the system prompt (profile,
                         priorities, standing context...). Keep this small;
                         it is sent on every request.

Design constraints, deliberate:
  * READ-ONLY. There is no write, move, or delete path in this module, and
    none should ever be added. The advisor reasons over your notes; it does
    not edit them.
  * Sandboxed. Every requested path is resolved and rejected if it escapes
    the vault root, so a model-supplied path cannot walk into the rest of
    the filesystem.
  * Bounded. Reads and search results are truncated so a large vault cannot
    blow up the context window or the bill.
"""

from __future__ import annotations

import os
from pathlib import Path

# Caps chosen so a runaway search cannot dominate the context window.
MAX_NOTE_CHARS = 20_000
MAX_SNIPPET_CHARS = 400
MAX_RESULTS = 10
SKIP_DIRS = {".git", ".obsidian", ".trash", "node_modules", ".venv", "__pycache__"}


def vault_root() -> Path | None:
    """Configured vault root, or None when the integration is switched off."""
    raw = os.environ.get("ADVISOR_VAULT_PATH", "").strip()
    if not raw:
        return None
    root = Path(raw).expanduser()
    if not root.is_dir():
        return None
    return root.resolve()


def is_enabled() -> bool:
    return vault_root() is not None


def _iter_notes(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            if fn.lower().endswith(".md"):
                yield Path(dirpath) / fn


def _safe_resolve(root: Path, relative: str) -> Path | None:
    """Resolve a vault-relative path, refusing anything outside the vault."""
    candidate = (root / relative).expanduser()
    try:
        resolved = candidate.resolve()
    except OSError:
        return None
    if resolved != root and root not in resolved.parents:
        return None  # traversal attempt
    return resolved


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def search_vault(query: str, limit: int = 5) -> str:
    """Case-insensitive substring search across the vault. Returns markdown."""
    root = vault_root()
    if root is None:
        return "Vault integration is not configured (ADVISOR_VAULT_PATH unset)."

    needle = (query or "").strip().lower()
    if not needle:
        return "Empty query."

    limit = max(1, min(int(limit or 5), MAX_RESULTS))
    hits: list[tuple[int, str, str]] = []

    for note in _iter_notes(root):
        try:
            text = _read_text(note)
        except OSError:
            continue
        low = text.lower()
        count = low.count(needle)
        if not count:
            continue

        pos = low.find(needle)
        start = max(0, pos - MAX_SNIPPET_CHARS // 2)
        snippet = text[start:start + MAX_SNIPPET_CHARS].replace("\n", " ").strip()
        rel = note.relative_to(root).as_posix()
        hits.append((count, rel, snippet))

    if not hits:
        return f'No notes matched "{query}".'

    # Most mentions first: a note that discusses the topic beats one that
    # merely name-drops it.
    hits.sort(key=lambda h: -h[0])

    lines = [f'{len(hits)} note(s) matched "{query}". Top {min(limit, len(hits))}:', ""]
    for count, rel, snippet in hits[:limit]:
        lines.append(f"- **{rel}** ({count} mention(s))")
        lines.append(f"  ...{snippet}...")
    if len(hits) > limit:
        lines.append("")
        lines.append(f"({len(hits) - limit} more not shown - narrow the query or raise limit.)")
    lines.append("")
    lines.append("Use read_vault_note with one of these paths to read the full note.")
    return "\n".join(lines)


def read_vault_note(path: str) -> str:
    """Read one note by vault-relative path."""
    root = vault_root()
    if root is None:
        return "Vault integration is not configured (ADVISOR_VAULT_PATH unset)."

    resolved = _safe_resolve(root, path)
    if resolved is None:
        return f"Refused: '{path}' resolves outside the vault."
    if not resolved.is_file():
        return f"Not found: '{path}'. Use search_vault to find the correct path."

    try:
        text = _read_text(resolved)
    except OSError as exc:
        return f"Could not read '{path}': {exc}"

    if len(text) > MAX_NOTE_CHARS:
        text = text[:MAX_NOTE_CHARS] + f"\n\n[...truncated at {MAX_NOTE_CHARS} chars...]"
    return f"# {path}\n\n{text}"


def core_context() -> str:
    """
    Always-on context block built from ADVISOR_VAULT_CORE.

    Returned as one string so the caller can put it in a cached system block:
    it is resent every turn, and prompt caching is what keeps that affordable.
    """
    root = vault_root()
    if root is None:
        return ""

    raw = os.environ.get("ADVISOR_VAULT_CORE", "").strip()
    if not raw:
        return ""

    chunks: list[str] = []
    missing: list[str] = []
    for rel in [p.strip() for p in raw.split(",") if p.strip()]:
        resolved = _safe_resolve(root, rel)
        if resolved is None or not resolved.is_file():
            missing.append(rel)
            continue
        try:
            text = _read_text(resolved)
        except OSError:
            missing.append(rel)
            continue
        if len(text) > MAX_NOTE_CHARS:
            text = text[:MAX_NOTE_CHARS] + "\n[...truncated...]"
        chunks.append(f"### {rel}\n\n{text}")

    if not chunks:
        return ""

    header = (
        "═══════════════════════════════════════════════════════════════\n"
        "OPERATOR CONTEXT - LOADED FROM THE USER'S KNOWLEDGE VAULT\n"
        "═══════════════════════════════════════════════════════════════\n\n"
        "The notes below are this user's real situation: who they are, what they\n"
        "are working on, and what is currently open. Ground your advice in these\n"
        "specifics. Never answer with generic frameworks when the vault already\n"
        "tells you the actual numbers, names, and constraints.\n\n"
        "You can search the rest of the vault with search_vault and open any note\n"
        "with read_vault_note. Do that whenever a question touches something the\n"
        "context below only mentions in passing.\n\n"
        "Your vault access is read-only. If the user asks you to change a note,\n"
        "explain the edit and let them make it.\n\n"
    )
    if missing:
        header += f"(Configured but unreadable, ignore: {', '.join(missing)})\n\n"

    return header + "\n\n---\n\n".join(chunks)


VAULT_TOOLS = [
    {
        "name": "search_vault",
        "description": (
            "Search the user's knowledge vault for notes mentioning a term. "
            "Returns matching note paths with snippets, ranked by how often the "
            "term appears. Use this before answering any question about the "
            "user's own business, projects, people, finances, or history - their "
            "notes are the source of truth, not your assumptions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Term to search for, e.g. a project, company, person, or metric.",
                },
                "limit": {
                    "type": "integer",
                    "description": f"Max results, 1-{MAX_RESULTS}. Default 5.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "read_vault_note",
        "description": (
            "Read one note from the user's vault in full, by vault-relative path "
            "(as returned by search_vault). Use after search_vault when a snippet "
            "is not enough to answer precisely."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Vault-relative path, e.g. '00 - System/Active Priorities.md'.",
                },
            },
            "required": ["path"],
        },
    },
]
