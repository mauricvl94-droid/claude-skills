#!/usr/bin/env python3
"""
Read-only knowledge-vault adapter for the advisor chatbots.

Points the advisor at one or more folders of markdown notes (an Obsidian
vault, a docs tree, a project folder) so it can ground its advice in your real
context instead of generic frameworks.

Configuration is entirely by environment variable, so this file stays generic
and contains no personal paths or data:

    ADVISOR_VAULT_PATH   One or more roots, separated by the platform path
                         separator (';' on Windows, ':' elsewhere).
                         Integration is off when unset.
    ADVISOR_VAULT_CORE   Optional. Comma-separated paths that are always
                         loaded into the system prompt (profile, priorities,
                         standing context...). Keep this small; it is sent
                         on every request.

Path form:
    One root  -> paths are plain and root-relative:  '00 - System/Home.md'
    Several   -> paths carry a root label first:     'vault/00 - System/Home.md'
    The label is the root folder's own name, so it reads naturally.

Design constraints, deliberate:
  * READ-ONLY. There is no write, move, or delete path in this module, and
    none should ever be added. The advisor reasons over your notes; it does
    not edit them.
  * Sandboxed per root. Every requested path is resolved and rejected if it
    escapes its root, so a model-supplied path cannot walk into the rest of
    the filesystem. Adding a root widens access to exactly that root and
    nothing else.
  * Bounded. Reads and search results are truncated so a large corpus cannot
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


def vault_roots() -> list[tuple[str, Path]]:
    """
    Configured roots as (label, resolved_path), in declaration order.

    Unreadable or duplicate entries are dropped silently: a stale path in the
    config should degrade the advisor, not crash it. Labels are made unique so
    two roots that happen to share a folder name stay addressable.
    """
    raw = os.environ.get("ADVISOR_VAULT_PATH", "").strip()
    if not raw:
        return []

    roots: list[tuple[str, Path]] = []
    seen: set[Path] = set()
    used_labels: set[str] = set()

    for part in raw.split(os.pathsep):
        part = part.strip().strip('"')
        if not part:
            continue
        candidate = Path(part).expanduser()
        if not candidate.is_dir():
            continue
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)

        label = resolved.name or str(resolved)
        if label in used_labels:
            n = 2
            while f"{label}-{n}" in used_labels:
                n += 1
            label = f"{label}-{n}"
        used_labels.add(label)

        roots.append((label, resolved))

    return roots


def vault_root() -> Path | None:
    """First configured root, or None. Kept for callers that just want one."""
    roots = vault_roots()
    return roots[0][1] if roots else None


def is_enabled() -> bool:
    return bool(vault_roots())


def _multi() -> bool:
    return len(vault_roots()) > 1


def _iter_notes(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            if fn.lower().endswith(".md"):
                yield Path(dirpath) / fn


def _display_path(label: str, root: Path, note: Path) -> str:
    rel = note.relative_to(root).as_posix()
    return f"{label}/{rel}" if _multi() else rel


def _resolve(path: str) -> tuple[Path | None, str]:
    """
    Resolve a model-supplied path to a real file inside one of the roots.

    Returns (resolved_or_None, reason_when_None). Refusal is never silent:
    the model gets told why, so it can correct itself instead of looping.
    """
    roots = vault_roots()
    if not roots:
        return None, "Vault integration is not configured (ADVISOR_VAULT_PATH unset)."

    raw = (path or "").strip().strip('"').replace("\\", "/")
    if not raw:
        return None, "Empty path."

    if _multi():
        head, _, rest = raw.partition("/")
        match = [(lbl, rt) for lbl, rt in roots if lbl == head]
        if not match:
            names = ", ".join(lbl for lbl, _ in roots)
            return None, f"Unknown root '{head}'. Configured roots: {names}."
        if not rest:
            return None, f"'{raw}' is a root, not a note. Give a path inside it."
        targets = [(match[0][0], match[0][1], rest)]
    else:
        targets = [(roots[0][0], roots[0][1], raw)]

    for _label, root, rel in targets:
        candidate = (root / rel).expanduser()
        try:
            resolved = candidate.resolve()
        except OSError:
            return None, f"Could not resolve '{path}'."
        if resolved != root and root not in resolved.parents:
            return None, f"Refused: '{path}' resolves outside the configured root."
        if not resolved.is_file():
            return None, f"Not found: '{path}'. Use search_vault to find the correct path."
        return resolved, ""

    return None, f"Not found: '{path}'."


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def search_vault(query: str, limit: int = 5) -> str:
    """Case-insensitive substring search across every configured root."""
    roots = vault_roots()
    if not roots:
        return "Vault integration is not configured (ADVISOR_VAULT_PATH unset)."

    needle = (query or "").strip().lower()
    if not needle:
        return "Empty query."

    limit = max(1, min(int(limit or 5), MAX_RESULTS))
    hits: list[tuple[int, str, str]] = []

    for label, root in roots:
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
            hits.append((count, _display_path(label, root, note), snippet))

    if not hits:
        return f'No notes matched "{query}".'

    # Most mentions first: a note that discusses the topic beats one that
    # merely name-drops it.
    hits.sort(key=lambda h: -h[0])

    lines = [f'{len(hits)} note(s) matched "{query}". Top {min(limit, len(hits))}:', ""]
    for count, shown, snippet in hits[:limit]:
        lines.append(f"- **{shown}** ({count} mention(s))")
        lines.append(f"  ...{snippet}...")
    if len(hits) > limit:
        lines.append("")
        lines.append(f"({len(hits) - limit} more not shown - narrow the query or raise limit.)")
    lines.append("")
    lines.append("Use read_vault_note with one of these paths to read the full note.")
    return "\n".join(lines)


def read_vault_note(path: str) -> str:
    """Read one note in full, by the path form search_vault returns."""
    resolved, reason = _resolve(path)
    if resolved is None:
        return reason

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
    roots = vault_roots()
    if not roots:
        return ""

    raw = os.environ.get("ADVISOR_VAULT_CORE", "").strip()
    if not raw:
        return ""

    chunks: list[str] = []
    missing: list[str] = []
    for rel in [p.strip() for p in raw.split(",") if p.strip()]:
        resolved, _reason = _resolve(rel)
        if resolved is None:
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

    where = ", ".join(f"{lbl} ({rt})" for lbl, rt in roots)
    header = (
        "===============================================================\n"
        "OPERATOR CONTEXT - LOADED FROM THE USER'S KNOWLEDGE BASE\n"
        "===============================================================\n\n"
        "The notes below are this user's real situation: who they are, what they\n"
        "are working on, and what is currently open. Ground your advice in these\n"
        "specifics. Never answer with generic frameworks when the knowledge base\n"
        "already tells you the actual numbers, names, and constraints.\n\n"
        f"Configured source(s): {where}\n\n"
        "You can search everything with search_vault and open any note with\n"
        "read_vault_note. Do that whenever a question touches something the\n"
        "context below only mentions in passing - it is a small slice of a much\n"
        "larger corpus.\n\n"
        "Your access is read-only. If the user asks you to change a note,\n"
        "explain the edit and let them make it.\n\n"
    )
    if missing:
        header += f"(Configured but unreadable, ignore: {', '.join(missing)})\n\n"

    return header + "\n\n---\n\n".join(chunks)


VAULT_TOOLS = [
    {
        "name": "search_vault",
        "description": (
            "Search the user's knowledge base for notes mentioning a term. "
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
            "Read one note in full, using a path exactly as returned by "
            "search_vault. Use after search_vault when a snippet is not enough "
            "to answer precisely."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Path as shown by search_vault. With a single configured "
                        "root this is root-relative ('00 - System/Home.md'); with "
                        "several roots it starts with the root label "
                        "('vault/00 - System/Home.md')."
                    ),
                },
            },
            "required": ["path"],
        },
    },
]
