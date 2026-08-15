#!/usr/bin/env python3
"""
Validate ANTHROPIC_API_KEY before starting the advisor.

Why this exists: the launchers read the key with hidden input, so a paste that
silently failed or picked up stray whitespace looks identical to a good one.
Without this check the first symptom is a 401 in the middle of the first
question, which reads like the app is broken rather than the key being wrong.

Uses the models endpoint, which authenticates without spending tokens.

Exit codes:  0 = key works   1 = key rejected or unreachable   2 = not set
"""

from __future__ import annotations

import os
import sys

import anthropic


def mask(key: str) -> str:
    """Enough to recognise a key, not enough to use one."""
    if len(key) <= 14:
        return f"{key[:4]}...({len(key)} chars)"
    return f"{key[:12]}...{key[-3:]}  ({len(key)} chars)"


def main() -> int:
    raw = os.environ.get("ANTHROPIC_API_KEY")
    if not raw:
        print("  ANTHROPIC_API_KEY is not set.")
        return 2

    key = raw.strip()
    if key != raw:
        print("  Note: the key had leading/trailing whitespace. Trimmed for this check,")
        print("  but fix it at the source - a stray space or newline is a common cause of 401.")

    print(f"  Key seen : {mask(key)}")

    if key in {
        "sk-ant-api03-KUNCI-ASLIMU",
        "sk-ant-api03-PASTE_KUNCI_ASLI_DI_SINI",
        "sk-ant-kunci-aslimu",
        "sk-ant-...",
    }:
        print("  This is the placeholder text, not a real key.")
        print("  Get a real one at https://console.anthropic.com/settings/keys")
        return 1

    if not key.startswith("sk-ant-"):
        print("  Warning: real keys start with 'sk-ant-'. This one does not.")

    try:
        client = anthropic.Anthropic(api_key=key)
        models = client.models.list(limit=1)
    except anthropic.AuthenticationError:
        print("  REJECTED: the API rejected this key (401).")
        print("  Usual causes: wrong key, key was revoked, or only part of it got pasted.")
        print("  Check it at https://console.anthropic.com/settings/keys")
        return 1
    except anthropic.PermissionDeniedError:
        print("  REJECTED: key is valid but lacks permission for this workspace.")
        return 1
    except anthropic.APIConnectionError as exc:
        print(f"  Could not reach the API (network/proxy/firewall): {exc}")
        return 1
    except anthropic.APIStatusError as exc:
        print(f"  API returned {exc.status_code}: {exc}")
        return 1

    name = models.data[0].id if models.data else "unknown"
    print(f"  ACCEPTED: key works. Reachable model, e.g. {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
