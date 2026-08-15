#!/usr/bin/env python3
"""
Local HUD server for the advisor chatbot.

Serves a single-page browser UI and streams model output to it over SSE.
The reasoning side is not reimplemented here - system prompt, tools, vault
grounding and the tool loop all come from chatbot.py, so the CLI and the HUD
can never drift apart.

    pip install anthropic
    set ANTHROPIC_API_KEY, ADVISOR_VAULT_PATH (see vault.py)
    python server.py            -> http://127.0.0.1:8787

Binds to loopback only, on purpose: the advisor can read your notes, so the
socket must not be reachable from the network. Do not change the host without
adding authentication first.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator

import anthropic

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chatbot  # noqa: E402  - reuse prompt, tools, vault wiring, tool loop
import vault  # noqa: E402

HOST = "127.0.0.1"
PORT = int(os.environ.get("ADVISOR_HUD_PORT", "8787"))
HERE = Path(__file__).resolve().parent

# Single local user, so one conversation lives in the process. Guarded because
# ThreadingHTTPServer can overlap requests (a stray double-submit otherwise
# interleaves two turns into one history and corrupts it).
_messages: list[dict[str, Any]] = []
_lock = threading.Lock()


def _sse(event: str, data: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")


def _turn(client: anthropic.Anthropic, messages: list[dict[str, Any]]) -> Iterator[bytes]:
    """
    One assistant turn as SSE events, following tool calls until the model is
    done. Mirrors chatbot._stream_turn but yields to the browser instead of
    printing, and reports tool activity so the HUD can show what it is doing.
    """
    tool_calls: list[dict[str, Any]] = []

    with client.messages.stream(
        model=chatbot.MODEL if hasattr(chatbot, "MODEL") else "claude-opus-4-8",
        max_tokens=16000,
        thinking={"type": "adaptive"},
        system=chatbot._system_blocks(),
        tools=chatbot._active_tools(),
        messages=messages,
    ) as stream:
        for event in stream:
            etype = event.type

            if etype == "content_block_start":
                block = event.content_block
                if block.type == "tool_use":
                    tool_calls.append({"id": block.id, "name": block.name, "input_str": ""})
                elif block.type == "thinking":
                    yield _sse("thinking", {"state": "start"})

            elif etype == "content_block_delta":
                delta = event.delta
                if delta.type == "text_delta":
                    yield _sse("text", {"chunk": delta.text})
                elif delta.type == "thinking_delta":
                    yield _sse("thinking", {"state": "tick"})
                elif delta.type == "input_json_delta" and tool_calls:
                    tool_calls[-1]["input_str"] += delta.partial_json

        final = stream.get_final_message()

    messages.append({"role": "assistant", "content": list(final.content)})

    if not tool_calls:
        return

    results: list[dict[str, Any]] = []
    for call in tool_calls:
        try:
            parsed = json.loads(call["input_str"]) if call["input_str"] else {}
        except json.JSONDecodeError:
            parsed = {}

        yield _sse("tool", {"name": call["name"], "input": parsed, "state": "running"})
        try:
            output = chatbot._run_tool(call["name"], parsed)
        except Exception as exc:  # a broken tool must not kill the turn
            output = f"Tool '{call['name']}' failed: {exc}"
        yield _sse("tool", {"name": call["name"], "state": "done", "preview": output[:600]})

        results.append({"type": "tool_result", "tool_use_id": call["id"], "content": output})

    messages.append({"role": "user", "content": results})
    yield _sse("text", {"chunk": "\n\n"})
    yield from _turn(client, messages)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # keep the console readable
        pass

    # -- helpers ---------------------------------------------------------
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, payload: dict[str, Any]) -> None:
        self._send(code, json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    # -- routes ----------------------------------------------------------
    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            try:
                body = (HERE / "hud.html").read_bytes()
            except OSError:
                self._send(500, b"hud.html missing", "text/plain; charset=utf-8")
                return
            self._send(200, body, "text/html; charset=utf-8")
            return

        if self.path == "/api/status":
            roots = [{"label": lbl, "path": str(p)} for lbl, p in vault.vault_roots()]
            core = [c.strip() for c in os.environ.get("ADVISOR_VAULT_CORE", "").split(",") if c.strip()]
            with _lock:
                turns = sum(1 for m in _messages if m["role"] == "user" and isinstance(m["content"], str))
            self._json(200, {
                "model": "claude-opus-4-8",
                "roots": roots,
                "core_notes": core,
                "grounded": bool(roots),
                "turns": turns,
            })
            return

        self._send(404, b"not found", "text/plain; charset=utf-8")

    def do_POST(self) -> None:
        if self.path == "/api/reset":
            with _lock:
                _messages.clear()
            self._json(200, {"ok": True})
            return

        if self.path != "/api/chat":
            self._send(404, b"not found", "text/plain; charset=utf-8")
            return

        length = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._json(400, {"error": "bad json"})
            return

        text = (payload.get("message") or "").strip()
        if not text:
            self._json(400, {"error": "empty message"})
            return

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            self._json(500, {"error": "ANTHROPIC_API_KEY is not set"})
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        client = anthropic.Anthropic(api_key=api_key)

        with _lock:
            _messages.append({"role": "user", "content": text})
            snapshot = _messages

            try:
                for chunk in _turn(client, snapshot):
                    self.wfile.write(chunk)
                    self.wfile.flush()
                self.wfile.write(_sse("done", {}))
            except anthropic.APIError as exc:
                # Drop the failed turn so the next question starts from a
                # consistent history rather than replaying a broken exchange.
                while snapshot and snapshot[-1]["role"] != "user":
                    snapshot.pop()
                if snapshot:
                    snapshot.pop()
                self.wfile.write(_sse("error", {"message": str(exc)}))
            except (BrokenPipeError, ConnectionResetError):
                return  # browser navigated away mid-stream
            self.wfile.flush()


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit(
            "Error: ANTHROPIC_API_KEY is not set.\n"
            "  PowerShell: $env:ANTHROPIC_API_KEY = 'sk-ant-api03-...'"
        )

    roots = vault.vault_roots()
    url = f"http://{HOST}:{PORT}/"

    print()
    print("  CEO ADVISOR - HUD")
    print(f"  {url}")
    if roots:
        for label, path in roots:
            print(f"  source: {label} -> {path}")
    else:
        print("  source: none configured - answers will be generic")
    print("  Ctrl-C to stop")
    print()

    server = ThreadingHTTPServer((HOST, PORT), Handler)
    threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped")
        server.server_close()


if __name__ == "__main__":
    main()
