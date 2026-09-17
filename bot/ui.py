"""Local chat window for tuning the persona.

Talk to her, watch which boundary fires, edit the voice card, talk again. That
loop is the whole point of this file -- it is a tuning instrument, not a
preview of a product surface.

Standard library only. Start it with `python -m bot.ui`. Set ANTHROPIC_API_KEY
first for live replies; without one it still classifies every message and shows
which boundary fired, which is the half you can tune without spending anything.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from bot.chat import TERMINATE, ChatAgent, classify
from bot.driver import LocalDriver
from bot.persona import Persona, PersonaError

TEMPLATE = Path(__file__).with_name("ui.html")
THREAD = "tuning"


def build_agent(persona: Persona) -> ChatAgent | None:
    """Wire a live agent when credentials exist, else run gates-only."""
    try:
        from bot.chat import AnthropicChatLLM

        return ChatAgent(persona, AnthropicChatLLM())
    except Exception:
        return None


class Handler(BaseHTTPRequestHandler):
    driver: LocalDriver
    persona: Persona
    agent: ChatAgent | None = None
    persona_path: Path = Path("persona.json")
    annotations: dict[int, list[dict]] = {}

    def log_message(self, *args) -> None:  # quiet
        pass

    # -- plumbing --------------------------------------------------------

    def _send(self, body: bytes, ctype: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: dict, status: int = 200) -> None:
        self._send(json.dumps(payload).encode(), "application/json", status)

    def _read_json(self) -> dict | None:
        length = int(self.headers.get("Content-Length") or 0)
        try:
            return json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return None

    # -- state -----------------------------------------------------------

    def _state(self) -> dict:
        cls = self.__class__
        return {
            "messages": [
                dict(m.__dict__, ann=cls.annotations.get(m.id, []))
                for m in self.driver.thread(THREAD, limit=200)
            ],
            "prompt": self.persona.voice_card(),
            "rules": list(self.persona.voice.rules),
            "facts": list(self.persona.facts),
            "live": cls.agent is not None,
        }

    def _page(self) -> bytes:
        ident = self.persona.identity
        live = self.__class__.agent is not None
        html = TEMPLATE.read_text(encoding="utf-8")
        for token, value in (
            ("{{NAME}}", ident.name),
            ("{{MODE}}", "she's listening" if live
             else "gates only — set ANTHROPIC_API_KEY and restart for replies"),
        ):
            html = html.replace(token, value)
        return html.encode()

    # -- routes ----------------------------------------------------------

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self._send(self._page(), "text/html; charset=utf-8")
        elif path == "/api/state":
            self._json(self._state())
        elif path == "/api/timeline":
            self._json({"posts": [p.__dict__ for p in self.driver.timeline(100)]})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/chat":
            self._chat()
        elif path == "/api/persona":
            self._tune()
        elif path == "/api/reset":
            self._reset()
        else:
            self._json({"error": "not found"}, 404)

    def _chat(self) -> None:
        payload = self._read_json()
        if payload is None:
            self._json({"error": "bad json"}, 400)
            return
        text = str(payload.get("text", "")).strip()
        if not text:
            self._json({"error": "empty message"}, 400)
            return

        cls = self.__class__
        inbound = self.driver.receive_dm(THREAD, text)
        assessment = classify(text, self.persona)

        ann = [{"kind": "gate", "text": assessment.action}]
        ann += [{"kind": "warn", "text": p} for p in assessment.policies
                if p != assessment.action]
        cls.annotations[inbound.id] = ann

        result = {"ok": True, "gate": assessment.action,
                  "policies": list(assessment.policies)}

        if cls.agent is None:
            cls.annotations[inbound.id].append(
                {"kind": "warn", "text": "no credentials, nothing generated"})
            result["replied"] = False
            self._json(result)
            return

        history = [
            {"role": "user" if m.inbound else "assistant", "content": m.body}
            for m in self.driver.thread(THREAD, limit=40)
        ]
        reply = cls.agent.respond(text, history=history)

        if reply.terminated:
            cls.annotations[inbound.id].append(
                {"kind": "stop", "text": "thread terminated, flagged"})
        elif reply.sends:
            out = self.driver.send_dm(THREAD, reply.text)
            marks = []
            if reply.canned:
                marks.append({"kind": "warn", "text": "model output rejected, canned"})
            if reply.flagged:
                marks.append({"kind": "stop", "text": "flagged"})
            # send_dm does not hand back the row, so tag the newest outbound.
            latest = self.driver.thread(THREAD, limit=1)
            if latest and marks:
                cls.annotations[latest[-1].id] = marks

        result |= {"replied": reply.sends, "canned": reply.canned,
                   "terminated": reply.terminated}
        self._json(result)

    def _tune(self) -> None:
        """Apply voice-card edits live, optionally writing them back to disk."""
        payload = self._read_json()
        if payload is None:
            self._json({"error": "bad json"}, 400)
            return

        raw = json.loads(self.__class__.persona_path.read_text(encoding="utf-8"))
        if payload.get("rules"):
            raw["voice"]["rules"] = list(payload["rules"])
        if payload.get("facts"):
            raw["facts"] = list(payload["facts"])

        try:
            persona = Persona.from_dict(raw, source=self.__class__.persona_path)
        except PersonaError as exc:
            self._json({"ok": False, "error": str(exc)}, 400)
            return

        cls = self.__class__
        cls.persona = persona
        if cls.agent is not None:
            cls.agent.persona = persona

        written = False
        if payload.get("save"):
            cls.persona_path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
            written = True
        self._json({"ok": True, "written": written})

    def _reset(self) -> None:
        cls = self.__class__
        self.driver.clear_thread(THREAD)
        cls.annotations = {}
        self._json({"ok": True})


def serve(host: str = "127.0.0.1", port: int = 8000,
          db: str | Path = "content/local.db",
          persona_path: str | Path = "persona.json") -> None:
    Handler.persona_path = Path(persona_path)
    Handler.persona = Persona.load(Handler.persona_path)
    Handler.driver = LocalDriver(db)
    Handler.agent = build_agent(Handler.persona)
    mode = "live replies" if Handler.agent else "gates only (set ANTHROPIC_API_KEY for replies)"
    print(f"{Handler.persona.identity.name} at http://{host}:{port}  [{mode}]")
    print("ctrl-c to stop")
    ThreadingHTTPServer((host, port), Handler).serve_forever()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Chat with the persona and tune her voice.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--db", default="content/local.db")
    ap.add_argument("--persona", default="persona.json")
    args = ap.parse_args()
    try:
        serve(args.host, args.port, args.db, args.persona)
    except KeyboardInterrupt:
        print()
