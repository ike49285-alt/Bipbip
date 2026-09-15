"""A local web UI. `python -m thirsttrap serve`.

Standard-library http.server, bound to loopback by default. It talks to the
same local backend the CLI uses, so the model stays on your machine and the
page is just a nicer front end than a terminal.

Bound to 127.0.0.1 deliberately: this serves an unauthenticated endpoint that
drives a language model, and that should not be reachable from the network
without the operator saying so explicitly with --host.
"""

from __future__ import annotations

import json
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .chat import Repl, Session
from .llm import Backend, detect
from .profile import Profile
from .rank import Ranked

PAGE = (Path(__file__).parent / "ui.html").read_text(encoding="utf-8")


def _payload(repl: Repl, items: list[Ranked], note: str = "") -> dict:
    profile = repl.profile
    return {
        "note": note,
        "reply": repl.last_reply,
        "persona": repl.session.persona,
        "backend": repl.session.engine().describe(),
        "confidence": profile.confidence(),
        "rules": profile.standing.describe(),
        "constraints": repl.session.constraints().describe(),
        "adopted": repl.last_adopted,
        "weights": {k: round(v, 4) for k, v in profile.weights.items()},
        "kept": [k.text for k in repl.kept],
        "posts": [
            {
                "text": item.text,
                "score": round(item.final, 1),
                "duplicate": item.duplicate_of,
                "components": {k: round(v, 3) for k, v in item.score.components.items()},
                "notes": item.score.notes,
            }
            for item in items
        ],
    }


class Handler(BaseHTTPRequestHandler):
    repl: Repl = None  # set by serve()

    def log_message(self, *args):
        pass

    def _send(self, body: bytes, content_type: str, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        # This page only ever talks to its own origin.
        self.send_header("Content-Security-Policy", "default-src 'self' 'unsafe-inline'")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data: dict, status: int = 200):
        self._send(json.dumps(data).encode("utf-8"), "application/json", status)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif self.path == "/api/state":
            self._json(_payload(self.repl, self.repl.shown))
        else:
            self._json({"error": "not found"}, status=404)

    def do_POST(self):
        if self.path != "/api/say":
            self._json({"error": "not found"}, status=404)
            return

        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            self._json({"error": "malformed request"}, status=400)
            return

        line = str(body.get("line", "")).strip()
        if not line:
            self._json(_payload(self.repl, self.repl.shown))
            return

        # Repl.handle already turns every ordinary failure into a message.
        output, _ = self.repl.handle(line)
        # Commands and errors surface as a note; an ordinary turn renders as
        # her reply plus drafts, which the payload already carries.
        note = output if line.startswith("/") or output.startswith("error:") else ""
        self._json(_payload(self.repl, self.repl.shown, note=note))


def serve(
    host: str = "127.0.0.1",
    port: int = 8765,
    backend: Backend | None = None,
    profile: Profile | None = None,
    n: int = 12,
    top: int = 5,
    open_browser: bool = True,
) -> None:
    Handler.repl = Repl(
        session=Session(
            profile=profile if profile is not None else Profile.load(),
            backend=backend or detect(),
            n=n,
        ),
        top=top,
    )

    httpd = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{httpd.server_address[1]}"
    print(f"thirsttrap on {url}")
    print(f"generating with {Handler.repl.session.engine().describe()}")
    print("ctrl-c to stop")

    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass  # headless is fine; the URL is printed above

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()
