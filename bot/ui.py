"""Local web UI: a fake timeline and a DM inbox you can type into.

Runs on the standard library alone -- no Flask, no install step. Start it with
`python -m bot.ui` and open the printed URL. Everything it shows comes from
LocalDriver, so what you see is exactly what the bot did.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from bot.driver import LocalDriver
from bot.persona import Persona

PAGE = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__NAME__ — local</title>
<style>
:root{--bg:#fff;--fg:#14151a;--dim:#6b7280;--line:#e5e7eb;--card:#fafafa;--accent:#2563eb}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){
--bg:#0d0e12;--fg:#e8e9ed;--dim:#9096a3;--line:#24262e;--card:#15171d;--accent:#6b9fff}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 ui-sans-serif,system-ui,-apple-system,sans-serif}
.wrap{max-width:620px;margin:0 auto;padding:0 16px 64px}
header{padding:20px 0 12px;border-bottom:1px solid var(--line);margin-bottom:16px}
h1{font-size:19px;margin:0 0 2px}
.handle{color:var(--dim);font-size:14px}
.badge{display:inline-block;margin-top:8px;padding:3px 9px;border:1px solid var(--line);
border-radius:999px;font-size:12px;color:var(--dim)}
nav{display:flex;gap:4px;margin-bottom:18px}
nav button{flex:1;padding:9px;border:1px solid var(--line);background:var(--card);color:var(--dim);
border-radius:8px;font:inherit;font-size:14px;cursor:pointer}
nav button[aria-selected=true]{background:var(--accent);border-color:var(--accent);color:#fff}
.post{border:1px solid var(--line);border-radius:12px;padding:14px;margin-bottom:12px;background:var(--card)}
.post .img{aspect-ratio:1;background:repeating-linear-gradient(45deg,var(--line),var(--line) 8px,transparent 8px,transparent 16px);
border-radius:8px;display:grid;place-items:center;color:var(--dim);font-size:12px;margin-bottom:10px;text-align:center;padding:8px}
.meta{color:var(--dim);font-size:12px;margin-top:8px}
.msg{max-width:78%;padding:9px 13px;border-radius:16px;margin-bottom:8px;white-space:pre-wrap;word-break:break-word}
.msg.in{background:var(--card);border:1px solid var(--line);border-bottom-left-radius:4px}
.msg.out{background:var(--accent);color:#fff;margin-left:auto;border-bottom-right-radius:4px}
form{display:flex;gap:8px;margin-top:16px}
input{flex:1;padding:11px;border:1px solid var(--line);border-radius:9px;background:var(--bg);color:var(--fg);font:inherit;min-width:0}
input:focus{outline:2px solid var(--accent);outline-offset:-1px}
button.send{padding:11px 16px;border:0;border-radius:9px;background:var(--accent);color:#fff;font:inherit;cursor:pointer}
.empty{color:var(--dim);text-align:center;padding:40px 16px;font-size:14px}
.note{color:var(--dim);font-size:12px;margin-top:14px;padding:10px;border:1px dashed var(--line);border-radius:8px}
</style></head><body><div class="wrap">
<header>
  <h1>__NAME__</h1><div class="handle">@__HANDLE__</div>
  <div class="badge">__DISCLOSURE__</div>
</header>
<nav>
  <button id="tab-tl" aria-selected="true" onclick="show('tl')">Timeline</button>
  <button id="tab-dm" aria-selected="false" onclick="show('dm')">DMs</button>
</nav>
<div id="tl"></div>
<div id="dm" hidden></div>
</div>
<script>
let tab='tl', thread='u/1';
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
function show(t){tab=t;
  document.getElementById('tl').hidden = t!=='tl';
  document.getElementById('dm').hidden = t!=='dm';
  document.getElementById('tab-tl').ariaSelected = t==='tl';
  document.getElementById('tab-dm').ariaSelected = t==='dm';
  render();}
async function render(){
  if(tab==='tl'){
    const d=await (await fetch('/api/timeline')).json();
    document.getElementById('tl').innerHTML = d.posts.length ? d.posts.map(p=>`
      <div class="post"><div class="img">${esc(p.image)}</div>
      <div>${esc(p.caption)}</div><div class="meta">${esc(p.created_at)}</div></div>`).join('')
      : '<div class="empty">No posts yet. The post job draws from the approved queue.</div>';
  } else {
    const d=await (await fetch('/api/thread?id='+encodeURIComponent(thread))).json();
    document.getElementById('dm').innerHTML =
      (d.messages.length ? d.messages.map(m=>`<div class="msg ${m.direction}">${esc(m.body)}</div>`).join('')
        : '<div class="empty">Say something to the persona.</div>')
      + `<form onsubmit="return sendDM(event)">
           <input id="box" placeholder="Message @__HANDLE__" autocomplete="off">
           <button class="send" type="submit">Send</button></form>
         <div class="note">Inbound messages are recorded. The DM agent is step 5 &mdash;
         until it lands, nothing replies automatically.</div>`;
  }
}
async function sendDM(e){e.preventDefault();
  const box=document.getElementById('box'), text=box.value.trim();
  if(!text) return false;
  box.value='';
  await fetch('/api/dm',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({thread,text})});
  render(); return false;}
render();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    driver: LocalDriver
    persona: Persona

    def log_message(self, *args) -> None:  # quiet
        pass

    def _send(self, body: bytes, ctype: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: dict, status: int = 200) -> None:
        self._send(json.dumps(payload).encode(), "application/json", status)

    def do_GET(self) -> None:
        route = urlparse(self.path)
        if route.path == "/":
            ident = self.persona.identity
            page = (
                PAGE.replace("__NAME__", ident.name)
                .replace("__HANDLE__", ident.handle)
                .replace("__DISCLOSURE__", ident.disclosure_short)
            )
            self._send(page.encode(), "text/html; charset=utf-8")
        elif route.path == "/api/timeline":
            self._json({"posts": [p.__dict__ for p in self.driver.timeline()]})
        elif route.path == "/api/thread":
            tid = parse_qs(route.query).get("id", ["u/1"])[0]
            self._json({"messages": [m.__dict__ for m in self.driver.thread(tid)]})
        elif route.path == "/api/threads":
            self._json({"threads": self.driver.threads()})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/dm":
            self._json({"error": "not found"}, 404)
            return
        length = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._json({"error": "bad json"}, 400)
            return
        text = str(payload.get("text", "")).strip()
        if not text:
            self._json({"error": "empty message"}, 400)
            return
        dm = self.driver.receive_dm(str(payload.get("thread", "u/1")), text)
        self._json({"ok": True, "id": dm.id})


def serve(host: str = "127.0.0.1", port: int = 8000, db: str | Path = "content/local.db") -> None:
    Handler.driver = LocalDriver(db)
    Handler.persona = Persona.load()
    print(f"{Handler.persona.identity.name} running at http://{host}:{port}  (ctrl-c to stop)")
    ThreadingHTTPServer((host, port), Handler).serve_forever()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Local timeline + DM inbox for the persona bot.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--db", default="content/local.db")
    args = ap.parse_args()
    try:
        serve(args.host, args.port, args.db)
    except KeyboardInterrupt:
        print()
