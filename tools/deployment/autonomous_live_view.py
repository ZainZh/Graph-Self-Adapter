"""Read-only current-frame/stage HTTP view for autonomous shared execution."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading


PAGE = b'''<!doctype html><meta charset="utf-8"><title>Cup autonomous live</title>
<style>body{font:16px sans-serif;background:#111;color:#eee;margin:20px}img{max-width:100%;border:1px solid #555}pre{white-space:pre-wrap}</style>
<h2>Current autonomous cup execution</h2><img id="f"><pre id="s"></pre>
<script>async function u(){f.src='frame.jpg?t='+Date.now();try{s.textContent=JSON.stringify(await(await fetch('status.json?t='+Date.now())).json(),null,2)}catch(e){s.textContent=String(e)}}setInterval(u,500);u()</script>'''


class LiveView:
    def __init__(self, root, host="127.0.0.1", port=18724):
        # The simulator owns creation of its fresh output directory.  Merely
        # constructing the read-only viewer must not consume that safety gate.
        self.root = Path(root)
        self.host = host; self.port = int(port); self.server = None; self.thread = None

    def start(self):
        root = self.root
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                name = self.path.split("?", 1)[0]
                if name == "/": data, kind = PAGE, "text/html; charset=utf-8"
                elif name == "/frame.jpg" and (root / "LATEST_RGB.jpg").is_file(): data, kind = (root / "LATEST_RGB.jpg").read_bytes(), "image/jpeg"
                elif name == "/status.json" and (root / "LIVE_STATUS.json").is_file():
                    status = json.loads((root / "LIVE_STATUS.json").read_text())
                    node = root.parent / "NODE_STATUS.json"
                    if node.is_file(): status["node_progress"] = json.loads(node.read_text())
                    data, kind = json.dumps(status).encode(), "application/json"
                else: self.send_error(404); return
                self.send_response(200); self.send_header("Content-Type", kind); self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(data)
            def log_message(self, *_): pass
        self.server = ThreadingHTTPServer((self.host, self.port), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, name="autonomous-live-view", daemon=True); self.thread.start()
        return f"http://{self.host}:{self.port}/"

    def close(self):
        if self.server: self.server.shutdown(); self.server.server_close(); self.server = None
        if self.thread: self.thread.join(timeout=2); self.thread = None
