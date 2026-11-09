"""Back end for the React chatbot:  python server.py  ->  http://localhost:8000  (run run_project.py once first).
Uses only the standard library plus the project's own modules."""
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

import rag  # noqa: E402
from config import PROCESSED_DIR  # noqa: E402

COPILOT = rag.CoPilot(pd.read_csv(PROCESSED_DIR / "orders_clean.csv"))   # escalations are appended to outputs/escalations.csv


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")          # lets the page work from Live Server (:5500) too
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, (ROOT / "frontend" / "index.html").read_bytes(), "text/html; charset=utf-8")
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/api/chat":
            return self._send(404, {"error": "not found"})
        try:
            payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        except json.JSONDecodeError:
            return self._send(400, {"error": "invalid JSON"})
        self._send(200, COPILOT.respond(str(payload.get("message", ""))))


if __name__ == "__main__":
    print("Serving on http://localhost:8000")
    HTTPServer(("", 8000), Handler).serve_forever()
