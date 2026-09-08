"""Local HTTP server for the browser terminal.

Standard library only. A framework would be a dependency, a build step and a second
process to keep alive, for a JSON API with six endpoints served to one browser on one
machine.

**Bound to 127.0.0.1 and nothing else.** There is no authentication, so the bind address is
the access control: anything that can reach the port can start or kill a session. On
loopback that is you. Binding 0.0.0.0 would expose those controls to the local network, so
the address is not configurable from the command line - changing it is a code edit, which
is the right amount of friction for a decision with that consequence.

The session it drives places no real orders: fills are simulated against live prices and
settle against a demo balance. There is no credential path anywhere in this package.
"""

from __future__ import annotations

import json
import logging
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from quantdesk.web.session import PaperSession

log = logging.getLogger(__name__)

STATIC = Path(__file__).resolve().parent / "static"
HOST = "127.0.0.1"

#: Only these Host headers are served. A request arriving with anything else reached us
#: by a route we did not intend, which is worth refusing rather than answering.
_ALLOWED_HOSTS = {"localhost", "127.0.0.1", "[::1]"}


class Handler(BaseHTTPRequestHandler):
    server_version = "quantdesk"
    sys_version = ""

    #: Set by :func:`serve`.
    session: PaperSession = None  # type: ignore[assignment]

    # ------------------------------------------------------------------ plumbing
    def log_message(self, fmt: str, *args) -> None:
        # The default handler writes every request to stderr, which fights with the
        # console output the operator is actually reading.
        log.debug("%s - %s", self.address_string(), fmt % args)

    def _host_ok(self) -> bool:
        host = (self.headers.get("Host") or "").split(":")[0]
        return host in _ALLOWED_HOSTS or host == ""

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        # No framing from anywhere, and no sniffing. Cheap, and this page holds controls.
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            # The browser navigated away mid-response. Not an error worth surfacing.
            pass

    def _json(self, payload: dict, code: int = 200) -> None:
        self._send(code, json.dumps(payload, default=str).encode(), "application/json")

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        if length > 65_536:
            raise ValueError("request body too large")
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"body is not valid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("body must be a JSON object")
        return data

    # ----------------------------------------------------------------------- GET
    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's naming
        if not self._host_ok():
            self._json({"error": "refused: unexpected Host header"}, 403)
            return
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            self._serve_static("index.html", "text/html; charset=utf-8")
        elif path == "/api/state":
            focus = None
            if "?" in self.path:
                from urllib.parse import parse_qs, urlparse

                focus = parse_qs(urlparse(self.path).query).get("focus", [None])[0]
            self._json(self.session.snapshot(focus))
        elif path == "/api/symbols":
            self._symbols()
        elif path == "/api/health":
            self._json({"ok": True, "status": self.session.status})
        else:
            self._json({"error": "not found"}, 404)

    def _symbols(self) -> None:
        """Search the tradeable perpetual universe.

        Several hundred contracts, so results are capped and the query is required to be
        short - this backs a search box, not a bulk export.
        """
        from urllib.parse import parse_qs, urlparse

        params = parse_qs(urlparse(self.path).query)
        needle = (params.get("q", [""])[0] or "")[:24]
        quote = (params.get("quote", ["USDT"])[0] or "USDT")[:8].upper()
        try:
            from quantdesk.data.symbols import UNIVERSE

            hits = UNIVERSE.search(needle, quote=quote, limit=60)
            self._json({
                "total": UNIVERSE.count,
                "query": needle,
                "results": [
                    {
                        "symbol": c.desk_symbol,
                        "venue": c.symbol,
                        "base": c.base,
                        "quote": c.quote,
                        "maint_margin_pct": c.maint_margin_pct,
                        "margin_model_max_leverage": c.max_venue_leverage,
                    }
                    for c in hits
                ],
            })
        except OSError as exc:
            self._json({"error": f"could not reach the venue: {exc}"}, 503)

    def _serve_static(self, name: str, content_type: str) -> None:
        # Fixed filenames only. Joining a request path onto a directory is how a static
        # handler becomes a way to read the rest of the disk.
        target = STATIC / name
        if not target.is_file():
            self._json({"error": f"{name} is missing from the package"}, 500)
            return
        self._send(200, target.read_bytes(), content_type)

    # ---------------------------------------------------------------------- POST
    def do_POST(self) -> None:  # noqa: N802
        if not self._host_ok():
            self._json({"error": "refused: unexpected Host header"}, 403)
            return
        path = self.path.split("?")[0]
        try:
            if path == "/api/start":
                body = self._read_json()
                capital = float(str(body.get("capital", 0)).replace(",", ""))
                raw = body.get("symbols") or []
                if isinstance(raw, str):
                    raw = raw.split(",")
                symbols = [str(s).strip().upper() for s in raw if str(s).strip()]
                timeframe = str(body.get("timeframe") or "1Min")
                max_leverage = float(body.get("max_leverage") or 1.0)
                confidence_leverage = bool(body.get("confidence_leverage"))
                self.session.start(
                    capital, symbols, timeframe, max_leverage, confidence_leverage
                )
                self._json({"ok": True, "status": self.session.status})
            elif path == "/api/pause":
                self._json({"ok": True, "paused": self.session.pause()})
            elif path == "/api/kill":
                self._json({"ok": True, "detail": self.session.kill()})
            elif path == "/api/stop":
                self._json({"ok": True, "detail": self.session.stop()})
            elif path == "/api/reset":
                self.session.reset()
                self._json({"ok": True, "status": self.session.status})
            else:
                self._json({"error": "not found"}, 404)
        except (ValueError, RuntimeError) as exc:
            # Operator error - bad capital, no symbols, session already running. Reported
            # rather than logged as a fault.
            self._json({"error": str(exc)}, 400)
        except Exception as exc:  # noqa: BLE001
            log.exception("request failed: %s", path)
            self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)


def serve(port: int = 8787, open_browser: bool = True) -> int:
    """Run the terminal until interrupted."""
    session = PaperSession()
    Handler.session = session
    httpd = ThreadingHTTPServer((HOST, port), Handler)
    httpd.daemon_threads = True
    url = f"http://{HOST}:{port}/"

    print("=" * 66)
    print("  QUANTDESK PAPER TRADING TERMINAL")
    print("=" * 66)
    print(f"  {url}")
    print()
    print("  Live prices and funding from Binance USD-M perps. No API key.")
    print("  Orders are SIMULATED and settle against a demo balance.")
    print("  No real order can be placed from here.")
    print()
    print("  Bound to loopback only, and there is no password: treat the port")
    print("  as the access control. Ctrl-C to stop.")
    print("=" * 66)

    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping...")
    finally:
        try:
            session.stop()
        except Exception:  # noqa: BLE001 - shutting down anyway
            pass
        httpd.shutdown()
        httpd.server_close()
    return 0
