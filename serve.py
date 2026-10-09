"""Serve the search app (the static web/ directory) on localhost.

- HTTP range requests, so OpenLayers / geotiff.js can read the COGs tile by tile.
- Cross-origin isolation headers, which onnxruntime-web needs for multi-threaded wasm. --no-isolation leaves them out
  to mimic a static host that can't set headers (web/coi-serviceworker.js then adds them client side).

Open http://localhost:<port>/ (or forward the port over SSH); browsers only allow isolation and the Cache API on
localhost or https.
"""

import argparse
import functools
import http.server
import os
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RANGE = re.compile(r"bytes=(\d*)-(\d*)$")


class Handler(http.server.SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".mjs": "text/javascript",
        ".js": "text/javascript",
        ".wasm": "application/wasm",
        ".onnx": "application/octet-stream",
        ".tif": "image/tiff",
        ".json": "application/json",
        ".map": "application/json",
    }
    verbose = False
    isolation = True

    def end_headers(self):
        if self.isolation:
            self.send_header("Cross-Origin-Opener-Policy", "same-origin")
            self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

    def send_head(self):
        self._remaining = None
        match = RANGE.match(self.headers.get("Range", "").strip())
        path = self.translate_path(self.path)
        if not match or not os.path.isfile(path):
            return super().send_head()
        size = os.path.getsize(path)
        first, last = match.groups()
        if first:
            start, end = int(first), min(int(last) if last else size - 1, size - 1)
        elif last:
            start, end = max(size - int(last), 0), size - 1
        else:
            return super().send_head()
        if start >= size or start > end:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return None
        f = open(path, "rb")
        f.seek(start)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Last-Modified", self.date_time_string(int(os.path.getmtime(path))))
        self.end_headers()
        self._remaining = end - start + 1
        return f

    def copyfile(self, source, outputfile):
        remaining = getattr(self, "_remaining", None)
        if remaining is None:
            return shutil.copyfileobj(source, outputfile)
        while remaining > 0:
            chunk = source.read(min(remaining, 1 << 20))
            if not chunk:
                break
            outputfile.write(chunk)
            remaining -= len(chunk)

    def log_message(self, format, *args):
        if self.verbose or (len(args) > 1 and str(args[1])[:1] in "45"):
            super().log_message(format, *args)


class Server(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        # Browsers abort range requests for tiles that scroll out of view; don't print those.
        if not isinstance(sys.exc_info()[1], (ConnectionError, TimeoutError)):
            super().handle_error(request, client_address)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--verbose", action="store_true", help="Log every request")
    ap.add_argument("--no-isolation", action="store_true", help="Don't send COOP/COEP headers")
    args = ap.parse_args()
    Handler.verbose = args.verbose
    Handler.isolation = not args.no_isolation
    server = Server((args.host, args.port), functools.partial(Handler, directory=ROOT / "web"))
    print(f"serving {ROOT / 'web'} at http://localhost:{args.port}/")
    server.serve_forever()


if __name__ == "__main__":
    main()
