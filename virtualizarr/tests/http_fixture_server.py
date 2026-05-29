#!/usr/bin/env python3
"""
Standalone async HTTP fixture server for e2e performance tests.

Must be started with sudo to bind on port 80 (required for icechunk, which
strips non-standard ports from virtual-chunk URLs):

    sudo /path/to/.venv/bin/python http_fixture_server.py --serve-dir /tmp/fixtures

Two servers run concurrently:
  - Port 80   : file server (GET + HEAD, Range/206 support), records stats
  - Port 18080: control API (no sudo needed to reach it from tests)
      GET  /ping          → {"ok": true}
      GET  /stats         → {"get_count": N, "bytes_transferred": N}
      POST /stats/reset   → {"ok": true}

pytest connects to port 18080 to read/reset counters.
icechunk uses http://127.0.0.1/ (port 80, no port suffix in URL).
kerchunk uses http://127.0.0.1:80/ or http://127.0.0.1/ — same thing.
"""

import argparse
import asyncio
import logging
import os
import pathlib
import sys

from aiohttp import web

log = logging.getLogger("fixture_server")

# ---------------------------------------------------------------------------
# Shared stats (atomically updated via asyncio — single-threaded event loop)
# ---------------------------------------------------------------------------

class _Stats:
    def __init__(self) -> None:
        self.get_count: int = 0
        self.bytes_transferred: int = 0

    def record(self, nbytes: int) -> None:
        self.get_count += 1
        self.bytes_transferred += nbytes

    def reset(self) -> None:
        self.get_count = 0
        self.bytes_transferred = 0

    def as_dict(self) -> dict:
        return {
            "get_count": self.get_count,
            "bytes_transferred": self.bytes_transferred,
        }


STATS = _Stats()


# ---------------------------------------------------------------------------
# File server (port 80)
# ---------------------------------------------------------------------------

def _make_file_app(serve_dir: pathlib.Path) -> web.Application:
    app = web.Application()

    async def handle_head(request: web.Request) -> web.Response:
        rel = request.match_info["path"]
        fpath = serve_dir / rel
        if not fpath.is_file():
            raise web.HTTPNotFound()
        size = fpath.stat().st_size
        return web.Response(
            status=200,
            headers={
                "Content-Length": str(size),
                "Accept-Ranges": "bytes",
                "Content-Type": "application/octet-stream",
            },
        )

    async def handle_get(request: web.Request) -> web.StreamResponse:
        rel = request.match_info["path"]
        fpath = serve_dir / rel
        if not fpath.is_file():
            raise web.HTTPNotFound()

        file_size = fpath.stat().st_size
        start, end = 0, file_size  # [start, end)

        range_header = request.headers.get("Range", "")
        partial = False
        if range_header.startswith("bytes="):
            partial = True
            rng = range_header[len("bytes="):]
            s, e = rng.split("-")
            start = int(s) if s else 0
            end = int(e) + 1 if e else file_size

        length = end - start

        if partial:
            resp = web.StreamResponse(
                status=206,
                headers={
                    "Content-Range": f"bytes {start}-{end - 1}/{file_size}",
                    "Content-Length": str(length),
                    "Accept-Ranges": "bytes",
                    "Content-Type": "application/octet-stream",
                },
            )
        else:
            resp = web.StreamResponse(
                status=200,
                headers={
                    "Content-Length": str(length),
                    "Accept-Ranges": "bytes",
                    "Content-Type": "application/octet-stream",
                },
            )

        await resp.prepare(request)

        # Read and send in chunks to avoid loading huge files into memory
        CHUNK = 256 * 1024
        loop = asyncio.get_event_loop()
        with open(fpath, "rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                data = await loop.run_in_executor(
                    None, f.read, min(CHUNK, remaining)
                )
                if not data:
                    break
                await resp.write(data)
                remaining -= len(data)

        STATS.record(length)
        await resp.write_eof()
        return resp

    app.router.add_route("HEAD", "/{path:.+}", handle_head)
    app.router.add_route("GET",  "/{path:.+}", handle_get)
    return app


# ---------------------------------------------------------------------------
# Control API (port 18080)
# ---------------------------------------------------------------------------

def _make_control_app() -> web.Application:
    app = web.Application()

    async def ping(_: web.Request) -> web.Response:
        return web.json_response({"ok": True})

    async def get_stats(_: web.Request) -> web.Response:
        return web.json_response(STATS.as_dict())

    async def reset_stats(_: web.Request) -> web.Response:
        STATS.reset()
        return web.json_response({"ok": True})

    app.router.add_get("/ping", ping)
    app.router.add_get("/stats", get_stats)
    app.router.add_post("/stats/reset", reset_stats)
    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def _main(serve_dir: pathlib.Path, file_port: int, control_port: int) -> None:
    file_app = _make_file_app(serve_dir)
    ctrl_app = _make_control_app()

    file_runner = web.AppRunner(file_app, access_log=None)
    ctrl_runner = web.AppRunner(ctrl_app, access_log=None)

    await file_runner.setup()
    await ctrl_runner.setup()

    file_site = web.TCPSite(file_runner, "127.0.0.1", file_port)
    ctrl_site = web.TCPSite(ctrl_runner, "127.0.0.1", control_port)

    await file_site.start()
    await ctrl_site.start()

    log.info("File server  : http://127.0.0.1:%d/", file_port)
    log.info("Control API  : http://127.0.0.1:%d/", control_port)
    log.info("Serving files from: %s", serve_dir)
    log.info("Press Ctrl-C to stop.")

    # Run until cancelled
    try:
        await asyncio.Event().wait()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        await file_runner.cleanup()
        await ctrl_runner.cleanup()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--serve-dir",
        required=True,
        type=pathlib.Path,
        help="Directory to serve files from",
    )
    parser.add_argument(
        "--file-port",
        type=int,
        default=80,
        help="Port for the file server (default: 80, requires sudo)",
    )
    parser.add_argument(
        "--control-port",
        type=int,
        default=18080,
        help="Port for the control API (default: 18080)",
    )
    args = parser.parse_args()

    if not args.serve_dir.is_dir():
        args.serve_dir.mkdir(parents=True, exist_ok=True)
        log.info("Created serve directory: %s", args.serve_dir)

    if args.file_port < 1024 and os.geteuid() != 0:
        print(
            f"ERROR: file_port={args.file_port} requires root. "
            "Re-run with sudo.",
            file=sys.stderr,
        )
        sys.exit(1)

    asyncio.run(_main(args.serve_dir, args.file_port, args.control_port))


if __name__ == "__main__":
    main()
