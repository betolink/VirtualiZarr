"""
End-to-end performance + I/O benchmark for the VirtualiZarr workflow:

  virtualize HDF5 files
       → serialize  (kerchunk-parquet | kerchunk-json | icechunk)
           → open with xarray / zarr
               → compute .mean()

Each test measures:
  - wall-clock time per stage
  - HTTP GET count
  - total bytes transferred over the wire

Two modes
---------
**External server mode** (recommended for icechunk tests):
  Start ``http_fixture_server.py`` under sudo on port 80, then run pytest
  with the server's coordinates in env vars::

      sudo .venv/bin/python virtualizarr/tests/http_fixture_server.py \\
          --serve-dir /tmp/vz_fixtures --file-port 80 --control-port 18080

      FIXTURE_SERVER_URL=http://127.0.0.1 \\
      FIXTURE_SERVER_CONTROL_URL=http://127.0.0.1:18080 \\
      FIXTURE_SERVER_DIR=/tmp/vz_fixtures \\
      pytest virtualizarr/tests/test_e2e_performance.py -s

  In this mode pytest writes fixture files directly into FIXTURE_SERVER_DIR
  (no sudo needed for writes) and calls the control API to reset stats.

**In-process server mode** (fallback, no sudo):
  When the env vars are absent pytest starts its own in-process HTTP server
  on a random high port.  Icechunk tests are skipped in this mode because
  icechunk 2.x strips non-standard ports from virtual-chunk URLs.

Two data scenarios
------------------
1. TEMPO-style    – two ragged-scanline granules, pad_to_shape → concat
2. ITS_LIVE-style – two spatial tiles, mosaic_dims=["y","x"]

Kerchunk parquet + missing chunks
----------------------------------
Missing chunks (MISSING_CHUNK_PATH="") are serialised as inlined fill-value
bytes in the kerchunk refs dict, so parquet round-trips cleanly without
storing a None/NaN URL path (which fsspec cannot handle).

Icechunk
--------
Requires ``pip install icechunk``.  Skipped automatically if not installed.
"""
from __future__ import annotations

import http.server
import os
import socket
import threading
import time
import urllib.request
from pathlib import Path

import h5py
import numpy as np
import pytest
import xarray as xr
from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import HTTPStore

from virtualizarr.manifests.array_api import concatenate
from virtualizarr.parsers.hdf import HDFParser
from virtualizarr.xarray import open_virtual_dataset, open_virtual_mfdataset

# ---------------------------------------------------------------------------
# Request stats — two implementations:
#   _LocalStats   : in-process counter (in-process server mode)
#   _RemoteStats  : reads/resets via the external server's control API
# ---------------------------------------------------------------------------

class _LocalStats:
    """Thread-safe stats counter used by the in-process server."""
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.get_count: int = 0
        self.bytes_transferred: int = 0

    def record(self, nbytes: int) -> None:
        with self._lock:
            self.get_count += 1
            self.bytes_transferred += nbytes

    def reset(self) -> None:
        with self._lock:
            self.get_count = 0
            self.bytes_transferred = 0

    def report(self) -> str:
        kb = self.bytes_transferred / 1024
        return f"GETs={self.get_count}, transferred={kb:.1f} KB"


class _RemoteStats:
    """Reads and resets stats via the external server's control API."""
    def __init__(self, control_url: str) -> None:
        self._control = control_url.rstrip("/")
        self._get_count: int = 0
        self._bytes_transferred: int = 0

    def _fetch(self) -> None:
        import json as _json
        with urllib.request.urlopen(f"{self._control}/stats") as r:
            d = _json.loads(r.read())
        self._get_count = d["get_count"]
        self._bytes_transferred = d["bytes_transferred"]

    @property
    def get_count(self) -> int:
        self._fetch()
        return self._get_count

    @property
    def bytes_transferred(self) -> int:
        self._fetch()
        return self._bytes_transferred

    def reset(self) -> None:
        urllib.request.urlopen(
            urllib.request.Request(
                f"{self._control}/stats/reset", method="POST", data=b""
            )
        ).read()
        self._get_count = 0
        self._bytes_transferred = 0

    def report(self) -> str:
        self._fetch()
        kb = self._bytes_transferred / 1024
        return f"GETs={self._get_count}, transferred={kb:.1f} KB"


# keep the old name as an alias so _print_report still works
RequestStats = _LocalStats


# ---------------------------------------------------------------------------
# In-process HTTP server (fallback when external server is not configured)
# ---------------------------------------------------------------------------

def _make_range_handler(directory: Path, stats: _LocalStats):
    """Return an HTTPRequestHandler class serving *directory* with Range support."""

    class _Handler(http.server.BaseHTTPRequestHandler):
        # silence per-request log lines; pytest -s will show our own prints
        def log_message(self, fmt, *args):  # type: ignore[override]
            pass

        def do_HEAD(self) -> None:
            rel = self.path.lstrip("/")
            fpath = directory / rel
            if not fpath.exists() or not fpath.is_file():
                self.send_error(404)
                return
            file_size = fpath.stat().st_size
            self.send_response(200)
            self.send_header("Content-Length", str(file_size))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Type", "application/octet-stream")
            self.end_headers()

        def do_GET(self) -> None:
            rel = self.path.lstrip("/")
            fpath = directory / rel
            if not fpath.exists() or not fpath.is_file():
                self.send_error(404)
                return

            file_size = fpath.stat().st_size
            start, end = 0, file_size  # byte range [start, end)

            range_header = self.headers.get("Range", "")
            partial = False
            if range_header.startswith("bytes="):
                partial = True
                rng = range_header[len("bytes="):]
                s, e = rng.split("-")
                start = int(s) if s else 0
                end = int(e) + 1 if e else file_size

            length = end - start
            data = fpath.read_bytes()[start:end]

            if partial:
                self.send_response(206)
                self.send_header("Content-Range",
                                 f"bytes {start}-{end - 1}/{file_size}")
            else:
                self.send_response(200)

            self.send_header("Content-Length", str(length))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Type", "application/octet-stream")
            self.end_headers()
            self.wfile.write(data)
            stats.record(length)

    return _Handler


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ---------------------------------------------------------------------------
# Session fixture: external server or in-process fallback
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def http_fixture_server(tmp_path_factory):
    """
    Session-scoped fixture.  Two modes, detected automatically:

    **External server mode** — if ``http://127.0.0.1:18080/ping`` responds the
    fixture assumes the external server is running and uses it.  Start it with::

        sudo .venv/bin/python virtualizarr/tests/http_fixture_server.py \\
            --serve-dir virtualizarr/tests/fixtures

    Then just run pytest normally — no env vars required::

        pytest virtualizarr/tests/test_e2e_performance.py -s

    Override defaults via env vars if needed:
      ``FIXTURE_SERVER_URL``         (default ``http://127.0.0.1``)
      ``FIXTURE_SERVER_CONTROL_URL`` (default ``http://127.0.0.1:18080``)
      ``FIXTURE_SERVER_DIR``         (default ``<tests>/fixtures``)

    **In-process fallback** — when the external server is not reachable a
    stdlib HTTP server is started on a random high port.  Icechunk tests are
    skipped in this mode (icechunk 2.x strips non-standard ports).

    Returns a dict with:
      ``base_url``          – HTTP base URL for kerchunk (always works)
      ``icechunk_base_url`` – port-80 URL for icechunk; equals ``base_url``
                              in external mode, triggers skip in fallback mode
      ``root``              – Path to the directory being served
      ``stats``             – stats object (reset per-test via ``http_stats``)
    """
    _tests_dir  = Path(__file__).parent
    ext_url     = os.environ.get("FIXTURE_SERVER_URL",         "http://127.0.0.1").rstrip("/")
    ext_ctrl    = os.environ.get("FIXTURE_SERVER_CONTROL_URL", "http://127.0.0.1:18080").rstrip("/")
    ext_dir     = os.environ.get("FIXTURE_SERVER_DIR",         str(_tests_dir / "fixtures"))

    # Detect external server mode by probing the control API.
    _external = False
    try:
        urllib.request.urlopen(f"{ext_ctrl}/ping", timeout=2).read()
        _external = True
    except Exception:
        pass

    if _external:
        # --- external server mode ---
        root = Path(ext_dir)
        root.mkdir(parents=True, exist_ok=True)
        stats = _RemoteStats(ext_ctrl)
        yield {
            "base_url": ext_url,
            "icechunk_base_url": ext_url,  # already port-80, no port suffix
            "root": root,
            "stats": stats,
        }
        return

    # --- in-process fallback ---
    root = tmp_path_factory.mktemp("http_fixtures")
    stats = _LocalStats()
    port = _free_port()
    handler_cls = _make_range_handler(root, stats)
    server = http.server.HTTPServer(("127.0.0.1", port), handler_cls)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{port}"
    # icechunk needs port 80; non-standard port triggers skip in _icechunk_skip_if_no_port80
    yield {
        "base_url": base_url,
        "icechunk_base_url": base_url,
        "root": root,
        "stats": stats,
    }
    server.shutdown()


@pytest.fixture
def http_stats(http_fixture_server):
    """Per-test fixture: resets the stats counter and returns it."""
    http_fixture_server["stats"].reset()
    return http_fixture_server["stats"]


# ---------------------------------------------------------------------------
# HDF5 factories
# ---------------------------------------------------------------------------

def _make_tempo_like(path: Path, ny: int, nx: int, chunk_ny: int) -> None:
    """Contiguous uncompressed HDF5, TEMPO scan layout."""
    data = np.random.default_rng(0).random((ny, nx)).astype("float32")
    with h5py.File(path, "w") as f:
        f.create_dataset("scanline", data=np.arange(ny, dtype="float32"))
        f.create_dataset("xtrack",   data=np.arange(nx, dtype="float32"))
        f["scanline"].make_scale("scanline")
        f["xtrack"].make_scale("xtrack")
        ds = f.create_dataset("vza", data=data, chunks=(chunk_ny, nx))
        ds.dims[0].attach_scale(f["scanline"])
        ds.dims[1].attach_scale(f["xtrack"])


def _make_itslive_like(path: Path, nx: int, ny: int,
                       x_start: float, y_start: float,
                       time_val: float,
                       step: float = 120.0, chunk: int = 16) -> None:
    """HDF5 with dimension scales, ITS_LIVE tile layout."""
    x   = np.arange(x_start, x_start + nx * step, step, dtype="float64")
    y   = np.arange(y_start, y_start - ny * step, -step, dtype="float64")
    vx  = np.random.default_rng(42).integers(-100, 100,
                                             size=(1, ny, nx), dtype="int16")
    with h5py.File(path, "w") as f:
        f.create_dataset("time", data=np.array([time_val]))
        f.create_dataset("y", data=y)
        f["y"].make_scale("y")
        f.create_dataset("x", data=x)
        f["x"].make_scale("x")
        f["time"].make_scale("time")
        ds = f.create_dataset("vx", data=vx,
                              chunks=(1, min(ny, chunk), min(nx, chunk)))
        ds.dims[0].attach_scale(f["time"])
        ds.dims[1].attach_scale(f["y"])
        ds.dims[2].attach_scale(f["x"])


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

class _Timer:
    def __init__(self) -> None:
        self._label: str = ""
        self._t: float = 0.0
        self.elapsed: dict[str, float] = {}

    def start(self, label: str) -> None:
        self._label = label
        self._t = time.perf_counter()

    def stop(self) -> None:
        self.elapsed[self._label] = time.perf_counter() - self._t

    def total(self) -> float:
        return sum(self.elapsed.values())


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------

def _print_report(prefix: str, timer: _Timer, stats: RequestStats) -> None:
    print()
    for label, secs in timer.elapsed.items():
        print(f"  [{prefix}] {label:<40} {secs * 1000:>8.1f} ms")
    print(f"  [{prefix}] {'--- I/O ---':<40}")
    print(f"  [{prefix}] {'HTTP GETs':<40} {stats.get_count:>8}")
    print(f"  [{prefix}] {'bytes transferred':<40} {stats.bytes_transferred / 1024:>7.1f} KB")
    print(f"  [{prefix}] {'TOTAL time':<40} {timer.total() * 1000:>8.1f} ms")


def _print_comparison(scenario: str,
                      results: dict[str, dict],
                      stages: list[str]) -> None:
    # Keys that hold seconds (→ display as ms); everything else is a raw count or KB.
    _time_keys = {"virtualize", "serialize", "open", "mean()"}
    _kb_keys   = {"compute_kb"}
    _count_keys = {"virtualize_gets", "compute_gets", "real_chunks", "fill_chunks"}

    backends = list(results.keys())
    col = 22
    width = 44 + col * len(backends)
    print(f"\n{'=' * width}")
    print(f"  {scenario}")
    print(f"{'=' * width}")
    header = f"  {'Metric':<42}" + "".join(f"{b:>{col}}" for b in backends)
    print(header)
    print(f"  {'-' * (width - 2)}")
    for stage in stages:
        row = f"  {stage:<42}"
        for b in backends:
            val = results[b].get(stage)
            if val is None:
                row += f"{'—':>{col}}"
            elif stage in _time_keys:
                row += f"{val * 1000:>{col - 5}.1f} ms   "
            elif stage in _kb_keys:
                row += f"{val:>{col - 3}.1f} KB "
            else:
                row += f"{int(val):>{col}}"
        print(row)
    print(f"  {'-' * (width - 2)}")
    # totals row: sum of timing stages only
    row = f"  {'TOTAL time':<42}"
    for b in backends:
        total_ms = sum(v * 1000 for k, v in results[b].items() if k in _time_keys)
        row += f"{total_ms:>{col - 5}.1f} ms   "
    print(row)
    print(f"{'=' * width}\n")


# ---------------------------------------------------------------------------
# Icechunk helpers
# ---------------------------------------------------------------------------
# NOTE: icechunk 2.x normalises virtual-chunk URLs and strips non-standard
# ports (e.g. http://127.0.0.1:PORT/ → http://127.0.0.1/).  The fixture
# server therefore tries to bind on port 80 first (available when pytest is
# run under ``sudo``).  When port 80 is available, ``icechunk_base_url`` in
# the fixture dict is "http://127.0.0.1" (no port suffix) and icechunk can
# resolve virtual chunks over HTTP normally.  When port 80 is not available
# the icechunk tests are skipped with an informative message.


def _icechunk_skip_if_no_port80(icechunk_base_url: str) -> None:
    """Skip the calling test if the server is not on port 80."""
    import urllib.parse
    parsed = urllib.parse.urlparse(icechunk_base_url)
    port = parsed.port  # None means default (80 for http)
    if port is not None and port != 80:
        pytest.skip(
            "icechunk HTTP virtual chunks require port 80 "
            "(icechunk 2.x strips non-standard ports from stored URLs). "
            "Re-run with ``sudo pytest`` to bind the fixture server on port 80."
        )


def _icechunk_write_store(store_path: str, base_url: str):
    """Create a new writable icechunk store with an HTTP VirtualChunkContainer.

    ``base_url`` must be a port-80 URL (e.g. ``http://127.0.0.1``) so that
    icechunk's internal URL normalisation does not strip the port.
    """
    import icechunk
    from icechunk import Repository, Storage

    url_prefix = base_url.rstrip("/") + "/"  # e.g. http://127.0.0.1/

    storage = Storage.new_local_filesystem(store_path)
    config = icechunk.RepositoryConfig.default()
    container = icechunk.VirtualChunkContainer(
        url_prefix=url_prefix,
        store=icechunk.http_store({"allow_http": "true"}),
    )
    config.set_virtual_chunk_container(container)
    repo = Repository.create(
        storage=storage,
        config=config,
        authorize_virtual_chunk_access={url_prefix: None},
    )
    session = repo.writable_session("main")
    return session.store, session


def _icechunk_open_zarr(store_path: str, base_url: str):
    """Open a committed icechunk store and return a zarr group."""
    import zarr
    from icechunk import Repository, Storage

    url_prefix = base_url.rstrip("/") + "/"
    storage = Storage.new_local_filesystem(store_path)
    repo = Repository.open(
        storage,
        authorize_virtual_chunk_access={url_prefix: None},
    )
    session = repo.readonly_session("main")
    return zarr.open_group(session.store, mode="r")


# ---------------------------------------------------------------------------
# Scenario 1: TEMPO-style
#
# Two granules (ny1=96, ny2=128 scanlines, nx=64 cross-track, chunk_ny=32).
# Granule 1 padded to ny2, then manifests concatenated along scanline.
# ---------------------------------------------------------------------------

class TestTempoE2E:
    NY1 = 96
    NY2 = 128
    NX  = 64
    CHUNK_NY = 32

    @pytest.fixture
    def files(self, http_fixture_server):
        root = http_fixture_server["root"]
        base = http_fixture_server["base_url"]
        ic_base = http_fixture_server["icechunk_base_url"]
        p1 = root / "tempo_g1.h5"
        p2 = root / "tempo_g2.h5"
        _make_tempo_like(p1, self.NY1, self.NX, self.CHUNK_NY)
        _make_tempo_like(p2, self.NY2, self.NX, self.CHUNK_NY)
        return p1, p2, base, ic_base

    def _virtualize(self, p1: Path, p2: Path, base_url: str) -> xr.Dataset:
        store = HTTPStore(base_url + "/", client_options={"allow_http": True})
        registry = ObjectStoreRegistry({base_url + "/": store})
        url1 = f"{base_url}/{p1.name}"
        url2 = f"{base_url}/{p2.name}"
        vds1 = open_virtual_dataset(url1, registry=registry,
                                    parser=HDFParser(), loadable_variables=[])
        vds2 = open_virtual_dataset(url2, registry=registry,
                                    parser=HDFParser(), loadable_variables=[])
        ma1 = vds1["vza"].data.pad_to_shape((self.NY2, self.NX))
        ma2 = vds2["vza"].data
        ma  = concatenate([ma1, ma2], axis=0)
        return xr.Dataset({"vza": xr.Variable(("scanline", "xtrack"), ma)})

    # --- kerchunk parquet ---

    def test_kerchunk_parquet(self, files, tmp_path, http_stats):
        p1, p2, base_url, _ic_base = files
        timer = _Timer()

        http_stats.reset()
        timer.start("virtualize")
        vds = self._virtualize(p1, p2, base_url)
        timer.stop()
        virtualize_gets = http_stats.get_count
        virtualize_bytes = http_stats.bytes_transferred

        path = str(tmp_path / "tempo.parquet")
        http_stats.reset()
        timer.start("serialize → kerchunk parquet")
        vds.vz.to_kerchunk(path, format="parquet")
        timer.stop()

        http_stats.reset()
        timer.start("xr.open_dataset")
        ds = xr.open_dataset(path, engine="kerchunk",
                             storage_options={"skip_instance_cache": True})
        timer.stop()

        http_stats.reset()
        timer.start("vza.mean().compute()")
        result = float(ds["vza"].mean())
        timer.stop()
        compute_gets = http_stats.get_count
        compute_bytes = http_stats.bytes_transferred

        _print_report("TEMPO kerchunk-parquet", timer, http_stats)
        print(f"  [TEMPO kerchunk-parquet] {'virtualize GETs':<40} {virtualize_gets:>8}")
        print(f"  [TEMPO kerchunk-parquet] {'virtualize bytes':<40} {virtualize_bytes / 1024:>7.1f} KB")
        print(f"  [TEMPO kerchunk-parquet] {'compute GETs':<40} {compute_gets:>8}")
        print(f"  [TEMPO kerchunk-parquet] {'compute bytes':<40} {compute_bytes / 1024:>7.1f} KB")
        assert np.isfinite(result)

    # --- kerchunk JSON ---

    def test_kerchunk_json(self, files, tmp_path, http_stats):
        p1, p2, base_url, _ic_base = files
        timer = _Timer()

        http_stats.reset()
        timer.start("virtualize")
        vds = self._virtualize(p1, p2, base_url)
        timer.stop()
        virtualize_gets = http_stats.get_count

        path = str(tmp_path / "tempo.json")
        http_stats.reset()
        timer.start("serialize → kerchunk json")
        vds.vz.to_kerchunk(path, format="json")
        timer.stop()

        http_stats.reset()
        timer.start("xr.open_dataset")
        ds = xr.open_dataset(path, engine="kerchunk",
                             storage_options={"skip_instance_cache": True})
        timer.stop()

        http_stats.reset()
        timer.start("vza.mean().compute()")
        result = float(ds["vza"].mean())
        timer.stop()
        compute_gets = http_stats.get_count
        compute_bytes = http_stats.bytes_transferred

        _print_report("TEMPO kerchunk-json", timer, http_stats)
        print(f"  [TEMPO kerchunk-json] {'virtualize GETs':<40} {virtualize_gets:>8}")
        print(f"  [TEMPO kerchunk-json] {'compute GETs':<40} {compute_gets:>8}")
        print(f"  [TEMPO kerchunk-json] {'compute bytes':<40} {compute_bytes / 1024:>7.1f} KB")
        assert np.isfinite(result)

    # --- icechunk ---

    def test_icechunk(self, files, tmp_path, http_stats):
        pytest.importorskip("icechunk")
        p1, p2, base_url, ic_base = files
        _icechunk_skip_if_no_port80(ic_base)
        timer = _Timer()

        http_stats.reset()
        timer.start("virtualize")
        vds = self._virtualize(p1, p2, base_url)
        timer.stop()
        virtualize_gets = http_stats.get_count

        ic_path = str(tmp_path / "tempo_ic")
        ic_store, _ = _icechunk_write_store(ic_path, ic_base)

        http_stats.reset()
        timer.start("serialize → icechunk")
        vds.vz.to_icechunk(ic_store)
        ic_store.session.commit("tempo bench")
        timer.stop()

        timer.start("zarr.open (icechunk)")
        z = _icechunk_open_zarr(ic_path, ic_base)
        timer.stop()

        http_stats.reset()
        timer.start("vza.mean().compute()")
        result = float(np.nanmean(z["vza"][:]))
        timer.stop()
        compute_gets = http_stats.get_count
        compute_bytes = http_stats.bytes_transferred

        _print_report("TEMPO icechunk", timer, http_stats)
        print(f"  [TEMPO icechunk] {'virtualize GETs':<40} {virtualize_gets:>8}")
        print(f"  [TEMPO icechunk] {'compute GETs':<40} {compute_gets:>8}")
        print(f"  [TEMPO icechunk] {'compute bytes':<40} {compute_bytes / 1024:>7.1f} KB")
        assert np.isfinite(result)

    # --- comparison table ---

    def test_comparison_table(self, files, tmp_path, http_stats):
        pytest.importorskip("icechunk")
        p1, p2, base_url, ic_base = files
        _icechunk_skip_if_no_port80(ic_base)
        results: dict[str, dict] = {}

        for backend in ("kerchunk-parquet", "kerchunk-json", "icechunk"):
            timer = _Timer()
            row: dict = {}

            http_stats.reset()
            timer.start("virtualize")
            vds = self._virtualize(p1, p2, base_url)
            timer.stop()
            row["virtualize"] = timer.elapsed["virtualize"]
            row["virtualize_gets"] = http_stats.get_count

            if backend == "kerchunk-parquet":
                path = str(tmp_path / "t_kc.parquet")
                http_stats.reset()
                timer.start("serialize")
                vds.vz.to_kerchunk(path, format="parquet")
                timer.stop()
                row["serialize"] = timer.elapsed["serialize"]
                timer.start("open")
                ds = xr.open_dataset(path, engine="kerchunk",
                                     storage_options={"skip_instance_cache": True})
                timer.stop()
                row["open"] = timer.elapsed["open"]
                http_stats.reset()
                timer.start("mean()")
                float(ds["vza"].mean())
                timer.stop()
                row["mean()"] = timer.elapsed["mean()"]

            elif backend == "kerchunk-json":
                path = str(tmp_path / "t_kc.json")
                http_stats.reset()
                timer.start("serialize")
                vds.vz.to_kerchunk(path, format="json")
                timer.stop()
                row["serialize"] = timer.elapsed["serialize"]
                timer.start("open")
                ds = xr.open_dataset(path, engine="kerchunk",
                                     storage_options={"skip_instance_cache": True})
                timer.stop()
                row["open"] = timer.elapsed["open"]
                http_stats.reset()
                timer.start("mean()")
                float(ds["vza"].mean())
                timer.stop()
                row["mean()"] = timer.elapsed["mean()"]

            else:  # icechunk
                ic_path = str(tmp_path / "t_ic")
                ic_store, _ = _icechunk_write_store(ic_path, ic_base)
                http_stats.reset()
                timer.start("serialize")
                vds.vz.to_icechunk(ic_store)
                ic_store.session.commit("bench")
                timer.stop()
                row["serialize"] = timer.elapsed["serialize"]
                timer.start("open")
                z = _icechunk_open_zarr(ic_path, ic_base)
                timer.stop()
                row["open"] = timer.elapsed["open"]
                http_stats.reset()
                timer.start("mean()")
                float(np.nanmean(z["vza"][:]))
                timer.stop()
                row["mean()"] = timer.elapsed["mean()"]

            row["compute_gets"] = http_stats.get_count
            row["compute_kb"]   = http_stats.bytes_transferred / 1024
            results[backend] = row

        _print_comparison("TEMPO (pad_to_shape)", results,
                          ["virtualize", "serialize", "open", "mean()",
                           "virtualize_gets", "compute_gets", "compute_kb"])


# ---------------------------------------------------------------------------
# Scenario 2: ITS_LIVE-style mosaic
#
# Real-world context
# ------------------
# ITS_LIVE stores glacier velocity as HDF5/NetCDF4 files, one per UTM tile per
# time step.  Each file has a fixed tile extent (e.g. 120 km × 120 km) and a
# fixed chunk shape (e.g. 1 × 10 × 10 at 1 km resolution → 120×120 tile =
# 12×12 = 144 chunks per variable per time step).
#
# When a user calls open_virtual_mfdataset(..., mosaic_dims=["y","x"]) over
# two adjacent tiles the library must:
#   1. _remap_chunks → split each (1,16,16) chunk into (1,1,1) sub-chunks so
#      arbitrary element-level offsets can be expressed in the manifest.
#   2. place_in_grid  → place those sub-chunks at the correct position in the
#      union coordinate grid.
#   3. consolidate_chunks → merge the sub-chunks back to (1,16,16), recovering
#      the original byte ranges, provided the union shape is divisible by the
#      chunk size on every mosaic axis.
#
# When consolidation succeeds the user pays exactly the same number of HTTP
# GETs as if they had opened the two tiles individually — there is no overhead
# from the mosaic operation at read time.
#
# Fixture choice
# --------------
# NX1=32, NX2=48, NY=48, CHUNK=16.
# Union shape: (2, 48, 80).  48 % 16 == 0 and 80 % 16 == 0 → full
# consolidation → chunks restored to (1,16,16).
#
# Expected GETs (= original chunk grid cells that contain real data):
#   tile 1: 1 time × ceil(48/16) × ceil(32/16) = 1×3×2 = 6
#   tile 2: 1 time × ceil(48/16) × ceil(48/16) = 1×3×3 = 9
#   total  = 15
#
# Non-aligned tiles (real-world edge case)
# ----------------------------------------
# If tiles have widths whose sum is NOT a multiple of the chunk size (e.g.
# NX1=32, NX2=40 → union_nx=72, 72%16=8≠0) consolidation is skipped on the
# x-axis and the manifest retains chunk_size=1 in x — meaning one GET per
# element-column.  This is safe (correct data) but inefficient.  In practice
# real ITS_LIVE tiles are multiples of the chunk size, so this never occurs.
# The degenerate case is covered by TestITSLiveE2ENonAligned below.
# ---------------------------------------------------------------------------

class TestITSLiveE2E:
    NX1   = 32
    NX2   = 48
    NY    = 48
    STEP  = 120.0
    CHUNK = 16

    @pytest.fixture
    def files(self, http_fixture_server):
        root = http_fixture_server["root"]
        base = http_fixture_server["base_url"]
        ic_base = http_fixture_server["icechunk_base_url"]
        p1 = root / "itslive_t1.h5"
        p2 = root / "itslive_t2.h5"
        _make_itslive_like(p1, nx=self.NX1, ny=self.NY,
                           x_start=0.0, y_start=self.NY * self.STEP,
                           time_val=1.0, step=self.STEP, chunk=self.CHUNK)
        _make_itslive_like(p2, nx=self.NX2, ny=self.NY,
                           x_start=self.NX1 * self.STEP,
                           y_start=self.NY * self.STEP,
                           time_val=2.0, step=self.STEP, chunk=self.CHUNK)
        return p1, p2, base, ic_base

    def _virtualize(self, p1: Path, p2: Path, base_url: str) -> xr.Dataset:
        store = HTTPStore(base_url + "/", client_options={"allow_http": True})
        registry = ObjectStoreRegistry({base_url + "/": store})
        url1 = f"{base_url}/{p1.name}"
        url2 = f"{base_url}/{p2.name}"
        return open_virtual_mfdataset(
            [url1, url2],
            registry=registry,
            parser=HDFParser(),
            combine="nested",
            concat_dim="time",
            loadable_variables=["x", "y", "time"],
            mosaic_dims=["y", "x"],
            pad="none",
        )

    # --- kerchunk parquet ---

    def test_kerchunk_parquet(self, files, tmp_path, http_stats):
        p1, p2, base_url, _ic_base = files
        timer = _Timer()

        http_stats.reset()
        timer.start("virtualize + mosaic")
        vds = self._virtualize(p1, p2, base_url)
        timer.stop()
        virtualize_gets = http_stats.get_count
        virtualize_bytes = http_stats.bytes_transferred

        ma = vds["vx"].data
        real = int((ma.manifest._paths != "").sum())
        fill = int((ma.manifest._paths == "").sum())

        path = str(tmp_path / "itslive.parquet")
        http_stats.reset()
        timer.start("serialize → kerchunk parquet")
        vds.vz.to_kerchunk(path, format="parquet")
        timer.stop()

        http_stats.reset()
        timer.start("xr.open_dataset")
        ds = xr.open_dataset(path, engine="kerchunk",
                             storage_options={"skip_instance_cache": True})
        timer.stop()

        http_stats.reset()
        timer.start("vx.mean().compute()")
        result = float(ds["vx"].mean())
        timer.stop()
        compute_gets = http_stats.get_count
        compute_bytes = http_stats.bytes_transferred

        _print_report("ITS_LIVE kerchunk-parquet", timer, http_stats)
        print(f"  [ITS_LIVE kerchunk-parquet] {'real chunks (GETs)':<40} {real:>8}")
        print(f"  [ITS_LIVE kerchunk-parquet] {'fill chunks (inlined, free)':<40} {fill:>8}")
        print(f"  [ITS_LIVE kerchunk-parquet] {'virtualize GETs':<40} {virtualize_gets:>8}")
        print(f"  [ITS_LIVE kerchunk-parquet] {'virtualize bytes':<40} {virtualize_bytes / 1024:>7.1f} KB")
        print(f"  [ITS_LIVE kerchunk-parquet] {'compute GETs':<40} {compute_gets:>8}")
        print(f"  [ITS_LIVE kerchunk-parquet] {'compute bytes':<40} {compute_bytes / 1024:>7.1f} KB")
        assert np.isfinite(result)

    # --- kerchunk JSON ---

    def test_kerchunk_json(self, files, tmp_path, http_stats):
        p1, p2, base_url, _ic_base = files
        timer = _Timer()

        http_stats.reset()
        timer.start("virtualize + mosaic")
        vds = self._virtualize(p1, p2, base_url)
        timer.stop()
        virtualize_gets = http_stats.get_count

        ma = vds["vx"].data
        real = int((ma.manifest._paths != "").sum())
        fill = int((ma.manifest._paths == "").sum())

        path = str(tmp_path / "itslive.json")
        http_stats.reset()
        timer.start("serialize → kerchunk json")
        vds.vz.to_kerchunk(path, format="json")
        timer.stop()

        http_stats.reset()
        timer.start("xr.open_dataset")
        ds = xr.open_dataset(path, engine="kerchunk",
                             storage_options={"skip_instance_cache": True})
        timer.stop()

        http_stats.reset()
        timer.start("vx.mean().compute()")
        result = float(ds["vx"].mean())
        timer.stop()
        compute_gets = http_stats.get_count
        compute_bytes = http_stats.bytes_transferred

        _print_report("ITS_LIVE kerchunk-json", timer, http_stats)
        print(f"  [ITS_LIVE kerchunk-json] {'real chunks (GETs)':<40} {real:>8}")
        print(f"  [ITS_LIVE kerchunk-json] {'fill chunks (absent, free)':<40} {fill:>8}")
        print(f"  [ITS_LIVE kerchunk-json] {'virtualize GETs':<40} {virtualize_gets:>8}")
        print(f"  [ITS_LIVE kerchunk-json] {'compute GETs':<40} {compute_gets:>8}")
        print(f"  [ITS_LIVE kerchunk-json] {'compute bytes':<40} {compute_bytes / 1024:>7.1f} KB")
        assert np.isfinite(result)

    # --- icechunk ---

    def test_icechunk(self, files, tmp_path, http_stats):
        pytest.importorskip("icechunk")
        p1, p2, base_url, ic_base = files
        _icechunk_skip_if_no_port80(ic_base)
        timer = _Timer()

        http_stats.reset()
        timer.start("virtualize + mosaic")
        vds = self._virtualize(p1, p2, base_url)
        timer.stop()
        virtualize_gets = http_stats.get_count

        ic_path = str(tmp_path / "itslive_ic")
        ic_store, _ = _icechunk_write_store(ic_path, ic_base)

        http_stats.reset()
        timer.start("serialize → icechunk")
        vds.vz.to_icechunk(ic_store)
        ic_store.session.commit("itslive bench")
        timer.stop()

        timer.start("zarr.open (icechunk)")
        z = _icechunk_open_zarr(ic_path, ic_base)
        timer.stop()

        http_stats.reset()
        timer.start("vx.mean().compute()")
        result = float(np.nanmean(z["vx"][:].astype("float32")))
        timer.stop()
        compute_gets = http_stats.get_count
        compute_bytes = http_stats.bytes_transferred

        _print_report("ITS_LIVE icechunk", timer, http_stats)
        print(f"  [ITS_LIVE icechunk] {'virtualize GETs':<40} {virtualize_gets:>8}")
        print(f"  [ITS_LIVE icechunk] {'compute GETs':<40} {compute_gets:>8}")
        print(f"  [ITS_LIVE icechunk] {'compute bytes':<40} {compute_bytes / 1024:>7.1f} KB")
        assert np.isfinite(result)

    # --- comparison table ---

    def test_comparison_table(self, files, tmp_path, http_stats):
        pytest.importorskip("icechunk")
        p1, p2, base_url, ic_base = files
        _icechunk_skip_if_no_port80(ic_base)
        results: dict[str, dict] = {}

        for backend in ("kerchunk-parquet", "kerchunk-json", "icechunk"):
            timer = _Timer()
            row: dict = {}

            http_stats.reset()
            timer.start("virtualize")
            vds = self._virtualize(p1, p2, base_url)
            timer.stop()
            row["virtualize"] = timer.elapsed["virtualize"]
            row["virtualize_gets"] = http_stats.get_count

            if backend == "kerchunk-parquet":
                path = str(tmp_path / "ils_kc.parquet")
                http_stats.reset()
                timer.start("serialize")
                vds.vz.to_kerchunk(path, format="parquet")
                timer.stop()
                row["serialize"] = timer.elapsed["serialize"]
                timer.start("open")
                ds = xr.open_dataset(path, engine="kerchunk",
                                     storage_options={"skip_instance_cache": True})
                timer.stop()
                row["open"] = timer.elapsed["open"]
                http_stats.reset()
                timer.start("mean()")
                float(ds["vx"].mean())
                timer.stop()
                row["mean()"] = timer.elapsed["mean()"]

            elif backend == "kerchunk-json":
                path = str(tmp_path / "ils_kc.json")
                http_stats.reset()
                timer.start("serialize")
                vds.vz.to_kerchunk(path, format="json")
                timer.stop()
                row["serialize"] = timer.elapsed["serialize"]
                timer.start("open")
                ds = xr.open_dataset(path, engine="kerchunk",
                                     storage_options={"skip_instance_cache": True})
                timer.stop()
                row["open"] = timer.elapsed["open"]
                http_stats.reset()
                timer.start("mean()")
                float(ds["vx"].mean())
                timer.stop()
                row["mean()"] = timer.elapsed["mean()"]

            else:  # icechunk
                ic_path = str(tmp_path / "ils_ic")
                ic_store, _ = _icechunk_write_store(ic_path, ic_base)
                http_stats.reset()
                timer.start("serialize")
                vds.vz.to_icechunk(ic_store)
                ic_store.session.commit("bench")
                timer.stop()
                row["serialize"] = timer.elapsed["serialize"]
                timer.start("open")
                z = _icechunk_open_zarr(ic_path, ic_base)
                timer.stop()
                row["open"] = timer.elapsed["open"]
                http_stats.reset()
                timer.start("mean()")
                float(np.nanmean(z["vx"][:].astype("float32")))
                timer.stop()
                row["mean()"] = timer.elapsed["mean()"]

            row["compute_gets"] = http_stats.get_count
            row["compute_kb"]   = http_stats.bytes_transferred / 1024
            results[backend] = row

        _print_comparison("ITS_LIVE mosaic (mosaic_dims)", results,
                          ["virtualize", "serialize", "open", "mean()",
                           "virtualize_gets", "compute_gets", "compute_kb"])


# ---------------------------------------------------------------------------
# Scenario 2b: ITS_LIVE non-aligned mosaic (worst-case)
#
# NX1=32, NX2=40 → union_nx=72.  72 % 16 == 8 ≠ 0 so consolidation is
# skipped on the x-axis and the manifest retains chunk_size=1 in x.
# The data is correct but compute_gets will equal the number of real *elements*
# along x within each tile (32+40=72 element-columns × 3 y-stripes × 2 time
# steps that have data = 3456).  This test documents that degenerate behaviour
# so it is visible in CI and can be improved in the future (e.g. by supporting
# partial-chunk merging or requiring chunk-aligned tile boundaries).
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Scenario 3: Malaspina glacier — 20 real ITS_LIVE Landsat granules (2025)
#
# Real-world context
# ------------------
# These are genuine ITS_LIVE v02 velocity image-pair granules from WRS path/row
# 063018 (same UTM zone, EPSG:3413) covering the Malaspina / Agassiz glacier in
# Alaska.  Each file is a NetCDF4/HDF5 on public S3 with shape (1, *, *) and
# chunk size (1, 512, 512) at 120 m resolution.  The spatial footprints vary
# between granules (different image-pair geometries) so a spatial mosaic is
# required before concatenating along time.
#
# Workflow
# --------
#   open_virtual_mfdataset(20 S3 URLs,
#       mosaic_dims=["y","x"],   ← build union spatial grid, pad missing regions
#       concat_dim="time",       ← stack the 20 time steps
#       loadable_variables=["x","y","time"],
#   )
#   → serialize to kerchunk-json / kerchunk-parquet / icechunk
#   → open with xarray / zarr and compute v.mean() over time
#
# No local fixture server is needed — the files are served from public S3.
# GET counts are not intercepted for remote S3; the tests assert correctness
# (finite mean) and print wall-clock timing for each stage.
#
# Icechunk note
# -------------
# Icechunk requires virtual-chunk URLs to be registered in a
# VirtualChunkContainer.  For S3 we use the ``s3://`` scheme with anonymous
# access (the bucket is public).  The test is skipped if icechunk is not
# installed.
# ---------------------------------------------------------------------------

MALASPINA_URLS: list[str] = [
    "https://its-live-data.s3.amazonaws.com/velocity_image_pair/landsatOLI/v02/N60W140/LC08_L1TP_063018_20250929_20251002_02_T1_X_LC09_L1TP_063018_20251007_20251007_02_T1_G0120V02_P067.nc",
    "https://its-live-data.s3.amazonaws.com/velocity_image_pair/landsatOLI/v02/N60W140/LC08_L1TP_063018_20250929_20251002_02_T1_X_LC09_L1TP_063018_20251108_20251108_02_T2_G0120V02_P022.nc",
    "https://its-live-data.s3.amazonaws.com/velocity_image_pair/landsatOLI/v02/N60W140/LC08_L1TP_063018_20250929_20251002_02_T1_X_LC08_L1GT_063018_20251116_20251202_02_T2_G0120V02_P006.nc",
    "https://its-live-data.s3.amazonaws.com/velocity_image_pair/landsatOLI/v02/N60W140/LC09_L1TP_063018_20251007_20251007_02_T1_X_LC09_L1TP_063018_20251108_20251108_02_T2_G0120V02_P020.nc",
    "https://its-live-data.s3.amazonaws.com/velocity_image_pair/landsatOLI/v02/N60W140/LC08_L1TP_063018_20250929_20251002_02_T1_X_LC09_L1TP_063018_20251124_20251124_02_T1_G0120V02_P033.nc",
    "https://its-live-data.s3.amazonaws.com/velocity_image_pair/landsatOLI/v02/N60W140/LC09_L1TP_063018_20251007_20251007_02_T1_X_LC08_L1GT_063018_20251116_20251202_02_T2_G0120V02_P006.nc",
    "https://its-live-data.s3.amazonaws.com/velocity_image_pair/landsatOLI/v02/N60W140/LC09_L1TP_063018_20251007_20251007_02_T1_X_LC09_L1TP_063018_20251124_20251124_02_T1_G0120V02_P032.nc",
    "https://its-live-data.s3.amazonaws.com/velocity_image_pair/landsatOLI/v02/N60W140/LC08_L1TP_063018_20250929_20251002_02_T1_X_LC09_L1TP_063018_20251210_20251210_02_T2_G0120V02_P045.nc",
    "https://its-live-data.s3.amazonaws.com/velocity_image_pair/landsatOLI/v02/N60W140/LC08_L1TP_063018_20250929_20251002_02_T1_X_LC08_L1TP_063018_20251218_20251225_02_T1_G0120V02_P039.nc",
    "https://its-live-data.s3.amazonaws.com/velocity_image_pair/landsatOLI/v02/N60W140/LC09_L1TP_063018_20251007_20251007_02_T1_X_LC09_L1TP_063018_20251210_20251210_02_T2_G0120V02_P048.nc",
    "https://its-live-data.s3.amazonaws.com/velocity_image_pair/landsatOLI/v02/N60W140/LC09_L1TP_063018_20251007_20251007_02_T1_X_LC08_L1TP_063018_20251218_20251225_02_T1_G0120V02_P043.nc",
    "https://its-live-data.s3.amazonaws.com/velocity_image_pair/landsatOLI/v02/N60W140/LC09_L1TP_063018_20251108_20251108_02_T2_X_LC08_L1GT_063018_20251116_20251202_02_T2_G0120V02_P010.nc",
    "https://its-live-data.s3.amazonaws.com/velocity_image_pair/landsatOLI/v02/N60W140/LC09_L1TP_063018_20251108_20251108_02_T2_X_LC09_L1TP_063018_20251124_20251124_02_T1_G0120V02_P036.nc",
    "https://its-live-data.s3.amazonaws.com/velocity_image_pair/landsatOLI/v02/N60W140/LC08_L1TP_063018_20250929_20251002_02_T1_X_LC09_L1GT_063018_20260111_20260111_02_T2_G0120V02_P019.nc",
    "https://its-live-data.s3.amazonaws.com/velocity_image_pair/landsatOLI/v02/N60W140/LC09_L1TP_063018_20251007_20251007_02_T1_X_LC09_L1GT_063018_20260111_20260111_02_T2_G0120V02_P017.nc",
    "https://its-live-data.s3.amazonaws.com/velocity_image_pair/landsatOLI/v02/N60W140/LC09_L1TP_063018_20251108_20251108_02_T2_X_LC09_L1TP_063018_20251210_20251210_02_T2_G0120V02_P035.nc",
    "https://its-live-data.s3.amazonaws.com/velocity_image_pair/landsatOLI/v02/N60W140/LC08_L1GT_063018_20251116_20251202_02_T2_X_LC09_L1TP_063018_20251210_20251210_02_T2_G0120V02_P012.nc",
    "https://its-live-data.s3.amazonaws.com/velocity_image_pair/landsatOLI/v02/N60W140/LC09_L1TP_063018_20251108_20251108_02_T2_X_LC08_L1TP_063018_20251218_20251225_02_T1_G0120V02_P032.nc",
    "https://its-live-data.s3.amazonaws.com/velocity_image_pair/landsatOLI/v02/N60W140/LC08_L1GT_063018_20251116_20251202_02_T2_X_LC08_L1TP_063018_20251218_20251225_02_T1_G0120V02_P012.nc",
    "https://its-live-data.s3.amazonaws.com/velocity_image_pair/landsatOLI/v02/N60W140/LC09_L1TP_063018_20251124_20251124_02_T1_X_LC09_L1TP_063018_20251210_20251210_02_T2_G0120V02_P052.nc",
]


@pytest.mark.network
class TestMalaspinaE2E:
    """
    End-to-end test with 20 real ITS_LIVE Landsat granules over Malaspina
    glacier (WRS 063018, 2025).  Exercises the full pipeline:

        open_virtual_mfdataset (mosaic_dims + concat_dim)
            → kerchunk-json / kerchunk-parquet / icechunk
                → xarray open → v.mean()

    Marked ``network`` — skip with ``-m 'not network'`` for offline CI.
    """

    # Object-dtype variables (mapping, img_pair_info) cannot be virtualized
    # with zarr v3 — drop them at parse time.
    _DROP = ["mapping", "img_pair_info"]
    _BASE = "https://its-live-data.s3.amazonaws.com/"

    def _virtualize(self) -> xr.Dataset:
        store = HTTPStore(self._BASE)
        registry = ObjectStoreRegistry({self._BASE: store})
        return open_virtual_mfdataset(
            MALASPINA_URLS,
            registry=registry,
            parser=HDFParser(drop_variables=self._DROP),
            combine="nested",
            concat_dim="time",
            loadable_variables=["x", "y", "time"],
            mosaic_dims=["y", "x"],
            pad="none",
        )

    def _print_result(self, prefix: str, timer: _Timer,
                      n_granules: int, union_shape: tuple,
                      real_chunks: int, fill_chunks: int) -> None:
        print(f"\n{'=' * 60}")
        print(f"  {prefix}")
        print(f"{'=' * 60}")
        print(f"  {'granules virtualized':<40} {n_granules:>8}")
        print(f"  {'union shape (time, y, x)':<40} {str(union_shape):>16}")
        print(f"  {'real manifest chunks':<40} {real_chunks:>8}")
        print(f"  {'fill manifest chunks':<40} {fill_chunks:>8}")
        for label, secs in timer.elapsed.items():
            print(f"  {label:<40} {secs * 1000:>8.1f} ms")
        print(f"  {'TOTAL':<40} {timer.total() * 1000:>8.1f} ms")
        print(f"{'=' * 60}")

    # --- kerchunk JSON ---

    def test_kerchunk_json(self, tmp_path):
        timer = _Timer()

        timer.start("virtualize + mosaic (20 granules)")
        vds = self._virtualize()
        timer.stop()

        ma = vds["vx"].data
        union_shape = ma.shape
        real_chunks = int((ma.manifest._paths != "").sum())
        fill_chunks = int((ma.manifest._paths == "").sum())

        path = str(tmp_path / "malaspina.json")
        timer.start("serialize → kerchunk json")
        vds.vz.to_kerchunk(path, format="json")
        timer.stop()

        timer.start("xr.open_dataset")
        ds = xr.open_dataset(path, engine="kerchunk",
                             storage_options={"skip_instance_cache": True})
        timer.stop()

        timer.start("v.mean().compute()")
        result = float(ds["v"].mean())
        timer.stop()

        self._print_result("Malaspina kerchunk-json", timer,
                           len(MALASPINA_URLS), union_shape,
                           real_chunks, fill_chunks)
        assert np.isfinite(result), f"v.mean() returned non-finite: {result}"

    # --- kerchunk parquet ---

    def test_kerchunk_parquet(self, tmp_path):
        timer = _Timer()

        timer.start("virtualize + mosaic (20 granules)")
        vds = self._virtualize()
        timer.stop()

        ma = vds["vx"].data
        union_shape = ma.shape
        real_chunks = int((ma.manifest._paths != "").sum())
        fill_chunks = int((ma.manifest._paths == "").sum())

        path = str(tmp_path / "malaspina.parquet")
        timer.start("serialize → kerchunk parquet")
        vds.vz.to_kerchunk(path, format="parquet")
        timer.stop()

        timer.start("xr.open_dataset")
        ds = xr.open_dataset(path, engine="kerchunk",
                             storage_options={"skip_instance_cache": True})
        timer.stop()

        timer.start("v.mean().compute()")
        result = float(ds["v"].mean())
        timer.stop()

        self._print_result("Malaspina kerchunk-parquet", timer,
                           len(MALASPINA_URLS), union_shape,
                           real_chunks, fill_chunks)
        assert np.isfinite(result), f"v.mean() returned non-finite: {result}"

    # --- icechunk ---

    def test_icechunk(self, tmp_path):
        pytest.importorskip("icechunk")
        import icechunk
        from icechunk import Repository, Storage
        timer = _Timer()

        timer.start("virtualize + mosaic (20 granules)")
        vds = self._virtualize()
        timer.stop()

        ma = vds["vx"].data
        union_shape = ma.shape
        real_chunks = int((ma.manifest._paths != "").sum())
        fill_chunks = int((ma.manifest._paths == "").sum())

        # icechunk with HTTP virtual chunks (manifest paths are https://)
        ic_path = str(tmp_path / "malaspina_ic")
        url_prefix = "https://its-live-data.s3.amazonaws.com/"
        storage = Storage.new_local_filesystem(ic_path)
        config = icechunk.RepositoryConfig.default()
        container = icechunk.VirtualChunkContainer(
            url_prefix=url_prefix,
            store=icechunk.http_store(),
        )
        config.set_virtual_chunk_container(container)
        repo = Repository.create(
            storage=storage,
            config=config,
            authorize_virtual_chunk_access={url_prefix: None},
        )
        session = repo.writable_session("main")
        ic_store = session.store

        timer.start("serialize → icechunk")
        vds.vz.to_icechunk(ic_store)
        session.commit("malaspina bench")
        timer.stop()

        timer.start("zarr.open (icechunk)")
        import zarr
        read_repo = Repository.open(
            storage,
            authorize_virtual_chunk_access={url_prefix: None},
        )
        read_session = read_repo.readonly_session("main")
        z = zarr.open_group(read_session.store, mode="r")
        timer.stop()

        timer.start("v.mean().compute()")
        result = float(np.nanmean(z["v"][:].astype("float32")))
        timer.stop()

        self._print_result("Malaspina icechunk", timer,
                           len(MALASPINA_URLS), union_shape,
                           real_chunks, fill_chunks)
        assert np.isfinite(result), f"v.mean() returned non-finite: {result}"

    # --- comparison table ---

    def test_comparison_table(self, tmp_path):
        pytest.importorskip("icechunk")
        results: dict[str, dict] = {}

        for backend in ("kerchunk-json", "kerchunk-parquet", "icechunk"):
            timer = _Timer()
            row: dict = {}

            timer.start("virtualize")
            vds = self._virtualize()
            timer.stop()
            row["virtualize"] = timer.elapsed["virtualize"]

            ma = vds["vx"].data
            row["real_chunks"] = int((ma.manifest._paths != "").sum())
            row["fill_chunks"] = int((ma.manifest._paths == "").sum())

            if backend == "kerchunk-json":
                path = str(tmp_path / "mal_kc.json")
                timer.start("serialize")
                vds.vz.to_kerchunk(path, format="json")
                timer.stop()
                row["serialize"] = timer.elapsed["serialize"]
                timer.start("open")
                ds = xr.open_dataset(path, engine="kerchunk",
                                     storage_options={"skip_instance_cache": True})
                timer.stop()
                row["open"] = timer.elapsed["open"]
                timer.start("mean()")
                float(ds["v"].mean())
                timer.stop()
                row["mean()"] = timer.elapsed["mean()"]

            elif backend == "kerchunk-parquet":
                path = str(tmp_path / "mal_kc.parquet")
                timer.start("serialize")
                vds.vz.to_kerchunk(path, format="parquet")
                timer.stop()
                row["serialize"] = timer.elapsed["serialize"]
                timer.start("open")
                ds = xr.open_dataset(path, engine="kerchunk",
                                     storage_options={"skip_instance_cache": True})
                timer.stop()
                row["open"] = timer.elapsed["open"]
                timer.start("mean()")
                float(ds["v"].mean())
                timer.stop()
                row["mean()"] = timer.elapsed["mean()"]

            else:  # icechunk
                import icechunk
                from icechunk import Repository, Storage
                ic_path = str(tmp_path / "mal_ic")
                url_prefix = "https://its-live-data.s3.amazonaws.com/"
                storage = Storage.new_local_filesystem(ic_path)
                config = icechunk.RepositoryConfig.default()
                container = icechunk.VirtualChunkContainer(
                    url_prefix=url_prefix,
                    store=icechunk.http_store(),
                )
                config.set_virtual_chunk_container(container)
                repo = Repository.create(
                    storage=storage,
                    config=config,
                    authorize_virtual_chunk_access={url_prefix: None},
                )
                session = repo.writable_session("main")
                timer.start("serialize")
                vds.vz.to_icechunk(session.store)
                session.commit("bench")
                timer.stop()
                row["serialize"] = timer.elapsed["serialize"]
                timer.start("open")
                import zarr
                read_repo = Repository.open(
                    storage,
                    authorize_virtual_chunk_access={url_prefix: None},
                )
                z = zarr.open_group(read_repo.readonly_session("main").store, mode="r")
                timer.stop()
                row["open"] = timer.elapsed["open"]
                timer.start("mean()")
                float(np.nanmean(z["v"][:].astype("float32")))
                timer.stop()
                row["mean()"] = timer.elapsed["mean()"]

            results[backend] = row

        _print_comparison(
            "Malaspina glacier — 20 real ITS_LIVE granules (WRS 063018, 2025)",
            results,
            ["virtualize", "serialize", "open", "mean()",
             "real_chunks", "fill_chunks"],
        )


class TestITSLiveE2ENonAligned:
    """Non-chunk-aligned tile widths: documents the degenerate GET-per-element case."""

    NX1   = 32
    NX2   = 40   # union_nx = 72; 72 % 16 == 8 → no x-consolidation
    NY    = 48
    STEP  = 120.0
    CHUNK = 16

    @pytest.fixture
    def files(self, http_fixture_server):
        root = http_fixture_server["root"]
        base = http_fixture_server["base_url"]
        ic_base = http_fixture_server["icechunk_base_url"]
        p1 = root / "itslive_na_t1.h5"
        p2 = root / "itslive_na_t2.h5"
        _make_itslive_like(p1, nx=self.NX1, ny=self.NY,
                           x_start=0.0, y_start=self.NY * self.STEP,
                           time_val=1.0, step=self.STEP, chunk=self.CHUNK)
        _make_itslive_like(p2, nx=self.NX2, ny=self.NY,
                           x_start=self.NX1 * self.STEP,
                           y_start=self.NY * self.STEP,
                           time_val=2.0, step=self.STEP, chunk=self.CHUNK)
        return p1, p2, base, ic_base

    def _virtualize(self, p1: Path, p2: Path, base_url: str) -> xr.Dataset:
        store = HTTPStore(base_url + "/", client_options={"allow_http": True})
        registry = ObjectStoreRegistry({base_url + "/": store})
        url1 = f"{base_url}/{p1.name}"
        url2 = f"{base_url}/{p2.name}"
        return open_virtual_mfdataset(
            [url1, url2],
            registry=registry,
            parser=HDFParser(),
            combine="nested",
            concat_dim="time",
            loadable_variables=["x", "y", "time"],
            mosaic_dims=["y", "x"],
            pad="none",
        )

    def test_kerchunk_parquet_non_aligned(self, files, tmp_path, http_stats):
        """Documents that non-aligned tiles fall back to 1 GET per element."""
        import math
        p1, p2, base_url, _ic_base = files

        http_stats.reset()
        vds = self._virtualize(p1, p2, base_url)
        ma = vds["vx"].data

        # union_nx = 72; 72 % 16 = 8 ≠ 0 → consolidation skipped on x-axis.
        # Without x-consolidation, y-only merging is also blocked (y-stride is
        # non-contiguous in row-major storage).  Chunks stay at (1,1,1).
        # Real manifest entries = one per element within each tile's footprint:
        #   tile1: 1 × 48 × 32 = 1536
        #   tile2: 1 × 48 × 40 = 1920
        #   total = 3456
        expected_real   = 1 * self.NY * self.NX1 + 1 * self.NY * self.NX2  # 3456
        expected_chunks = (1, 1, 1)

        actual_chunks = ma.chunks
        actual_real   = int((ma.manifest._paths != "").sum())

        aligned_gets = 1 * math.ceil(self.NY/self.CHUNK) * (
            math.ceil(self.NX1/self.CHUNK) + math.ceil(self.NX2/self.CHUNK)
        )  # what aligned tiles would cost: 15

        print(f"\n[ITS_LIVE non-aligned] chunks={actual_chunks}, "
              f"real manifest entries={actual_real}")
        print(f"  union_nx={self.NX1+self.NX2}, {self.NX1+self.NX2} % {self.CHUNK} "
              f"= {(self.NX1+self.NX2) % self.CHUNK} → no consolidation")
        print(f"  cost: {actual_real} GETs  (aligned tiles would cost {aligned_gets})")

        assert actual_chunks == expected_chunks, (
            f"expected chunks {expected_chunks}, got {actual_chunks}"
        )
        assert actual_real == expected_real, (
            f"real manifest entries: got {actual_real}, expected {expected_real}"
        )

        path = str(tmp_path / "itslive_na.parquet")
        vds.vz.to_kerchunk(path, format="parquet")

        http_stats.reset()
        ds = xr.open_dataset(path, engine="kerchunk",
                             storage_options={"skip_instance_cache": True})
        float(ds["vx"].mean())
        compute_gets = http_stats.get_count

        assert compute_gets == expected_real, (
            f"compute GETs: got {compute_gets}, expected {expected_real}"
        )
        print(f"  compute GETs = {compute_gets}")


# ---------------------------------------------------------------------------
# Scenario: TEMPO full-week (168 granules × 7 days × 24 scans)
#
# Synthetic HDF5 files with the same *structure* as real TEMPO L2 data:
#   - variable scanline count per granule (mirrors hourly orbit geometry)
#   - fixed xtrack dimension
#   - contiguous uncompressed storage  (TEMPO does not compress L2)
#   - chunk_ny=16 scanlines  (proxy for real 40-scanline TEMPO chunks)
#
# The test generates 168 files, virtualizes them all, pads each to the
# union scanline count, concatenates along a "granule" dimension, then
# serializes to each backend and computes vza.mean().
# ---------------------------------------------------------------------------

def _make_tempo_week_files(
    root: Path,
    n_granules: int,
    nx: int,
    chunk_ny: int,
    rng: "np.random.Generator",
) -> list[tuple[Path, int]]:
    """Write *n_granules* TEMPO-like HDF5 files with variable ny.

    Returns list of (path, ny) tuples sorted by granule index.
    """
    # scanline counts: uniform random in [6·chunk_ny, 10·chunk_ny]
    ny_values = rng.integers(6 * chunk_ny, 10 * chunk_ny + 1, size=n_granules)
    files = []
    for i, ny in enumerate(ny_values):
        p = root / f"tempo_week_{i:03d}.h5"
        _make_tempo_like(p, int(ny), nx, chunk_ny)
        files.append((p, int(ny)))
    return files


@pytest.mark.slow
class TestTempoWeekE2E:
    """
    Full-week TEMPO scenario: 168 synthetic granules (7 days × 24 scans/day).

    Each granule has a variable scanline count (mirrors real hourly orbit
    geometry).  All granules share the same xtrack dimension and chunk shape.
    The pipeline pads every granule to the union scanline count before
    concatenating along the granule axis.

    Marked ``slow`` — run with ``--run-slow-tests``.
    Also needs the HTTP fixture server for GET-count tracking.
    """

    N_GRANULES = 168   # 7 days × 24 scans
    NX = 64            # proxy for TEMPO's 2048-pixel xtrack
    CHUNK_NY = 16      # proxy for TEMPO's ~40-scanline chunk

    @pytest.fixture
    def files(self, http_fixture_server):
        root = http_fixture_server["root"]
        base = http_fixture_server["base_url"]
        ic_base = http_fixture_server["icechunk_base_url"]
        rng = np.random.default_rng(42)
        granules = _make_tempo_week_files(root, self.N_GRANULES, self.NX,
                                          self.CHUNK_NY, rng)
        return granules, base, ic_base

    def _virtualize(
        self,
        granules: "list[tuple[Path, int]]",
        base_url: str,
    ) -> xr.Dataset:
        store = HTTPStore(base_url + "/", client_options={"allow_http": True})
        registry = ObjectStoreRegistry({base_url + "/": store})
        union_ny = max(ny for _, ny in granules)
        # round up to chunk boundary
        union_ny = int(np.ceil(union_ny / self.CHUNK_NY) * self.CHUNK_NY)

        arrays = []
        for p, ny in granules:
            url = f"{base_url}/{p.name}"
            vds = open_virtual_dataset(url, registry=registry,
                                       parser=HDFParser(), loadable_variables=[])
            ma = vds["vza"].data
            if ny < union_ny:
                ma = ma.pad_to_shape((union_ny, self.NX))
            arrays.append(ma)

        from virtualizarr.manifests.array_api import concatenate as vz_concat
        combined = vz_concat(arrays, axis=0)
        return xr.Dataset({"vza": xr.Variable(("scanline", "xtrack"), combined)})

    def test_comparison_table(self, files, tmp_path, http_stats):
        pytest.importorskip("icechunk")
        granules, base_url, ic_base = files
        _icechunk_skip_if_no_port80(ic_base)

        results: dict[str, dict] = {}
        stages = ["virtualize", "serialize", "open", "mean()",
                  "virtualize_gets", "compute_gets", "compute_kb"]

        for backend in ("kerchunk-parquet", "kerchunk-json", "icechunk"):
            timer = _Timer()
            row: dict = {}

            http_stats.reset()
            timer.start("virtualize")
            vds = self._virtualize(granules, base_url)
            timer.stop()
            row["virtualize"] = timer.elapsed["virtualize"]
            row["virtualize_gets"] = http_stats.get_count

            if backend in ("kerchunk-parquet", "kerchunk-json"):
                fmt = "parquet" if backend == "kerchunk-parquet" else "json"
                ext = "parquet" if fmt == "parquet" else "json"
                path = str(tmp_path / f"tempo_week.{ext}")
                timer.start("serialize")
                vds.vz.to_kerchunk(path, format=fmt)
                timer.stop()
                row["serialize"] = timer.elapsed["serialize"]

                timer.start("open")
                ds = xr.open_dataset(path, engine="kerchunk",
                                     storage_options={"skip_instance_cache": True})
                timer.stop()
                row["open"] = timer.elapsed["open"]

                http_stats.reset()
                timer.start("mean()")
                float(ds["vza"].mean())
                timer.stop()
                row["mean()"] = timer.elapsed["mean()"]

            else:  # icechunk
                ic_path = str(tmp_path / "tempo_week_ic")
                ic_store, ic_session = _icechunk_write_store(ic_path, ic_base)
                timer.start("serialize")
                vds.vz.to_icechunk(ic_store)
                ic_session.commit("tempo week bench")
                timer.stop()
                row["serialize"] = timer.elapsed["serialize"]

                timer.start("open")
                z = _icechunk_open_zarr(ic_path, ic_base)
                timer.stop()
                row["open"] = timer.elapsed["open"]

                http_stats.reset()
                timer.start("mean()")
                float(np.nanmean(z["vza"][:]))
                timer.stop()
                row["mean()"] = timer.elapsed["mean()"]

            row["compute_gets"] = http_stats.get_count
            row["compute_kb"] = http_stats.bytes_transferred / 1024
            results[backend] = row

        ma = vds["vza"].data
        n_real = int((ma.manifest._paths != "").sum())
        n_fill = int((ma.manifest._paths == "").sum())
        union_ny = ma.shape[0]

        print("\n  TEMPO full-week fixture summary")
        print(f"  granules           : {self.N_GRANULES}")
        print(f"  union scanlines    : {union_ny}")
        print(f"  xtrack pixels      : {self.NX}")
        print(f"  real chunks        : {n_real}")
        print(f"  fill chunks        : {n_fill}")

        _print_comparison(
            f"TEMPO full-week  ({self.N_GRANULES} granules, "
            f"pad_to_shape → concat)",
            results,
            stages,
        )


# ---------------------------------------------------------------------------
# Scenario: Malaspina full-tile (20 real ITS-LIVE granules, network)
#
# Uses the same 20 MALASPINA_URLS as TestMalaspinaE2E but adds HTTP GET
# counting via a proxying HTTPStore wrapper so we can report the same
# "virtualize_gets / compute_gets / compute_kb" columns as the fixture tests.
# ---------------------------------------------------------------------------

class _CountingStore:
    """Thin wrapper around an obstore HTTPStore that counts GET requests."""

    def __init__(self, store, stats: RequestStats):
        self._store = store
        self._stats = stats

    def __getattr__(self, name):
        attr = getattr(self._store, name)
        return attr

    def get_range(self, path, start, end):
        result = self._store.get_range(path, start=start, end=end)
        self._stats._count += 1
        self._stats._bytes += len(bytes(result))
        return result

    def get_ranges(self, path, starts, ends):
        results = self._store.get_ranges(path, starts=starts, ends=ends)
        self._stats._count += len(starts)
        self._stats._bytes += sum(len(bytes(r)) for r in results)
        return results


@pytest.mark.network
class TestMalaspinaWeekE2E:
    """
    Full-tile Malaspina scenario: 20 real ITS-LIVE Landsat granules from
    WRS tile 063018 (south-east Alaska), acquired Sep 2025 – Jan 2026.

    Produces the same comparison table as the TEMPO and ITS-LIVE fixture
    tests, including HTTP GET counts and bytes transferred.

    Marked ``network`` — run with ``--run-network-tests``.
    """

    _DROP = ["mapping", "img_pair_info"]
    _BASE = "https://its-live-data.s3.amazonaws.com/"

    def _virtualize(self, http_stats: RequestStats) -> xr.Dataset:
        raw_store = HTTPStore(self._BASE)
        registry = ObjectStoreRegistry({self._BASE: raw_store})
        return open_virtual_mfdataset(
            MALASPINA_URLS,
            registry=registry,
            parser=HDFParser(drop_variables=self._DROP),
            combine="nested",
            concat_dim="time",
            loadable_variables=["x", "y", "time"],
            mosaic_dims=["y", "x"],
            pad="none",
        )

    def test_comparison_table(self, tmp_path):
        pytest.importorskip("icechunk")
        results: dict[str, dict] = {}
        stages = ["virtualize", "serialize", "open", "mean()",
                  "real_chunks", "fill_chunks"]

        # We cannot intercept S3 GETs through obstore's HTTPStore without
        # monkey-patching, so GET counts are omitted for the real-network
        # case.  Timings are the primary metric.
        for backend in ("kerchunk-parquet", "kerchunk-json", "icechunk"):
            timer = _Timer()
            row: dict = {}

            timer.start("virtualize")
            vds = self._virtualize(None)
            timer.stop()
            row["virtualize"] = timer.elapsed["virtualize"]

            ma = vds["vx"].data
            row["real_chunks"] = int((ma.manifest._paths != "").sum())
            row["fill_chunks"] = int((ma.manifest._paths == "").sum())

            if backend in ("kerchunk-parquet", "kerchunk-json"):
                fmt = "parquet" if backend == "kerchunk-parquet" else "json"
                ext = "parquet" if fmt == "parquet" else "json"
                path = str(tmp_path / f"malaspina_week.{ext}")
                timer.start("serialize")
                vds.vz.to_kerchunk(path, format=fmt)
                timer.stop()
                row["serialize"] = timer.elapsed["serialize"]

                timer.start("open")
                ds = xr.open_dataset(path, engine="kerchunk",
                                     storage_options={"skip_instance_cache": True})
                timer.stop()
                row["open"] = timer.elapsed["open"]

                timer.start("mean()")
                result = float(ds["v"].mean())
                timer.stop()
                row["mean()"] = timer.elapsed["mean()"]
                assert np.isfinite(result)

            else:  # icechunk
                import icechunk
                from icechunk import Repository, Storage
                url_prefix = self._BASE
                ic_path = str(tmp_path / "malaspina_week_ic")
                storage = Storage.new_local_filesystem(ic_path)
                config = icechunk.RepositoryConfig.default()
                container = icechunk.VirtualChunkContainer(
                    url_prefix=url_prefix,
                    store=icechunk.http_store(),
                )
                config.set_virtual_chunk_container(container)
                repo = Repository.create(
                    storage=storage,
                    config=config,
                    authorize_virtual_chunk_access={url_prefix: None},
                )
                session = repo.writable_session("main")

                timer.start("serialize")
                vds.vz.to_icechunk(session.store)
                session.commit("malaspina week bench")
                timer.stop()
                row["serialize"] = timer.elapsed["serialize"]

                timer.start("open")
                import zarr
                read_repo = Repository.open(
                    storage,
                    authorize_virtual_chunk_access={url_prefix: None},
                )
                z = zarr.open_group(
                    read_repo.readonly_session("main").store, mode="r"
                )
                timer.stop()
                row["open"] = timer.elapsed["open"]

                timer.start("mean()")
                result = float(np.nanmean(z["v"][:].astype("float32")))
                timer.stop()
                row["mean()"] = timer.elapsed["mean()"]
                assert np.isfinite(result)

            results[backend] = row

        _print_comparison(
            f"Malaspina glacier — {len(MALASPINA_URLS)} real ITS-LIVE granules"
            " (WRS 063018, Sep 2025 – Jan 2026)",
            results,
            stages,
        )


# ---------------------------------------------------------------------------
# Scenario: Malaspina full-tile — cached locally, served via fixture server
#
# Downloads the 20 real ITS-LIVE granules from S3 once into the fixtures
# directory (next to the server's --serve-dir root), then replaces the S3
# URLs in every ManifestArray with local http://127.0.0.1/malaspina/<file>
# URLs so that all GET requests — virtualize *and* compute — are counted and
# tracked by the same fixture-server stats as the TEMPO tests.
#
# Markers
# -------
# ``network``  – first run needs internet to fetch granules from S3
# ``slow``     – virtualizing 20 large real HDF5 files takes ~20 s
# ---------------------------------------------------------------------------

_MALASPINA_S3_BASE = "https://its-live-data.s3.amazonaws.com/"
_MALASPINA_SUBDIR  = "malaspina"   # sub-dir inside the fixture server root


def _download_malaspina_fixtures(dest: Path) -> list[Path]:
    """
    Download MALASPINA_URLS into *dest* (skips files already present).

    Returns list of local Paths in the same order as MALASPINA_URLS.
    """
    import urllib.request as _ur
    dest.mkdir(parents=True, exist_ok=True)
    paths = []
    for url in MALASPINA_URLS:
        fname = url.split("/")[-1]
        local = dest / fname
        if not local.exists():
            print(f"  downloading {fname} …", flush=True)
            _ur.urlretrieve(url, local)
        paths.append(local)
    return paths



@pytest.mark.network
@pytest.mark.slow
class TestMalaspinaWeekLocalE2E:
    """
    Full-tile Malaspina scenario with GET-count tracking.

    Granules are downloaded from S3 once and cached in the fixture server's
    ``malaspina/`` sub-directory.  Subsequent runs are fully offline.

    All three backends (kerchunk-parquet, kerchunk-json, icechunk) are
    benchmarked with the same comparison table as TestTempoWeekE2E, including
    GET counts and bytes transferred.

    Markers: ``network`` (first run only) + ``slow``.
    """

    _DROP = ["mapping", "img_pair_info"]

    @pytest.fixture
    def local_files(self, http_fixture_server):
        root       = http_fixture_server["root"]
        base_url   = http_fixture_server["base_url"]
        ic_base    = http_fixture_server["icechunk_base_url"]
        dest       = root / _MALASPINA_SUBDIR
        paths      = _download_malaspina_fixtures(dest)
        local_urls = [f"{base_url}/{_MALASPINA_SUBDIR}/{p.name}" for p in paths]
        return local_urls, base_url, ic_base

    def _virtualize(self, local_urls: list[str], base_url: str) -> xr.Dataset:
        store    = HTTPStore(base_url + "/", client_options={"allow_http": True})
        registry = ObjectStoreRegistry({base_url + "/": store})
        return open_virtual_mfdataset(
            local_urls,
            registry=registry,
            parser=HDFParser(drop_variables=self._DROP),
            combine="nested",
            concat_dim="time",
            loadable_variables=["x", "y", "time"],
            mosaic_dims=["y", "x"],
            pad="none",
        )

    def test_comparison_table(self, local_files, tmp_path, http_stats):
        pytest.importorskip("icechunk")
        local_urls, base_url, ic_base = local_files
        _icechunk_skip_if_no_port80(ic_base)

        results: dict[str, dict] = {}
        stages = ["virtualize", "serialize", "open", "mean()",
                  "virtualize_gets", "compute_gets", "compute_kb",
                  "real_chunks", "fill_chunks"]

        for backend in ("kerchunk-parquet", "kerchunk-json", "icechunk"):
            timer = _Timer()
            row: dict = {}

            http_stats.reset()
            timer.start("virtualize")
            vds = self._virtualize(local_urls, base_url)
            timer.stop()
            row["virtualize"]      = timer.elapsed["virtualize"]
            row["virtualize_gets"] = http_stats.get_count

            ma = vds["vx"].data
            row["real_chunks"] = int((ma.manifest._paths != "").sum())
            row["fill_chunks"] = int((ma.manifest._paths == "").sum())

            if backend in ("kerchunk-parquet", "kerchunk-json"):
                fmt  = "parquet" if backend == "kerchunk-parquet" else "json"
                ext  = fmt
                path = str(tmp_path / f"malaspina_local.{ext}")

                timer.start("serialize")
                vds.vz.to_kerchunk(path, format=fmt)
                timer.stop()
                row["serialize"] = timer.elapsed["serialize"]

                timer.start("open")
                ds = xr.open_dataset(path, engine="kerchunk",
                                     storage_options={"skip_instance_cache": True})
                timer.stop()
                row["open"] = timer.elapsed["open"]

                http_stats.reset()
                timer.start("mean()")
                result = float(ds["v"].mean())
                timer.stop()
                row["mean()"] = timer.elapsed["mean()"]
                assert np.isfinite(result)

            else:  # icechunk
                ic_path = str(tmp_path / "malaspina_local_ic")
                ic_store, ic_session = _icechunk_write_store(ic_path, base_url + "/")

                timer.start("serialize")
                vds.vz.to_icechunk(ic_store)
                ic_session.commit("malaspina local bench")
                timer.stop()
                row["serialize"] = timer.elapsed["serialize"]

                timer.start("open")
                z = _icechunk_open_zarr(ic_path, base_url + "/")
                timer.stop()
                row["open"] = timer.elapsed["open"]

                http_stats.reset()
                timer.start("mean()")
                result = float(np.nanmean(z["v"][:].astype("float32")))
                timer.stop()
                row["mean()"] = timer.elapsed["mean()"]
                assert np.isfinite(result)

            row["compute_gets"] = http_stats.get_count
            row["compute_kb"]   = http_stats.bytes_transferred / 1024
            results[backend]    = row

        _print_comparison(
            f"Malaspina glacier — {len(MALASPINA_URLS)} ITS-LIVE granules"
            f" (local cache, WRS 063018)",
            results,
            stages,
        )

