"""
Performance benchmark: request counts and wall-time for .mean() on virtual arrays.

Tests two scenarios:
  1. TEMPO-style  – pad_to_shape (contiguous HDF5, uncompressed, ragged along scanline)
  2. ITS_LIVE-style – place_in_grid / mosaic_dims (tiles at different spatial offsets)

For each scenario we report:
  - number of real range-GET requests issued  (one per non-missing chunk)
  - number of missing-chunk fill returns       (zero I/O; fill_value substituted locally)
  - wall-time for .mean() across all data

Chunks are read from local disk via LocalStore so timing reflects the full
codec/decompression pipeline without network latency.
"""
from __future__ import annotations

import math
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pytest
import zarr
from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import LocalStore

from virtualizarr.manifests import ManifestStore
from virtualizarr.manifests.array_api import concatenate
from virtualizarr.manifests.group import ManifestGroup
from virtualizarr.parsers.hdf import HDFParser
from virtualizarr.xarray import open_virtual_dataset, open_virtual_mfdataset

# ---------------------------------------------------------------------------
# Request counter
# ---------------------------------------------------------------------------

@contextmanager
def count_requests():
    """
    Patch ManifestStore.get to tally:
      real  – chunk key that resolves to a real byte range  → 1 range-GET
      fill  – chunk key with MISSING_CHUNK_PATH ("")        → 0 I/O, fill_value used
    Metadata keys (zarr.json / .zarray / etc.) are excluded.
    """
    counts = {"real": 0, "fill": 0}
    original_get = ManifestStore.get

    def _is_chunk_key(key: str) -> bool:
        return not any(key.endswith(s) for s in (
            "zarr.json", ".zattrs", ".zarray", ".zgroup", ".zmetadata"
        ))

    async def patched_get(self, key, prototype, byte_range=None):
        result = await original_get(self, key, prototype, byte_range)
        if _is_chunk_key(key):
            if result is None:
                counts["fill"] += 1
            else:
                counts["real"] += 1
        return result

    with patch.object(ManifestStore, "get", patched_get):
        yield counts


# ---------------------------------------------------------------------------
# HDF5 fixture factories
# ---------------------------------------------------------------------------

def _make_tempo_like(path: Path, ny: int, nx: int, chunk_ny: int) -> None:
    """Contiguous uncompressed HDF5, TEMPO scan layout."""
    rng = np.random.default_rng(0)
    data = rng.random((ny, nx)).astype("float32")
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
                       step: float = 120.0, chunk: int = 16) -> None:
    """HDF5 with dimension scales, ITS_LIVE tile layout."""
    rng = np.random.default_rng(42)
    x = np.arange(x_start, x_start + nx * step, step, dtype="float64")
    y = np.arange(y_start, y_start - ny * step, -step, dtype="float64")
    vx = rng.integers(-100, 100, size=(1, ny, nx), dtype="int16")
    with h5py.File(path, "w") as f:
        f.create_dataset("time", data=np.array([0.0]))
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
# Scenario 1: TEMPO-style  (pad_to_shape, ragged scanline dim)
# ---------------------------------------------------------------------------

class TestTempoPerformance:
    """
    Two TEMPO-like granules (ny1=96, ny2=128 scanlines, nx=64 cross-track).
    Granule 1 is padded to ny2 along scanline before concat.

    Chunk grid per granule (chunk_ny=32, nx=64 → one chunk wide):
      g1: 3 real + 1 fill  (96/32=3 real rows; padded row = 1 fill)
      g2: 4 real
    After concat along scanline axis:
      total = 7 real + 1 fill
    """

    NY1     = 96
    NY2     = 128
    NX      = 64
    CHUNK_NY = 32

    @pytest.fixture
    def store_and_array(self, tmp_path):
        p1 = tmp_path / "g1.h5"
        p2 = tmp_path / "g2.h5"
        _make_tempo_like(p1, self.NY1, self.NX, self.CHUNK_NY)
        _make_tempo_like(p2, self.NY2, self.NX, self.CHUNK_NY)

        registry = ObjectStoreRegistry({"file://": LocalStore()})
        vds1 = open_virtual_dataset(f"file://{p1}", registry=registry,
                                    parser=HDFParser(), loadable_variables=[])
        vds2 = open_virtual_dataset(f"file://{p2}", registry=registry,
                                    parser=HDFParser(), loadable_variables=[])

        # pad g1 scanline to match g2, then concatenate manifests
        ma1 = vds1["vza"].data.pad_to_shape((self.NY2, self.NX))
        ma2 = vds2["vza"].data
        ma_concat = concatenate([ma1, ma2], axis=0)

        group = ManifestGroup(arrays={"vza": ma_concat}, groups={})
        store = ManifestStore(group, registry=registry)
        return store, ma_concat

    def test_request_counts(self, store_and_array):
        store, ma = store_and_array

        expected_real = (self.NY1 // self.CHUNK_NY) + (self.NY2 // self.CHUNK_NY)  # 3+4=7
        expected_fill = (self.NY2 // self.CHUNK_NY) - (self.NY1 // self.CHUNK_NY)  # 4-3=1

        # verify manifest directly
        actual_real = int((ma.manifest._paths != "").sum())
        actual_fill = int((ma.manifest._paths == "").sum())
        assert actual_real == expected_real, f"manifest real={actual_real}, expected {expected_real}"
        assert actual_fill == expected_fill, f"manifest fill={actual_fill}, expected {expected_fill}"

        z = zarr.open(store, mode="r")
        with count_requests() as counts:
            result = float(np.nanmean(z["vza"][:]))

        print(f"\n[TEMPO] real requests={counts['real']}, fill returns={counts['fill']}, "
              f"mean={result:.4f}")

        assert counts["real"] == expected_real
        assert counts["fill"] == expected_fill

    def test_mean_wall_time(self, store_and_array, benchmark):
        store, _ = store_and_array
        z = zarr.open(store, mode="r")
        result = benchmark(lambda: float(np.nanmean(z["vza"][:])))
        assert result is not None


# ---------------------------------------------------------------------------
# Scenario 2: ITS_LIVE-style mosaic (place_in_grid, two spatial tiles)
# ---------------------------------------------------------------------------

class TestITSLiveMosaicPerformance:
    """
    Two ITS_LIVE-like tiles, non-overlapping in x, same y extent.
      tile 1: nx=32, ny=48, chunk=16  → chunk grid (1, 3, 2) = 6 real chunks
      tile 2: nx=48, ny=48, chunk=16  → chunk grid (1, 3, 3) = 9 real chunks
    Union grid shape: (2, 48, 80).
      ny=48: 48 % 16 == 0  — y axis consolidates back to chunk_size=16
      nx=80: 80 % 16 == 0  — x axis consolidates back to chunk_size=16
      Resulting chunks: (1, 16, 16), same as originals.
      After consolidation:
        tile1 real entries: 1 × ceil(48/16) × ceil(32/16) = 1×3×2 = 6
        tile2 real entries: 1 × ceil(48/16) × ceil(48/16) = 1×3×3 = 9
        total real = 15,  remaining cells in union grid are missing (fill).
    """

    NX1   = 32
    NX2   = 48
    NY    = 48
    STEP  = 120.0
    CHUNK = 16

    @pytest.fixture
    def store_and_array(self, tmp_path):
        p1 = tmp_path / "t1.h5"
        p2 = tmp_path / "t2.h5"
        _make_itslive_like(p1, nx=self.NX1, ny=self.NY,
                           x_start=0.0, y_start=self.NY * self.STEP,
                           step=self.STEP, chunk=self.CHUNK)
        _make_itslive_like(p2, nx=self.NX2, ny=self.NY,
                           x_start=self.NX1 * self.STEP, y_start=self.NY * self.STEP,
                           step=self.STEP, chunk=self.CHUNK)

        registry = ObjectStoreRegistry({"file://": LocalStore()})
        vds = open_virtual_mfdataset(
            [f"file://{p1}", f"file://{p2}"],
            registry=registry,
            parser=HDFParser(),
            combine="nested",
            concat_dim="time",
            loadable_variables=["x", "y", "time"],
            mosaic_dims=["y", "x"],
            pad="none",
        )
        ma = vds["vx"].data
        group = ManifestGroup(arrays={"vx": ma}, groups={})
        store = ManifestStore(group, registry=registry)
        return store, ma

    def test_request_counts(self, store_and_array):
        store, ma = store_and_array

        # Both axes consolidated back to chunk_size=16 (union dims both % 16 == 0).
        # tile1: (1, 3, 2) = 6 real chunks; tile2: (1, 3, 3) = 9 real chunks
        union_nx = self.NX1 + self.NX2           # 80
        ny_chunks = math.ceil(self.NY / self.CHUNK)   # 3
        nx1_chunks = math.ceil(self.NX1 / self.CHUNK)  # 2
        nx2_chunks = math.ceil(self.NX2 / self.CHUNK)  # 3
        expected_real = (
            1 * ny_chunks * nx1_chunks    # tile1: 6
            + 1 * ny_chunks * nx2_chunks  # tile2: 9
        )  # = 15
        union_nx_chunks = math.ceil(union_nx / self.CHUNK)  # 5
        union_chunk_grid_cells = 2 * ny_chunks * union_nx_chunks
        expected_fill = union_chunk_grid_cells - expected_real

        actual_chunks = ma.chunks
        assert actual_chunks == (1, self.CHUNK, self.CHUNK), (
            f"expected chunks (1,{self.CHUNK},{self.CHUNK}), got {actual_chunks}"
        )
        actual_real = int((ma.manifest._paths != "").sum())
        actual_fill = int((ma.manifest._paths == "").sum())
        assert actual_real == expected_real, f"manifest real={actual_real}, expected {expected_real}"
        assert actual_fill == expected_fill, f"manifest fill={actual_fill}, expected {expected_fill}"

        z = zarr.open(store, mode="r")
        with count_requests() as counts:
            result = float(np.nanmean(z["vx"][:].astype("float32")))

        print(f"\n[ITS_LIVE] real requests={counts['real']}, fill returns={counts['fill']}, "
              f"mean={result:.4f}")

        assert counts["real"] == expected_real
        assert counts["fill"] == expected_fill

    def test_mean_wall_time(self, store_and_array, benchmark):
        store, _ = store_and_array
        z = zarr.open(store, mode="r")
        result = benchmark(lambda: float(np.nanmean(z["vx"][:].astype("float32"))))
        assert result is not None


# ---------------------------------------------------------------------------
# Summary: print a comparison table
# ---------------------------------------------------------------------------

class TestRequestSummary:
    """Print a side-by-side request-count comparison for both scenarios."""

    def test_print_summary(self, tmp_path):
        registry = ObjectStoreRegistry({"file://": LocalStore()})

        # --- TEMPO ---
        ny1, ny2, nx, chunk_ny = 96, 128, 64, 32
        p1, p2 = tmp_path / "g1.h5", tmp_path / "g2.h5"
        _make_tempo_like(p1, ny1, nx, chunk_ny)
        _make_tempo_like(p2, ny2, nx, chunk_ny)
        vds1 = open_virtual_dataset(f"file://{p1}", registry=registry,
                                    parser=HDFParser(), loadable_variables=[])
        vds2 = open_virtual_dataset(f"file://{p2}", registry=registry,
                                    parser=HDFParser(), loadable_variables=[])
        ma1 = vds1["vza"].data.pad_to_shape((ny2, nx))
        ma2 = vds2["vza"].data
        ma_tempo = concatenate([ma1, ma2], axis=0)
        tempo_real = int((ma_tempo.manifest._paths != "").sum())
        tempo_fill = int((ma_tempo.manifest._paths == "").sum())
        tempo_total = tempo_real + tempo_fill
        g_tempo = ManifestGroup(arrays={"vza": ma_tempo}, groups={})
        s_tempo = ManifestStore(g_tempo, registry=registry)
        z_tempo = zarr.open(s_tempo, mode="r")
        t0 = time.perf_counter()
        float(np.nanmean(z_tempo["vza"][:]))
        tempo_ms = (time.perf_counter() - t0) * 1000

        # --- ITS_LIVE ---
        nx1, nx2, ny, step, chunk = 32, 48, 48, 120.0, 16
        pt1, pt2 = tmp_path / "t1.h5", tmp_path / "t2.h5"
        _make_itslive_like(pt1, nx=nx1, ny=ny, x_start=0.0, y_start=ny*step, step=step, chunk=chunk)
        _make_itslive_like(pt2, nx=nx2, ny=ny, x_start=nx1*step, y_start=ny*step, step=step, chunk=chunk)
        vds_ils = open_virtual_mfdataset(
            [f"file://{pt1}", f"file://{pt2}"],
            registry=registry, parser=HDFParser(),
            combine="nested", concat_dim="time",
            loadable_variables=["x", "y", "time"],
            mosaic_dims=["y", "x"], pad="none",
        )
        ma_ils = vds_ils["vx"].data
        ils_real = int((ma_ils.manifest._paths != "").sum())
        ils_fill = int((ma_ils.manifest._paths == "").sum())
        ils_total = ils_real + ils_fill
        g_ils = ManifestGroup(arrays={"vx": ma_ils}, groups={})
        s_ils = ManifestStore(g_ils, registry=registry)
        z_ils = zarr.open(s_ils, mode="r")
        t0 = time.perf_counter()
        float(np.nanmean(z_ils["vx"][:].astype("float32")))
        ils_ms = (time.perf_counter() - t0) * 1000

        print("\n")
        print("=" * 65)
        print(f"{'Scenario':<20} {'real GETs':>10} {'fill (free)':>12} {'total cells':>12} {'mean() ms':>10}")
        print("-" * 65)
        print(f"{'TEMPO (pad)':<20} {tempo_real:>10} {tempo_fill:>12} {tempo_total:>12} {tempo_ms:>9.1f}")
        print(f"{'ITS_LIVE (mosaic)':<20} {ils_real:>10} {ils_fill:>12} {ils_total:>12} {ils_ms:>9.1f}")
        print("=" * 65)
        print("\nNote: ITS_LIVE uses consolidate_chunks → 1 GET per original chunk.")
        print("TEMPO keeps original chunk shape → 1 GET per chunk.")
        print("In both cases fill cells cost zero I/O.")
