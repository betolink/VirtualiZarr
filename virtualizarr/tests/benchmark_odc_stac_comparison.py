"""
Benchmark: VirtualiZarr mosaic vs rasterio manual stacking
for 20 ITS-LIVE Malaspina granules (local files).

VirtualiZarr workflow:  virtualize → serialize (kerchunk JSON) → open with xarray → mask → mean
Rasterio workflow:      open 20 files → read full bands → stack → mask → mean

Measures:
  - wall-clock time per stage
  - pixel-level value agreement

Usage:
    uv run python virtualizarr/tests/benchmark_odc_stac_comparison.py
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "malaspina"

MALASPINA_FILENAMES = [
    "LC08_L1TP_063018_20250929_20251002_02_T1_X_LC08_L1TP_063018_20251218_20251225_02_T1_G0120V02_P039.nc",
    "LC08_L1TP_063018_20250929_20251002_02_T1_X_LC08_L1GT_063018_20251116_20251202_02_T2_G0120V02_P006.nc",
    "LC08_L1TP_063018_20250929_20251002_02_T1_X_LC09_L1TP_063018_20251007_20251007_02_T1_G0120V02_P067.nc",
    "LC08_L1TP_063018_20250929_20251002_02_T1_X_LC09_L1TP_063018_20251108_20251108_02_T2_G0120V02_P022.nc",
    "LC08_L1TP_063018_20250929_20251002_02_T1_X_LC09_L1TP_063018_20251124_20251124_02_T1_G0120V02_P033.nc",
    "LC08_L1TP_063018_20250929_20251002_02_T1_X_LC09_L1TP_063018_20251210_20251210_02_T2_G0120V02_P045.nc",
    "LC08_L1TP_063018_20250929_20251002_02_T1_X_LC09_L1GT_063018_20260111_20260111_02_T2_G0120V02_P019.nc",
    "LC08_L1GT_063018_20251116_20251202_02_T2_X_LC08_L1TP_063018_20251218_20251225_02_T1_G0120V02_P012.nc",
    "LC08_L1GT_063018_20251116_20251202_02_T2_X_LC09_L1TP_063018_20251210_20251210_02_T2_G0120V02_P012.nc",
    "LC09_L1TP_063018_20251007_20251007_02_T1_X_LC08_L1GT_063018_20251116_20251202_02_T2_G0120V02_P006.nc",
    "LC09_L1TP_063018_20251007_20251007_02_T1_X_LC08_L1TP_063018_20251218_20251225_02_T1_G0120V02_P043.nc",
    "LC09_L1TP_063018_20251007_20251007_02_T1_X_LC09_L1GT_063018_20260111_20260111_02_T2_G0120V02_P017.nc",
    "LC09_L1TP_063018_20251007_20251007_02_T1_X_LC09_L1TP_063018_20251108_20251108_02_T2_G0120V02_P020.nc",
    "LC09_L1TP_063018_20251007_20251007_02_T1_X_LC09_L1TP_063018_20251124_20251124_02_T1_G0120V02_P032.nc",
    "LC09_L1TP_063018_20251007_20251007_02_T1_X_LC09_L1TP_063018_20251210_20251210_02_T2_G0120V02_P048.nc",
    "LC09_L1TP_063018_20251108_20251108_02_T2_X_LC08_L1GT_063018_20251116_20251202_02_T2_G0120V02_P010.nc",
    "LC09_L1TP_063018_20251108_20251108_02_T2_X_LC08_L1TP_063018_20251218_20251225_02_T1_G0120V02_P032.nc",
    "LC09_L1TP_063018_20251108_20251108_02_T2_X_LC09_L1TP_063018_20251124_20251124_02_T1_G0120V02_P036.nc",
    "LC09_L1TP_063018_20251108_20251108_02_T2_X_LC09_L1TP_063018_20251210_20251210_02_T2_G0120V02_P035.nc",
    "LC09_L1TP_063018_20251124_20251124_02_T1_X_LC09_L1TP_063018_20251210_20251210_02_T2_G0120V02_P052.nc",
]


def _local_paths():
    paths = [str(FIXTURE_DIR / fn) for fn in MALASPINA_FILENAMES]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        pytest.skip(f"Missing {len(missing)} fixture files. Run TestMalaspinaWeekLocalE2E first.")
    return paths


def _virtualizarr_mean():
    """Full VirtualiZarr workflow: virtualize → kerchunk JSON → open → mask → mean."""
    from obspec_utils.registry import ObjectStoreRegistry
    from obstore.store import LocalStore

    from virtualizarr.parsers.hdf import HDFParser
    from virtualizarr.xarray import open_virtual_mfdataset

    paths = _local_paths()
    registry = ObjectStoreRegistry({"file://": LocalStore()})

    # Stage 1: virtualize
    t0 = time.perf_counter()
    vds = open_virtual_mfdataset(
        paths,
        registry=registry,
        combine="nested",
        concat_dim="time",
        loadable_variables=["x", "y", "time"],
        mosaic_dims=["y", "x"],
        pad="none",
        parser=HDFParser(drop_variables=["mapping", "img_pair_info"]),
    )
    t_virtualize = time.perf_counter() - t0

    # Stage 2: serialize to kerchunk JSON
    with tempfile.TemporaryDirectory() as tmpdir:
        ref_path = os.path.join(tmpdir, "refs.json")
        t0 = time.perf_counter()
        vds.vz.to_kerchunk(ref_path, format="json")
        t_serialize = time.perf_counter() - t0

        # Stage 3: open with xarray (kerchunk engine)
        t0 = time.perf_counter()
        ds = xr.open_dataset(ref_path, engine="kerchunk")
        t_open = time.perf_counter() - t0

        # Stage 4: mask fill values, then mean
        # xarray's kerchunk/zarr backend stores _FillValue in encoding but does NOT auto-mask
        fill_value = ds["v"].encoding.get("_FillValue", -32767)
        v_masked = ds["v"].where(ds["v"] != fill_value)

        t0 = time.perf_counter()
        result = v_masked.mean("time", skipna=True).compute()
        t_compute = time.perf_counter() - t0

    return {
        "virtualize_ms": round(t_virtualize * 1000),
        "serialize_ms": round(t_serialize * 1000),
        "open_ms": round(t_open * 1000),
        "compute_ms": round(t_compute * 1000),
        "total_ms": round((t_virtualize + t_serialize + t_open + t_compute) * 1000),
        "result": result,
        "shape": ds["v"].shape,
    }


def _rasterio_stack_mean():
    """Rasterio manual path: open each file, read band, place in union grid, stack, mean."""
    import rasterio

    paths = _local_paths()

    # First pass: collect metadata
    datasets = []
    for p in paths:
        subdataset = f"netcdf:{p}:v"
        with rasterio.open(subdataset) as ds:
            datasets.append({
                "path": p,
                "shape": ds.shape,
                "transform": ds.transform,
                "crs": ds.crs,
                "nodata": ds.nodata,
            })

    # Compute union grid from transforms
    # For EPSG:3413 with 120m pixels, transform is:
    #   | 120, 0, x_min |
    #   | 0, -120, y_max |
    x_mins = []
    x_maxs = []
    y_mins = []
    y_maxs = []
    for d in datasets:
        transform = d["transform"]
        height, width = d["shape"]
        x_min = transform.c
        x_max = transform.c + width * transform.a
        y_max = transform.f
        y_min = transform.f + height * transform.e  # e is negative
        x_mins.append(x_min)
        x_maxs.append(x_max)
        y_mins.append(y_min)
        y_maxs.append(y_max)

    union_x_min = min(x_mins)
    union_x_max = max(x_maxs)
    union_y_min = min(y_mins)
    union_y_max = max(y_maxs)

    # Resolution from first dataset
    res_x = datasets[0]["transform"].a
    res_y = abs(datasets[0]["transform"].e)

    union_width = int(round((union_x_max - union_x_min) / res_x))
    union_height = int(round((union_y_max - union_y_min) / res_y))

    t0 = time.perf_counter()
    arrays = []
    for d in datasets:
        subdataset = f"netcdf:{d['path']}:v"
        with rasterio.open(subdataset) as ds:
            data = ds.read(1)
            nodata = d["nodata"]
            if nodata is not None:
                data = np.where(data == nodata, np.nan, data).astype(np.float32)
            else:
                data = data.astype(np.float32)

        # Place in union grid
        transform = d["transform"]
        height, width = d["shape"]
        x_offset = int(round((transform.c - union_x_min) / res_x))
        y_offset = int(round((union_y_max - transform.f) / res_y))

        union = np.full((union_height, union_width), np.nan, dtype=np.float32)
        union[y_offset:y_offset + height, x_offset:x_offset + width] = data
        arrays.append(union)
    t_read = time.perf_counter() - t0

    t0 = time.perf_counter()
    stacked = np.stack(arrays, axis=0)
    da = xr.DataArray(stacked, dims=["time", "y", "x"], name="v")
    result = da.mean("time", skipna=True).compute()
    t_compute = time.perf_counter() - t0

    return {
        "read_ms": round(t_read * 1000),
        "compute_ms": round(t_compute * 1000),
        "total_ms": round((t_read + t_compute) * 1000),
        "result": result,
        "shape": da.shape,
    }


def main():
    print("=" * 60)
    print("VirtualiZarr mosaic  vs  Rasterio manual stacking")
    print("20 ITS-LIVE Malaspina granules (local)")
    print("=" * 60)

    print("\n[VirtualiZarr] Running...")
    vz = _virtualizarr_mean()
    print(f"  virtualize : {vz['virtualize_ms']} ms")
    print(f"  serialize  : {vz['serialize_ms']} ms")
    print(f"  open       : {vz['open_ms']} ms")
    print(f"  compute    : {vz['compute_ms']} ms")
    print(f"  total      : {vz['total_ms']} ms")
    print(f"  shape      : {vz['shape']}")

    print("\n[Rasterio manual] Running...")
    rio = _rasterio_stack_mean()
    print(f"  read files : {rio['read_ms']} ms")
    print(f"  compute    : {rio['compute_ms']} ms")
    print(f"  total      : {rio['total_ms']} ms")
    print(f"  shape      : {rio['shape']}")

    print("\n[Pixel comparison]")
    vz_result = vz["result"]
    rio_result = rio["result"]

    min_y = min(vz_result.shape[0], rio_result.shape[0])
    min_x = min(vz_result.shape[1], rio_result.shape[1])

    vz_slice = vz_result.values[:min_y, :min_x]
    rio_slice = rio_result.values[:min_y, :min_x]

    both_nan = np.isnan(vz_slice) & np.isnan(rio_slice)
    both_valid = np.isfinite(vz_slice) & np.isfinite(rio_slice)
    vz_nan_rio_valid = np.isnan(vz_slice) & np.isfinite(rio_slice)
    vz_valid_rio_nan = np.isfinite(vz_slice) & np.isnan(rio_slice)

    print(f"  overlapping region : {min_y} × {min_x}")
    print(f"  both NaN           : {both_nan.sum()} pixels")
    print(f"  both valid         : {both_valid.sum()} pixels")
    print(f"  VZ NaN / RIO valid : {vz_nan_rio_valid.sum()} pixels")
    print(f"  VZ valid / RIO NaN : {vz_valid_rio_nan.sum()} pixels")

    if both_valid.sum() > 0:
        diff = np.abs(vz_slice[both_valid] - rio_slice[both_valid])
        print(f"  max abs diff       : {diff.max():.6f}")
        print(f"  mean abs diff      : {diff.mean():.6f}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
