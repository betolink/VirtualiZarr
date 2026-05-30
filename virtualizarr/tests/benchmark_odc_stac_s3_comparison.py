"""
Benchmark: VirtualiZarr mosaic vs odc-stac stacking
for 20 ITS-LIVE Malaspina granules from real S3 + STAC catalog.

VirtualiZarr:  STAC item IDs → S3 URLs → virtualize → kerchunk JSON → open → mask → mean
odc-stac:     STAC items → patch_url to /vsicurl/ → stack → mean (auto nodata mask)

NOTE: odc-stac requires scheduler='synchronous' for NetCDF over HTTP because
HDF5 is not thread-safe. Each file is downloaded whole (~6 MB) to a temp location
by GDAL's /vsicurl/ driver during rasterio.open().

Usage:
    uv run python virtualizarr/tests/benchmark_odc_stac_s3_comparison.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request

import numpy as np
import xarray as xr

MALASPINA_ITEM_IDS = [
    "LC08_L1TP_063018_20250929_20251002_02_T1_X_LC08_L1TP_063018_20251218_20251225_02_T1_G0120V02_P039",
    "LC08_L1TP_063018_20250929_20251002_02_T1_X_LC08_L1GT_063018_20251116_20251202_02_T2_G0120V02_P006",
    "LC08_L1TP_063018_20250929_20251002_02_T1_X_LC09_L1TP_063018_20251007_20251007_02_T1_G0120V02_P067",
    "LC08_L1TP_063018_20250929_20251002_02_T1_X_LC09_L1TP_063018_20251108_20251108_02_T2_G0120V02_P022",
    "LC08_L1TP_063018_20250929_20251002_02_T1_X_LC09_L1TP_063018_20251124_20251124_02_T1_G0120V02_P033",
    "LC08_L1TP_063018_20250929_20251002_02_T1_X_LC09_L1TP_063018_20251210_20251210_02_T2_G0120V02_P045",
    "LC08_L1TP_063018_20250929_20251002_02_T1_X_LC09_L1GT_063018_20260111_20260111_02_T2_G0120V02_P019",
    "LC08_L1GT_063018_20251116_20251202_02_T2_X_LC08_L1TP_063018_20251218_20251225_02_T1_G0120V02_P012",
    "LC08_L1GT_063018_20251116_20251202_02_T2_X_LC09_L1TP_063018_20251210_20251210_02_T2_G0120V02_P012",
    "LC09_L1TP_063018_20251007_20251007_02_T1_X_LC08_L1GT_063018_20251116_20251202_02_T2_G0120V02_P006",
    "LC09_L1TP_063018_20251007_20251007_02_T1_X_LC08_L1TP_063018_20251218_20251225_02_T1_G0120V02_P043",
    "LC09_L1TP_063018_20251007_20251007_02_T1_X_LC09_L1GT_063018_20260111_20260111_02_T2_G0120V02_P017",
    "LC09_L1TP_063018_20251007_20251007_02_T1_X_LC09_L1TP_063018_20251108_20251108_02_T2_G0120V02_P020",
    "LC09_L1TP_063018_20251007_20251007_02_T1_X_LC09_L1TP_063018_20251124_20251124_02_T1_G0120V02_P032",
    "LC09_L1TP_063018_20251007_20251007_02_T1_X_LC09_L1TP_063018_20251210_20251210_02_T2_G0120V02_P048",
    "LC09_L1TP_063018_20251108_20251108_02_T2_X_LC08_L1GT_063018_20251116_20251202_02_T2_G0120V02_P010",
    "LC09_L1TP_063018_20251108_20251108_02_T2_X_LC08_L1TP_063018_20251218_20251225_02_T1_G0120V02_P032",
    "LC09_L1TP_063018_20251108_20251108_02_T2_X_LC09_L1TP_063018_20251124_20251124_02_T1_G0120V02_P036",
    "LC09_L1TP_063018_20251108_20251108_02_T2_X_LC09_L1TP_063018_20251210_20251210_02_T2_G0120V02_P035",
    "LC09_L1TP_063018_20251124_20251124_02_T1_X_LC09_L1TP_063018_20251210_20251210_02_T2_G0120V02_P052",
]


def _fetch_stac_items():
    import pystac

    base = "https://stac.itslive.cloud/collections/itslive-granules/items"
    items = []
    for item_id in MALASPINA_ITEM_IDS:
        url = f"{base}/{item_id}"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            items.append(pystac.Item.from_dict(json.loads(resp.read())))
    return items


def _s3_urls_from_items(items):
    return [item.assets["data"].href for item in items]


def _virtualizarr_mean(urls):
    from obspec_utils.registry import ObjectStoreRegistry
    from obstore.store import HTTPStore

    from virtualizarr.parsers.hdf import HDFParser
    from virtualizarr.xarray import open_virtual_mfdataset

    base = "https://its-live-data.s3.amazonaws.com/"
    registry = ObjectStoreRegistry({base: HTTPStore(base, retry_config={"max_retries": 3})})

    t0 = time.perf_counter()
    vds = open_virtual_mfdataset(
        urls,
        registry=registry,
        combine="nested",
        concat_dim="time",
        loadable_variables=["x", "y", "time"],
        mosaic_dims=["y", "x"],
        pad="none",
        parser=HDFParser(drop_variables=["mapping", "img_pair_info"]),
    )
    t_virtualize = time.perf_counter() - t0

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        ref_path = os.path.join(tmpdir, "refs.json")
        t0 = time.perf_counter()
        vds.vz.to_kerchunk(ref_path, format="json")
        t_serialize = time.perf_counter() - t0

        t0 = time.perf_counter()
        ds = xr.open_dataset(ref_path, engine="kerchunk")
        t_open = time.perf_counter() - t0

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


def _odc_stac_mean(items):
    import odc.stac

    os.environ["CPL_DEBUG"] = "OFF"

    def patch(url):
        if url.endswith(".nc"):
            return f"netcdf:/vsicurl/{url}:v"
        return url

    t0 = time.perf_counter()
    xx = odc.stac.load(
        items,
        bands=["data"],
        chunks={"y": 512, "x": 512},
        resolution=120,
        crs="EPSG:3413",
        patch_url=patch,
        groupby="time",
        fail_on_error=False,
    )
    # HDF5 is not thread-safe but is process-safe.
    # Use Dask's multiprocessing scheduler to read files in parallel.
    result = xx.data.mean("time", skipna=True).compute(scheduler="processes", num_workers=10)
    t_total = time.perf_counter() - t0

    return {
        "total_ms": round(t_total * 1000),
        "result": result,
        "shape": xx.data.shape,
    }


def main():
    print("=" * 60)
    print("VirtualiZarr mosaic  vs  odc-stac stacking")
    print("20 ITS-LIVE Malaspina granules (real S3)")
    print("=" * 60)

    print("\n[Fetching 20 STAC items from stac.itslive.cloud...]")
    t0 = time.perf_counter()
    items = _fetch_stac_items()
    urls = _s3_urls_from_items(items)
    t_fetch = time.perf_counter() - t0
    print(f"  Done in {round(t_fetch * 1000)} ms")

    print("\n[VirtualiZarr] Running...")
    vz = _virtualizarr_mean(urls)
    print(f"  virtualize : {vz['virtualize_ms']} ms")
    print(f"  serialize  : {vz['serialize_ms']} ms")
    print(f"  open       : {vz['open_ms']} ms")
    print(f"  compute    : {vz['compute_ms']} ms")
    print(f"  total      : {vz['total_ms']} ms")
    print(f"  shape      : {vz['shape']}")

    print("\n[odc-stac] Running...")
    print("           resolution=120, crs='EPSG:3413' (auto-computed bounds from items)")
    print("           scheduler='processes' (10 workers) because HDF5 is process-safe.")
    print("           Each file is ~6 MB downloaded whole by GDAL /vsicurl/.")
    sys.stdout.flush()
    odc = _odc_stac_mean(items)
    print(f"  total      : {odc['total_ms']} ms")
    print(f"  shape      : {odc['shape']}")

    print("\n[Pixel comparison]")
    vz_result = vz["result"]
    odc_result = odc["result"]

    # odc-stac snaps to a slightly different grid; align overlapping region
    min_y = min(vz_result.shape[0], odc_result.shape[0])
    min_x = min(vz_result.shape[1], odc_result.shape[1])

    vz_slice = vz_result.values[:min_y, :min_x]
    odc_slice = odc_result.values[:min_y, :min_x]

    both_nan = np.isnan(vz_slice) & np.isnan(odc_slice)
    both_valid = np.isfinite(vz_slice) & np.isfinite(odc_slice)
    vz_nan_odc_valid = np.isnan(vz_slice) & np.isfinite(odc_slice)
    vz_valid_odc_nan = np.isfinite(vz_slice) & np.isnan(odc_slice)

    print(f"  overlapping region : {min_y} × {min_x}")
    print(f"  both NaN           : {both_nan.sum()} pixels")
    print(f"  both valid         : {both_valid.sum()} pixels")
    print(f"  VZ NaN / ODC valid : {vz_nan_odc_valid.sum()} pixels")
    print(f"  VZ valid / ODC NaN : {vz_valid_odc_nan.sum()} pixels")

    if both_valid.sum() > 0:
        diff = np.abs(vz_slice[both_valid] - odc_slice[both_valid])
        print(f"  max abs diff       : {diff.max():.6f}")
        print(f"  mean abs diff      : {diff.mean():.6f}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
