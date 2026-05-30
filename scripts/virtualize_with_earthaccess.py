#!/usr/bin/env python3
"""
Virtualize NASA Earthdata collections using earthaccess + VirtualiZarr.

This script demonstrates the earthaccess → VirtualiZarr → kerchunk/icechunk
pipeline for real NASA data.  It uses ``earthaccess.virtualize()`` (with the
new ``mosaic_dims``, ``pad``, and ``loadable_variables`` parameters) to build
virtual datasets without downloading science data.

Supported workflows
-------------------
* **TEMPO** (ragged scanlines) -- ``pad="auto"``
* **ITS-LIVE** (spatial tiles)  -- ``mosaic_dims=["y","x"]``
* **Any single-group NetCDF/HDF5 collection** -- basic concatenation

Output formats
--------------
* ``kerchunk_json``  -- single JSON reference file
* ``kerchunk_parquet`` -- columnar reference store (recommended for scale)
* ``icechunk`` -- cloud-optimised virtual chunked array store

Usage examples
--------------

# TEMPO NO2 (ragged scanlines) → kerchunk parquet
    uv run python scripts/virtualize_with_earthaccess.py \
        --collection TEMPO_NO2_L2 \
        --temporal 2024-03-28 2024-03-29 \
        --version V03 \
        --group product \
        --pad auto \
        --concat-dim time \
        --loadable-variables time latitude longitude \
        --output-format kerchunk_parquet \
        --output tempo_product.parquet

# ITS-LIVE velocity mosaic (20 granules, Malaspina region)
    uv run python scripts/virtualize_with_earthaccess.py \
        --collection ITS_LIVE \
        --bbox -147.5 59.0 -143.0 61.5 \
        --concat-dim time \
        --mosaic-dims y x \
        --loadable-variables y x time \
        --pad none \
        --output-format kerchunk_parquet \
        --output itslive_mosaic.parquet

# Save to icechunk instead
    ... --output-format icechunk --output itslive_mosaic.icechunk

Authentication
--------------
Run ``earthaccess.login()`` interactively once, or set environment variables
``EARTHDATA_USERNAME`` and ``EARTHDATA_PASSWORD``.  The script calls
``earthaccess.login(persist=True)`` automatically.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any

import earthaccess
import icechunk
import xarray as xr


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Virtualize NASA Earthdata collections with earthaccess + VirtualiZarr",
    )
    p.add_argument("--collection", required=True, help="NASA Earthdata short_name")
    p.add_argument(
        "--version",
        help="Product version to filter by, e.g. V03",
    )
    p.add_argument(
        "--temporal",
        nargs=2,
        metavar=("START", "END"),
        help="Temporal bounds, e.g. 2024-03-28 2024-03-29",
    )
    p.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("W", "S", "E", "N"),
        help="Bounding box (west, south, east, north)",
    )
    p.add_argument("--group", default="/", help="HDF5/NetCDF4 group path")
    p.add_argument("--concat-dim", required=True, help="Dimension to concatenate on")
    p.add_argument(
        "--mosaic-dims",
        nargs="+",
        help="Spatial dimensions to mosaic, e.g. y x",
    )
    p.add_argument(
        "--pad",
        choices=["auto", "none"],
        default=None,
        help="Padding strategy for ragged arrays",
    )
    p.add_argument(
        "--loadable-variables",
        nargs="+",
        default=None,
        help="Coordinate variables to materialise (required for mosaic)",
    )
    p.add_argument(
        "--output-format",
        choices=["kerchunk_json", "kerchunk_parquet", "icechunk"],
        required=True,
    )
    p.add_argument("--output", required=True, help="Output path")
    p.add_argument(
        "--max-granules",
        type=int,
        default=None,
        help="Limit number of granules (for quick tests)",
    )
    p.add_argument(
        "--access",
        choices=["indirect", "direct"],
        default="indirect",
        help="Cloud access mode (indirect=HTTPS, direct=S3)",
    )
    p.add_argument(
        "--reference-format",
        choices=["json", "parquet"],
        default="parquet",
        help="Kerchunk reference format (when output-format is kerchunk_*)",
    )
    return p.parse_args()


def _auth() -> bool:
    """Authenticate with NASA Earthdata (strategy-agnostic)."""
    try:
        # earthaccess tries: environment → netrc → interactive
        earthaccess.login()
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"Authentication failed: {exc}", file=sys.stderr)
        return False


def _search(args: argparse.Namespace) -> list:
    """Search NASA Earthdata catalog and validate results."""
    import re

    kwargs: dict[str, Any] = {"short_name": args.collection}
    if args.temporal:
        kwargs["temporal"] = tuple(args.temporal)
    if args.bbox:
        kwargs["bounding_box"] = tuple(args.bbox)

    print(f"Searching {args.collection!r} …")
    granules = earthaccess.search_data(**kwargs)
    if not granules:
        raise RuntimeError("No granules found for the given query.")

    # --- detect product versions from URLs ---
    versions: dict[str, int] = {}
    for g in granules:
        url = g.data_links(access="indirect")[0]
        m = re.search(r"_V(\d+)(?:_|\b)", url)
        if m:
            v = f"V{m.group(1)}"
            versions[v] = versions.get(v, 0) + 1

    if len(versions) > 1:
        if args.version:
            before = len(granules)
            granules = [g for g in granules if args.version in g.data_links(access="indirect")[0]]
            print(f"  Filtered to version {args.version}: {before} → {len(granules)} granule(s)")
        else:
            print(
                f"WARNING: mixed product versions found: {versions}. "
                "Use --version to pick one.",
                file=sys.stderr,
            )
    else:
        print(f"  Product version: {list(versions.keys())[0]}")

    if args.max_granules:
        granules = granules[: args.max_granules]
    print(f"Found {len(granules)} granule(s)")
    return granules


def _virtualize(args: argparse.Namespace, granules: list) -> xr.Dataset:
    """Create a virtual Dataset via earthaccess.virtualize()."""
    kwargs: dict[str, Any] = {
        "granules": granules,
        "access": args.access,
        "concat_dim": args.concat_dim,
        "group": args.group,
        "loadable_variables": args.loadable_variables,
        "pad": args.pad,
    }
    if args.mosaic_dims:
        kwargs["mosaic_dims"] = args.mosaic_dims

    print("Virtualizing …")
    t0 = time.perf_counter()
    vds = earthaccess.virtualize(**kwargs)
    elapsed = time.perf_counter() - t0
    print(f"Virtualized in {elapsed:.2f} s")
    print(f"  Dataset shape: {dict(vds.dims)}")
    print(f"  Variables: {list(vds.data_vars)}")
    return vds


def _write_kerchunk(
    vds: xr.Dataset,
    output: str,
    format: str,  # "json" or "parquet"
) -> None:
    """Serialize virtual dataset to kerchunk references."""
    print(f"Writing kerchunk {format} to {output!r} …")
    t0 = time.perf_counter()
    if format == "json":
        vds.vz.to_kerchunk(output, format="json")
    else:
        vds.vz.to_kerchunk(output, format="parquet")
    elapsed = time.perf_counter() - t0
    print(f"Written in {elapsed:.2f} s")


def _write_icechunk(vds: xr.Dataset, output: str) -> None:
    """Serialize virtual dataset to an icechunk store."""
    print(f"Writing icechunk store to {output!r} …")
    t0 = time.perf_counter()

    repo = icechunk.Repository.create(
        icechunk.local_filesystem_storage(output),
    )
    session = repo.writable_session("main")

    vds.vz.to_icechunk(session.store)
    session.commit("virtualize_with_earthaccess")

    elapsed = time.perf_counter() - t0
    print(f"Written in {elapsed:.2f} s")


def _verify(output: str, output_format: str) -> None:
    """Quick smoke-test that the output can be re-opened."""
    print("Verifying output …")
    if output_format == "kerchunk_json":
        ds = xr.open_dataset(output, engine="kerchunk")
    elif output_format == "kerchunk_parquet":
        ds = xr.open_dataset(output, engine="kerchunk")
    else:  # icechunk
        repo = icechunk.Repository.open(icechunk.local_filesystem_storage(output))
        session = repo.readonly_session("main")
        ds = xr.open_zarr(session.store, consolidated=False)
    print(f"  Re-opened OK: {dict(ds.dims)}")


def main() -> int:
    args = _parse_args()

    if not _auth():
        return 1

    granules = _search(args)
    vds = _virtualize(args, granules)

    if args.output_format.startswith("kerchunk"):
        _write_kerchunk(vds, args.output, args.reference_format)
    elif args.output_format == "icechunk":
        _write_icechunk(vds, args.output)
    else:
        raise ValueError(f"Unknown output format: {args.output_format}")

    _verify(args.output, args.output_format)
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
