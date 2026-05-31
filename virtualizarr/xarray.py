from __future__ import annotations

import os
import warnings
from collections.abc import Callable, Iterable, Mapping, MutableMapping, Sequence
from concurrent.futures import Executor
from functools import partial
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Hashable,
    Literal,
    Optional,
    cast,
)

import numpy as np
import xarray as xr
import xarray.indexes
from obspec_utils.registry import ObjectStoreRegistry
from xarray import DataArray, Dataset, Index, combine_by_coords
from xarray.backends.common import _find_absolute_paths
from xarray.core import dtypes
from xarray.core.types import NestedSequence
from xarray.structure.combine import _infer_concat_order_from_positions, _nested_combine

from virtualizarr.manifests import ManifestArray, ManifestGroup, ManifestStore
from virtualizarr.manifests.manifest import validate_and_normalize_path_to_uri
from virtualizarr.parallel import get_executor
from virtualizarr.parsers.typing import Parser
from virtualizarr.utils import compose

if TYPE_CHECKING:
    from xarray.core.types import (
        CombineAttrsOptions,
        CompatOptions,
        JoinOptions,
    )


def open_virtual_datatree(
    url: str,
    registry: ObjectStoreRegistry,
    parser: Parser,
    *,
    loadable_variables: Iterable[str] | None = None,
    decode_times: bool | None = None,
) -> xr.DataTree:
    """
    Open an archival data source as an [xarray.DataTree][] wrapping virtualized zarr arrays.

    See the `loadable_variables` kwarg for a description of which data variables are loaded vs.
    virtualized.

    Parameters
    ----------
    url
        The url of the data source to virtualize. The URL should include a scheme. For example:

        - `url="file:///Users/my-name/Documents/my-project/my-data.nc"` for a local data source.
        - `url="s3://my-bucket/my-project/my-data.nc"` for a remote data source on an S3 compatible cloud.

    registry
        An [ObjectStoreRegistry][obspec_utils.registry.ObjectStoreRegistry] for resolving urls and reading data.
    parser
        A parser to use for the given data source. For example:

        - [virtualizarr.parsers.HDFParser][] for virtualizing NetCDF4 or HDF5 files.
        - [virtualizarr.parsers.FITSParser][] for virtualizing FITS files.
        - [virtualizarr.parsers.NetCDF3Parser][] for virtualizing NetCDF3 files.
        - [virtualizarr.parsers.KerchunkJSONParser][] for re-opening Kerchunk JSONs.
        - [virtualizarr.parsers.KerchunkParquetParser][] for re-opening Kerchunk Parquets.
        - [virtualizarr.parsers.ZarrParser][] for virtualizing Zarr stores.
        - [virtualizarr.parsers.ZarrParser][] for virtualizing Zarr stores.
        - [virtual_tiff.VirtualTIFF][] for virtualizing TIFFs.

    loadable_variables
        If ``None`` (the default), dimension coordinate variables (1D variables whose name matches
        their dimension) will be loaded automatically to enable xarray indexing.

        If an empty iterable, no variables will be loaded.

        Other options are not yet supported.

    decode_times
        Bool that is passed into [xarray.open_dataset][]. Allows time to be decoded into a datetime object.

    Returns
    -------
    vds
        An [xarray.DataTree][] containing virtual chunk references for all variables.

    Examples
    --------

    Virtualize a Cloud Optimized GeoTIFF (COG) using [virtual_tiff.VirtualTIFF][]:

    ```python
    from obstore.store import S3Store

    from virtualizarr import open_virtual_datatree
    from obspec_utils.registry import ObjectStoreRegistry
    from virtual_tiff import VirtualTIFF

    # Access a public Sentinel-2 COG from AWS
    store = S3Store("sentinel-cogs", region="us-west-2", skip_signature=True)
    registry = ObjectStoreRegistry({"s3://sentinel-cogs/": store})
    url = "s3://sentinel-cogs/sentinel-s2-l2a-cogs/12/S/UF/2022/6/S2B_12SUF_20220609_0_L2A/B04.tif"
    parser = VirtualTIFF(ifd_layout="nested")

    with open_virtual_datatree(url=url, parser=parser, registry=registry) as vdt:
        print(vdt)
    ```

    Virtualize a NetCDF4 file using the the [virtualizarr.parsers.HDFParser][]:

    ```python
    from obstore.store import HTTPStore

    from virtualizarr import open_virtual_datatree
    from virtualizarr.parsers import HDFParser
    from obspec_utils.registry import ObjectStoreRegistry

    base = "https://github.com"
    url = f"{base}/pydata/xarray-data/raw/refs/heads/master/precipitation.nc4"

    store = HTTPStore(base)

    parser = HDFParser()
    registry = ObjectStoreRegistry({base: store})

    vdt = open_virtual_datatree(url=url, registry=registry, parser=parser)
    print(vdt)
    ```

    Load prevent loading variables from any groups (default loads the coordinate variables "time", "lat", and "lon"):

    ```python
    vdt = open_virtual_datatree(
        url=url,
        registry=registry,
        parser=parser,
        loadable_variables=[],
    )
    ```

    Drop variables from a specific group after opening:

    ```python
    vdt = open_virtual_datatree(
        url=url,
        registry=registry,
        parser=parser,
    )
    vdt["/observed"] = vdt["/observed"].to_dataset().drop_vars(["lon"])
    ```

    """
    filepath = validate_and_normalize_path_to_uri(url, fs_root=Path.cwd().as_uri())

    if loadable_variables:
        raise NotImplementedError(
            f"Only `loadable_variables=[]` or `loadable_variables=None` are supported, got loadable_variables={loadable_variables}"
        )
    manifest_store = parser(
        url=filepath,
        registry=registry,
    )

    vdt = manifest_store.to_virtual_datatree(
        loadable_variables=loadable_variables,
        decode_times=decode_times,
    )

    # mirror xarray.open_datatree behaviour by recording the source url
    if "source" not in vdt.encoding:
        vdt.encoding["source"] = filepath

    return vdt


def open_virtual_dataset(
    url: str,
    registry: ObjectStoreRegistry,
    parser: Parser,
    drop_variables: Iterable[str] | None = None,
    loadable_variables: Iterable[str] | None = None,
    decode_times: bool | None = None,
) -> xr.Dataset:
    """
    Open an archival data source as an [xarray.Dataset][] wrapping virtualized zarr arrays.

    No data variables will be loaded unless specified in the ``loadable_variables`` kwarg (in which case they will open as lazily indexed arrays using xarray's standard lazy indexing classes).
    Coordinate variables are loaded by default following xarray's behavior.

    Parameters
    ----------
    url
        The url of the data source to virtualize. The URL should include a scheme. For example:

        - `url="file:///Users/my-name/Documents/my-project/my-data.nc"` for a local data source.
        - `url="s3://my-bucket/my-project/my-data.nc"` for a remote data source on an S3 compatible cloud.

    registry
        An [ObjectStoreRegistry][obspec_utils.registry.ObjectStoreRegistry] for resolving urls and reading data.
    parser
        A parser to use for the given data source. For example:

        - [virtualizarr.parsers.HDFParser][] for virtualizing NetCDF4 or HDF5 files.
        - [virtualizarr.parsers.FITSParser][] for virtualizing FITS files.
        - [virtualizarr.parsers.IcechunkParser][] for virtualizing existing icechunk repos.
        - [virtualizarr.parsers.NetCDF3Parser][] for virtualizing NetCDF3 files.
        - [virtualizarr.parsers.DMRPPParser][] for virtualizing DMR++ files.
        - [virtualizarr.parsers.KerchunkJSONParser][] for re-opening Kerchunk JSONs.
        - [virtualizarr.parsers.KerchunkParquetParser][] for re-opening Kerchunk Parquets.
        - [virtualizarr.parsers.ZarrParser][] for virtualizing Zarr stores.
        - [virtual_tiff.VirtualTIFF][] for virtualizing TIFFs.

    drop_variables
        Variables in the data source to drop before returning.
    loadable_variables
        Variables in the data source to load as Dask/NumPy arrays instead of as virtual arrays.
    decode_times
        Bool that is passed into [xarray.open_dataset][]. Allows time to be decoded into a datetime object.

    Returns
    -------
    vds
        An [xarray.Dataset][] containing virtual chunk references for all variables not included
        in `loadable_variables` and normal lazily indexed arrays for each variable in `loadable_variables`.
    """
    filepath = validate_and_normalize_path_to_uri(url, fs_root=Path.cwd().as_uri())

    manifest_store = parser(
        url=filepath,
        registry=registry,
    )

    ds = manifest_store.to_virtual_dataset(
        loadable_variables=loadable_variables,
        decode_times=decode_times,
    )
    ds = ds.drop_vars(list(drop_variables or ()))

    # mirror xarray.open_dataset behaviour by recording the source url
    if "source" not in ds.encoding:
        ds.encoding["source"] = filepath

    return ds


def _is_virtual_variable(var: xr.Variable) -> bool:
    """Return True if the variable is backed by a ManifestArray."""
    return isinstance(var.data, ManifestArray)


def _compute_pad_targets(
    datasets: list[xr.Dataset],
    scope_dims: set[str] | None,
) -> dict[str, int]:
    """Compute max sizes per-dim across datasets, only for dims that differ.

    If *scope_dims* is given, only those dims are considered.
    """
    dim_sets: dict[str, set[int]] = {}
    for ds in datasets:
        for dim, size in ds.sizes.items():
            dim_s = cast(str, dim)
            if scope_dims is not None and dim_s not in scope_dims:
                continue
            dim_sets.setdefault(dim_s, set()).add(size)

    targets: dict[str, int] = {}
    for dim, sizes in dim_sets.items():
        if len(sizes) > 1:
            targets[dim] = max(sizes)
    return targets


def _apply_pad_to_dataset(
    ds: xr.Dataset,
    targets: dict[str, int],
    target_chunks_map: dict[str, tuple[int, ...]] | None = None,
) -> tuple[xr.Dataset, dict[str, int]]:
    """Pad all variables (virtual and loaded) to given dim targets.

    Returns (padded_ds, report_dict) where report_dict maps dim -> target.
    """
    report: dict[str, int] = {}
    padded_vars: dict[str, xr.Variable] = {}

    for _name, var in ds.variables.items():
        name = cast(str, _name)

        # --- virtual variables: pad ManifestArray ---
        if _is_virtual_variable(var):
            ma: ManifestArray = var.data
            new_shape = list(ma.shape)
            for ax, (_dim_name, old_len) in enumerate(zip(var.dims, ma.shape)):
                dim_name = cast(str, _dim_name)
                t = targets.get(dim_name)
                if t is not None and t > old_len:
                    new_shape[ax] = t
                    report[dim_name] = max(report.get(dim_name, 0), t)

            if target_chunks_map and name in target_chunks_map:
                tc = target_chunks_map[name]
                if tuple(tc) != tuple(ma.chunks):
                    ma = ma._remap_chunks(tc)

            new_shape_t = tuple(new_shape)
            if new_shape_t != ma.shape:
                ma = ma.pad_to_shape(new_shape_t)
                # Convert missing fill-row cells to inlined bytes so that
                # consolidate_chunks can later merge boundary cells that mix
                # real virtual sub-chunks with fill-value sub-chunks.
                ma = ma.fill_missing_with_inline()

            padded_vars[name] = xr.Variable(data=ma, dims=var.dims, attrs=var.attrs)
            continue

        # --- loaded variables: pad numpy array if needed ---
        new_shape = list(var.shape)
        needs_pad = False
        for ax, _dim_name in enumerate(var.dims):
            dim_name = cast(str, _dim_name)
            t = targets.get(dim_name)
            if t is not None and t > new_shape[ax]:
                new_shape[ax] = t
                needs_pad = True
                report[dim_name] = max(report.get(dim_name, 0), t)

        if needs_pad:
            arr = var.values
            pad_width: list[tuple[int, int]] = []
            for ax, _dim_name in enumerate(var.dims):
                dim_name = cast(str, _dim_name)
                t = targets.get(dim_name)
                old_len = var.shape[ax]
                if t is not None and t > old_len:
                    pad_width.append((0, t - old_len))
                else:
                    pad_width.append((0, 0))

            # Determine fill value
            fv = var.attrs.get("_FillValue")
            if fv is None:
                fv = var.attrs.get("missing_value")
            if fv is None:
                if np.issubdtype(var.dtype, np.floating):
                    fv = float("nan")
                elif np.issubdtype(var.dtype, np.integer):
                    fv = 0
                else:
                    fv = None

            padded_arr = np.pad(arr, pad_width, mode="constant", constant_values=fv)
            padded_vars[name] = xr.Variable(
                data=padded_arr, dims=var.dims, attrs=var.attrs
            )
        else:
            padded_vars[name] = var

    new_ds = construct_fully_virtual_dataset(padded_vars, attrs=ds.attrs)
    return new_ds, report


def _is_regular_coord(arr: "np.ndarray") -> bool:
    """Return True iff *arr* is a 1-D array with uniform non-zero spacing."""

    arr = np.asarray(arr)
    if arr.ndim != 1 or arr.size < 2:
        return arr.size == 1  # single-element is trivially regular
    diffs = np.diff(arr.astype("float64"))
    if not np.isfinite(diffs).all():
        return False
    step = diffs[0]
    if step == 0.0:
        return False
    return bool(np.allclose(diffs, step, rtol=1e-9, atol=0.0))


def _build_union_coord(arrays: "list[np.ndarray]") -> "np.ndarray":
    """Build a regular union coordinate from a list of 1-D regular coordinate arrays.

    All arrays must share the same uniform spacing (sign and magnitude).
    The union grid is the smallest regular grid that contains every element of
    every input array.

    Raises
    ------
    ValueError
        If any array is irregular, if spacings differ, or if any array is not
        on the common grid (grid misalignment).
    """

    if not arrays:
        raise ValueError("arrays must be non-empty")

    arrays = [np.asarray(a, dtype="float64") for a in arrays]

    # validate regularity of each array
    for i, a in enumerate(arrays):
        if not _is_regular_coord(a):
            raise ValueError(
                f"Array {i} is not a regular (uniformly-spaced) coordinate."
            )

    # determine common step from the first multi-element array
    step: float | None = None
    for a in arrays:
        if a.size >= 2:
            s = float(np.diff(a)[0])
            if step is None:
                step = s
            elif not np.isclose(s, step, rtol=1e-9, atol=0.0):
                raise ValueError(
                    f"Coordinate spacing mismatch: expected {step} but got {s}. "
                    "All coordinate arrays must share the same spacing."
                )

    if step is None:
        # all single-element: just unique values
        return np.unique(np.concatenate(arrays))

    # union extents
    if step > 0:
        global_min = float(min(a[0] for a in arrays))
        global_max = float(max(a[-1] for a in arrays))
    else:
        global_min = float(min(a[-1] for a in arrays))
        global_max = float(max(a[0] for a in arrays))

    # check grid alignment: each array's first element must be on the global grid
    for i, a in enumerate(arrays):
        offset = (a[0] - global_min if step > 0 else global_max - a[0])
        if not np.isclose(offset % abs(step), 0.0, atol=abs(step) * 1e-6):
            raise ValueError(
                f"Array {i} is not aligned to the common grid "
                f"(global_min={global_min}, step={step}, a[0]={a[0]}, "
                f"remainder={offset % abs(step)})."
            )

    n = round((global_max - global_min) / abs(step)) + 1
    return np.linspace(global_min if step > 0 else global_max,
                       global_max if step > 0 else global_min,
                       n)


# ---------------------------------------------------------------------------
# Mosaic helpers
# ---------------------------------------------------------------------------

def _compute_mosaic_plan(
    datasets: "list[xr.Dataset]",
    mosaic_dims: "list[str]",
) -> "tuple[dict[str, np.ndarray], dict[str, list[int]]]":
    """Validate coords and compute the union grid for each mosaic dimension.

    Parameters
    ----------
    datasets
        List of virtual datasets (after optional preprocessing).
    mosaic_dims
        Dimension names to mosaic spatially (e.g. ``['y', 'x']``).

    Returns
    -------
    union_coords : dict[dim, 1-D float64 array]
        The union coordinate array for each mosaic dim.
    offsets : dict[dim, list[int]]
        For each mosaic dim, the element-level offset of each dataset's
        coordinate within the union grid (length = len(datasets)).

    Raises
    ------
    ValueError
        If any mosaic-dim coordinate is still virtual, irregular, or
        misaligned across datasets.
    """

    union_coords: dict[str, np.ndarray] = {}
    offsets: dict[str, list[int]] = {}

    for dim in mosaic_dims:
        coord_arrays: list[np.ndarray] = []
        for i, ds in enumerate(datasets):
            if dim not in ds.variables:
                raise ValueError(
                    f"mosaic_dims: dimension '{dim}' has no coordinate variable "
                    f"in dataset {i}."
                )
            var = ds.variables[dim]
            if _is_virtual_variable(var):
                raise ValueError(
                    f"mosaic_dims: coordinate '{dim}' is still virtual in dataset {i}. "
                    f"Add '{dim}' to loadable_variables so it is loaded into memory "
                    "before mosaicking."
                )
            coord_arrays.append(np.asarray(var.values, dtype="float64"))

        # _build_union_coord validates regularity, consistent spacing, alignment
        union = _build_union_coord(coord_arrays)
        union_coords[dim] = union

        step = float(np.diff(union)[0]) if union.size >= 2 else 1.0
        dim_offsets: list[int] = []
        for ca in coord_arrays:
            # find the index in the union where this coord starts
            # step may be negative (e.g. decreasing y), so use signed division
            start_val = float(ca[0])
            idx = round((start_val - float(union[0])) / step)
            dim_offsets.append(int(idx))
        offsets[dim] = dim_offsets

    return union_coords, offsets


def _apply_mosaic_to_dataset(
    ds: "xr.Dataset",
    ds_idx: int,
    mosaic_dims: "list[str]",
    union_coords: "dict[str, np.ndarray]",
    offsets: "dict[str, list[int]]",
) -> "xr.Dataset":
    """Place a single dataset's virtual arrays into the union grid.

    For each virtual variable whose dimensions include a mosaic dim:
    1. Remap the ManifestArray to chunk_size=1 on every mosaic axis (so any
       element offset is chunk-aligned).
    2. Call ``place_in_grid`` with the correct element offset.

    Loaded 1-D coordinate variables for mosaic dims are replaced with the
    union coordinate array.
    """

    new_vars: dict[str, xr.Variable] = {}

    # union shape for each mosaic dim
    union_sizes = {dim: int(union_coords[dim].size) for dim in mosaic_dims}

    for _name, var in ds.variables.items():
        name = cast(str, _name)
        if not _is_virtual_variable(var):
            # Replace loaded 1-D coord if it is a mosaic dim
            if var.dims == (name,) and name in mosaic_dims:
                union_arr = union_coords[name]
                new_vars[name] = xr.Variable(
                    dims=(name,), data=union_arr, attrs=var.attrs
                )
            else:
                new_vars[name] = var
            continue

        ma: ManifestArray = var.data
        dims = var.dims

        # check whether any mosaic dim appears in this variable's dimensions
        mosaic_axes = [
            ax for ax, d in enumerate(dims) if cast(str, d) in mosaic_dims
        ]
        if not mosaic_axes:
            new_vars[name] = var
            continue

        # --- step 1: build new_shape and element offset for place_in_grid ---
        original_chunks = ma.chunks
        new_shape = list(ma.shape)
        elem_offset = [0] * ma.ndim
        for ax, dim in enumerate(dims):
            if dim in mosaic_dims:
                new_shape[ax] = union_sizes[dim]
                elem_offset[ax] = offsets[dim][ds_idx]

        # --- step 2: place chunks into the union grid ---
        #
        # Fast path (chunk-aligned offset, works for compressed data too):
        # When the element offset on every mosaic axis is an exact multiple of
        # the chunk size on that axis, existing chunk boundaries map directly
        # onto the union grid.  We can call place_in_grid with the original
        # chunks without splitting or decompressing anything.
        #
        # Slow path (uncompressed data only):
        # When the offset is NOT chunk-aligned on some axis we fall back to
        # _remap_chunks (chunk_size=1) + place_in_grid + consolidate_chunks.
        # This requires uncompressed data because it byte-splits chunks.

        chunk_aligned = all(
            elem_offset[ax] % original_chunks[ax] == 0
            for ax in mosaic_axes
        )

        if chunk_aligned:
            # Fast path: place whole chunks directly — safe for compressed data.
            ma = ma.place_in_grid(tuple(new_shape), tuple(elem_offset), policy="error")
        else:
            # Slow path: requires uncompressed data.
            target_chunks = list(original_chunks)
            for ax in mosaic_axes:
                target_chunks[ax] = 1
            ma = ma._remap_chunks(tuple(target_chunks))
            ma = ma.place_in_grid(tuple(new_shape), tuple(elem_offset), policy="error")

            # Consolidate sub-chunks back to original chunk sizes where the
            # union shape is divisible by the original chunk size.
            consolidate_target = list(ma.chunks)
            any_consolidation = False
            for ax in mosaic_axes:
                desired = original_chunks[ax]
                if desired > 1 and ma.shape[ax] % desired == 0:
                    consolidate_target[ax] = desired
                    any_consolidation = True
            if any_consolidation:
                try:
                    ma = ma.consolidate_chunks(tuple(consolidate_target))
                except ValueError:
                    # Non-contiguous or cross-file sub-chunks: leave as-is.
                    pass

        new_vars[name] = xr.Variable(dims=dims, data=ma, attrs=var.attrs)

    return construct_fully_virtual_dataset(new_vars, attrs=ds.attrs)


def _resolve_pad_scope(
    pad: str | Mapping[str, int | None],
    datasets: list[xr.Dataset],
) -> set[str] | None:
    """Convert the user `pad` argument to a set of dims to consider.

    Returns None for the special ``"auto"`` mode (means "inspect all").
    """
    if isinstance(pad, str):
        if pad == "auto":
            return None
        raise ValueError(
            f"Invalid pad value: '{pad}'. Expected 'auto', 'none', "
            "or a dict of dim -> target."
        )
    return set(pad.keys())


def _collect_explicit_targets(
    pad: str | Mapping[str, int | None],
) -> dict[str, int]:
    """Extract explicit (non-None) integer targets from a pad mapping."""
    if isinstance(pad, str):
        return {}
    return {dim: val for dim, val in pad.items() if isinstance(val, int)}


def _compute_common_chunks(
    datasets: list[xr.Dataset],
    pad_dims: set[str] | None = None,
) -> dict[str, tuple[int, ...]]:
    """Compute common chunk shape per variable when chunk shapes differ.

    For uncompressed variables on padded dims, uses chunk size 1 so that any
    pad target is chunk-aligned (safe for contiguous/uncompressed data where
    ``_remap_chunks`` can split byte ranges).  For compressed variables the
    chunk sizes should already match; if they differ we use the minimum
    (``_remap_chunks`` will raise if called on compressed data).
    For non-pad dims the element-wise minimum across datasets is used.
    """
    from virtualizarr.codecs import is_uncompressed

    pad_dims = pad_dims or set()
    var_chunks: dict[str, set[tuple[int, ...]]] = {}
    var_dims: dict[str, tuple[str, ...]] = {}
    var_compressed: dict[str, bool] = {}  # True if ANY array for this var is compressed
    for ds in datasets:
        for _name, var in ds.variables.items():
            name = cast(str, _name)
            if _is_virtual_variable(var):
                ma = var.data
                var_chunks.setdefault(name, set()).add(ma.chunks)
                var_dims.setdefault(name, cast("tuple[str, ...]", var.dims))
                if not is_uncompressed(ma):
                    var_compressed[name] = True

    common: dict[str, tuple[int, ...]] = {}
    for name, chunk_set in var_chunks.items():
        dims = var_dims.get(name, ())
        if len(chunk_set) > 1:
            zipped = zip(*chunk_set)
            parts: list[int] = []
            compressed = var_compressed.get(name, False)
            for ax, dim_values in enumerate(zipped):
                if ax < len(dims) and dims[ax] in pad_dims and not compressed:
                    # uncompressed on a pad dim: use chunk_size=1 so any pad
                    # target is chunk-aligned after _remap_chunks splits bytes
                    parts.append(1)
                else:
                    parts.append(min(dim_values))
            common[name] = tuple(parts)
    return common


def _consolidate_dataset(
    ds: xr.Dataset,
    original_chunks: dict[str, tuple[int, ...]],
    storage_options: dict | None = None,
) -> xr.Dataset:
    """Consolidate ManifestArray variables back to pre-padding chunk sizes.

    After padding + concat, uncompressed variables on padded dims end up with
    chunk_size=1 everywhere.  This function tries to merge those size-1 chunks
    back to the original (per-granule) chunk sizes recorded before padding.
    Sub-chunks that are contiguous and from the same source file are merged;
    cells where every sub-chunk is missing are kept as missing.  Mixed
    virtual+inlined cells (boundary rows converted to fill bytes by
    fill_missing_with_inline) are merged by reading the virtual bytes once
    and concatenating with the inlined fill bytes.  Variables that cannot be
    consolidated are left unchanged.
    """
    new_vars: dict[str, xr.Variable] = {}
    for _name, var in ds.variables.items():
        name = cast(str, _name)
        if _is_virtual_variable(var) and name in original_chunks:
            ma: ManifestArray = var.data
            target = original_chunks[name]
            if target != ma.chunks:
                try:
                    ma = ma.consolidate_chunks(target, storage_options=storage_options)
                    var = xr.Variable(data=ma, dims=var.dims, attrs=var.attrs)
                except (ValueError, NotImplementedError):
                    pass  # leave unchanged if not consolidatable
        new_vars[name] = var

    return construct_fully_virtual_dataset(new_vars, attrs=ds.attrs)


def _original_chunks_from_datasets(
    datasets: list[xr.Dataset],
) -> dict[str, tuple[int, ...]]:
    """Record the natural chunk sizes per variable from a list of datasets,
    taking the maximum along each axis (largest natural chunk wins)."""
    result: dict[str, tuple[int, ...]] = {}
    for ds in datasets:
        for _name, var in ds.variables.items():
            name = cast(str, _name)
            if _is_virtual_variable(var):
                ma: ManifestArray = var.data
                if name not in result:
                    result[name] = ma.chunks
                else:
                    result[name] = tuple(
                        max(a, b) for a, b in zip(result[name], ma.chunks)
                    )
    return result


def _emit_pad_warning(report: dict[str, int], stage: str = "") -> None:
    """Emit a single warning summarizing padding actions."""
    if not report:
        return
    stage_prefix = f"[{stage}] " if stage else ""
    dims_str = ", ".join(f"{d}={s}" for d, s in sorted(report.items()))
    warnings.warn(
        f"{stage_prefix}Applied padding: {dims_str}"
    )


def _normalize_coords_for_concat(
    datasets: list[xr.Dataset],
    concat_dim: str,
) -> list[xr.Dataset]:
    """Ensure all datasets share identical 1-D coords for non-concat dims.

    ``xr.concat`` with ``join='outer'`` tries to align every non-concat
    dimension.  When coordinate values differ (even if shapes match),
    xarray issues fancy-index reindexers that ``ManifestArray`` does not
    support.  We side-step that by copying a reference coordinate
    (taken from the first dataset that has it) to every other dataset.
    """
    if len(datasets) <= 1:
        return datasets

    result = list(datasets)

    # collect every non-concat dimension that is present as a *loaded* 1-D coord
    # (skip virtual ManifestArray coords — assign_coords does not support them)
    dims_to_fix: set[str] = set()
    for ds in result:
        for _dim in ds.dims:
            dim = cast(str, _dim)
            if dim == concat_dim:
                continue
            if dim in ds.coords and ds.coords[dim].dims == (dim,):
                if not _is_virtual_variable(ds.coords[dim]):
                    dims_to_fix.add(dim)

    for dim in dims_to_fix:
        # pick a reference coord from the first dataset that carries it
        ref = None
        for ds in result:
            if dim in ds.coords:
                ref = ds.coords[dim]
                break
        if ref is None:
            continue

        for i, ds in enumerate(result):
            if dim in ds.coords:
                result[i] = ds.assign_coords({dim: ref})

    return result


def _storage_options_from_registry(registry: "ObjectStoreRegistry") -> dict:
    """Return fsspec storage_options for the first HTTPS store in the registry.

    Extracts the Bearer token from an ``obstore`` ``HTTPStore`` (if present) and
    converts it to fsspec ``{"headers": {"Authorization": "Bearer <token>"}}``
    so that ``fsspec.open`` can read boundary chunks from authenticated DAAC URLs.

    For local ``file://`` registries returns ``{}`` (fsspec handles local paths
    without extra options).
    """
    try:
        for store in registry._iter_stores():
            store_type = type(store).__name__
            if store_type == "HTTPStore":
                client_opts = getattr(store, "client_options", {}) or {}
                headers = client_opts.get("default_headers", {})
                if headers:
                    # obstore stores header values as bytes; fsspec wants str
                    str_headers = {
                        k: v.decode() if isinstance(v, bytes) else v
                        for k, v in headers.items()
                    }
                    return {"headers": str_headers}
    except Exception:
        pass
    return {}


def _nested_combine_with_padding(
    datasets: list[xr.Dataset],
    concat_dims: list,
    ids: list,
    pad_scope: set[str] | None,
    explicit_targets: dict[str, int],
    pad_warn: bool,
    consolidate: bool = True,
    storage_options: dict | None = None,
) -> xr.Dataset:
    """Combine datasets with padding, processing from innermost to outermost dim.

    ``concat_dims[0]`` is the outermost dimension (top-level list in the
    nested structure); ``concat_dims[-1]`` is the innermost.  IDs are tuples
    where ``id[0]`` is the outer index and ``id[-1]`` is the inner index.

    We process innermost first so that padding is applied within each inner
    group independently.  This is the correct behaviour for TEMPO-like data
    where only the innermost concat dimension (``mirror_step``) is ragged.
    """
    total_report: dict[str, int] = {}

    # Map each dataset to its id tuple (normalised to a tuple of ints).
    indexed: list[tuple[tuple[int, ...], xr.Dataset]] = []
    for sid, ds in zip(ids, datasets):
        key = tuple(sid) if isinstance(sid, (tuple, list)) else (int(sid),)
        indexed.append((key, ds))

    def _pad_and_normalize(
        parts_list: list[xr.Dataset], dim: str
    ) -> list[xr.Dataset]:
        """Pad within a group + normalise coords for the upcoming concat."""
        auto = _compute_pad_targets(parts_list, pad_scope)
        targets = {**auto, **explicit_targets}
        chunk_map = _compute_common_chunks(parts_list, pad_dims=set(targets.keys()))
        if targets or chunk_map:
            padded: list[xr.Dataset] = []
            for ds in parts_list:
                ds_padded, report = _apply_pad_to_dataset(ds, targets, chunk_map)
                padded.append(ds_padded)
                total_report.update(report)
            parts_list = padded
        return _normalize_coords_for_concat(parts_list, dim)

    # Process from innermost dimension to outermost.
    # At each step: group by the outer prefix (all but the last key element),
    # pad within each group, concat along the current (innermost remaining) dim.
    ndims = len(concat_dims)

    for step in range(ndims - 1, -1, -1):
        # dim for this step is concat_dims[step] (innermost not-yet-reduced)
        dim = concat_dims[step]

        # Group by the outer-prefix portion of the id (everything before index `step`).
        groups: dict[tuple[int, ...], list[tuple[tuple[int, ...], xr.Dataset]]] = {}
        for key, ds in indexed:
            prefix = key[:step]  # outer indices (empty tuple for outermost step)
            groups.setdefault(prefix, []).append((key, ds))

        next_indexed: list[tuple[tuple[int, ...], xr.Dataset]] = []
        for prefix, group_items in groups.items():
            group_ds = [ds for _, ds in group_items]

            if len(group_ds) == 1:
                # Nothing to concat at this level; just carry forward.
                next_indexed.append((prefix, group_ds[0]))
                continue

            # Record natural chunk sizes BEFORE padding remaps to chunk_size=1
            original_chunks = _original_chunks_from_datasets(group_ds) if consolidate else {}

            padded_group = _pad_and_normalize(group_ds, dim)
            combined = xr.concat(
                padded_group,
                dim=dim,
                join="override",
                compat="override",
                coords="minimal",
                data_vars="all",
                combine_attrs="override",
            )

            if consolidate:
                # Target = original per-granule chunk size (NOT scaled by n).
                # After concat, the manifest has n*orig_rows chunks of size 1
                # along the concat dim. Each granule's rows are contiguous and
                # from the same source file, so consolidating to orig_rows merges
                # them cleanly. Consolidating across granule boundaries would
                # span multiple source files and correctly raise ValueError.
                target_chunks: dict[str, tuple[int, ...]] = {}
                for name, chunks in original_chunks.items():
                    var = combined.variables.get(name)
                    if var is None:
                        continue
                    target_chunks[name] = chunks  # keep original, don't multiply
                combined = _consolidate_dataset(combined, target_chunks, storage_options=storage_options)

            next_indexed.append((prefix, combined))

        indexed = next_indexed

    if pad_warn:
        _emit_pad_warning(total_report)

    assert len(indexed) == 1, f"Expected 1 dataset after all concat steps, got {len(indexed)}"
    return indexed[0][1]


def open_virtual_mfdataset(
    urls: (
        str
        | os.PathLike
        | Sequence[str | os.PathLike]
        | NestedSequence[str | os.PathLike]
    ),
    registry: ObjectStoreRegistry,
    parser: Parser,
    concat_dim: (
        str
        | DataArray
        | Index
        | Sequence[str]
        | Sequence[DataArray]
        | Sequence[Index]
        | None
    ) = None,
    compat: "CompatOptions" = "no_conflicts",
    preprocess: Callable[[Dataset], Dataset] | None = None,
    data_vars: Literal["all", "minimal", "different"] | list[str] = "all",
    coords="different",
    combine: Literal["by_coords", "nested"] = "by_coords",
    parallel: Literal["dask", "lithops", False] | type[Executor] = False,
    join: "JoinOptions" = "outer",
    attrs_file: str | os.PathLike | None = None,
    combine_attrs: "CombineAttrsOptions" = "override",
    pad: Literal["auto", "none"] | Mapping[str, int | None] = "auto",
    pad_policy: Literal["round_to_chunks", "error"] = "round_to_chunks",
    pad_warn: bool = True,
    mosaic_dims: "list[str] | None" = None,
    consolidate: bool = True,
    **kwargs,
) -> Dataset:
    """
    Open multiple data sources as a single virtual dataset.

    This function is explicitly modelled after [xarray.open_mfdataset][], and works in the same way.

    If `combine='by_coords'` then the function `combine_by_coords` is used to combine
    the datasets into one before returning the result, and if combine='nested' then
    `combine_nested` is used. The urls must be structured according to which
    combining function is used, the details of which are given in the documentation for
    `combine_by_coords` and `combine_nested`. By default `combine='by_coords'`
    will be used. Global attributes from the `attrs_file` are used
    for the combined dataset.

    Parameters
    ----------
    urls
        Same as in [virtualizarr.open_virtual_dataset][]
    registry
        An [ObjectStoreRegistry][obspec_utils.registry.ObjectStoreRegistry] for resolving urls and reading data.
    concat_dim
        Same as in [xarray.open_mfdataset][]
    compat
        Same as in [xarray.open_mfdataset][]
    preprocess
        Same as in [xarray.open_mfdataset][]
    data_vars
        Same as in [xarray.open_mfdataset][]
    coords
        Same as in [xarray.open_mfdataset][]
    combine
        Same as in [xarray.open_mfdataset][]
    parallel : "dask", "lithops", False, or type of subclass of [concurrent.futures.Executor][]
        Specify whether the open and preprocess steps of this function will be
        performed in parallel using [lithops][], `dask.delayed`, or any executor compatible
        with the [concurrent.futures][] interface, or in serial.
        Default is False, which will execute these steps in serial.
    join
        Same as in [xarray.open_mfdataset][]
    attrs_file
        Same as in [xarray.open_mfdataset][]
    combine_attrs
        Same as in [xarray.open_mfdataset][]
    **kwargs : optional
        Additional arguments passed on to [virtualizarr.open_virtual_dataset][]. For an
        overview of some of the possible options, see the documentation of
        [virtualizarr.open_virtual_dataset][].

    Returns
    -------
    vds
        An [xarray.Dataset][] containing virtual chunk references for all variables not included
        in `loadable_variables` and normal lazily indexed arrays for each variable in `loadable_variables`.

    Notes
    -----
    The results of opening each virtual dataset in parallel are sent back to the client process, so must not be too large. See the docs page on [Scaling][].
    """

    # TODO this is practically all just copied from xarray.open_mfdataset - an argument for writing a virtualizarr engine for xarray?

    # TODO list kwargs passed to open_virtual_dataset explicitly in docstring?

    paths = cast(NestedSequence[str], _find_absolute_paths(urls))

    if not paths:
        raise OSError("No data sources to open, pass urls to the `urls` parameter.")

    paths1d: list[str]
    if combine == "nested":
        if isinstance(concat_dim, str | DataArray) or concat_dim is None:
            concat_dim = [concat_dim]  # type: ignore[assignment]

        # This creates a flat list which is easier to iterate over, whilst
        # encoding the originally-supplied structure as "ids".
        # The "ids" are not used at all if combine='by_coords`.
        combined_ids_paths = _infer_concat_order_from_positions(paths)
        ids, paths1d = (
            list(combined_ids_paths.keys()),
            list(combined_ids_paths.values()),
        )
    elif concat_dim is not None:
        raise ValueError(
            "When combine='by_coords', passing a value for `concat_dim` has no "
            "effect. To manually combine along a specific dimension you should "
            "instead specify combine='nested' along with a value for `concat_dim`.",
        )
    else:
        paths1d = paths  # type: ignore[assignment]

    # TODO this refactored preprocess and executor logic should be upstreamed
    # into xarray - see https://github.com/pydata/xarray/pull/9932

    open_vds = partial(open_virtual_dataset, registry=registry, parser=parser, **kwargs)
    mapper = open_vds if preprocess is None else compose(preprocess, open_vds)
    make_executor = get_executor(parallel=parallel)

    with make_executor() as exec:
        virtual_datasets = list(exec.map(mapper, paths1d))

    # --- mosaic: place each dataset into a shared spatial union grid ---
    if mosaic_dims:
        union_coords, offsets = _compute_mosaic_plan(virtual_datasets, mosaic_dims)
        virtual_datasets = [
            _apply_mosaic_to_dataset(ds, i, mosaic_dims, union_coords, offsets)
            for i, ds in enumerate(virtual_datasets)
        ]

    if pad != "none":
        pad_scope = _resolve_pad_scope(pad, virtual_datasets)
        explicit_targets = _collect_explicit_targets(pad)

        if combine == "by_coords":
            auto_targets = _compute_pad_targets(virtual_datasets, pad_scope)
            targets = {**auto_targets, **explicit_targets}
            chunk_map = _compute_common_chunks(
                virtual_datasets, pad_dims=set(targets.keys())
            )

            full_report: dict[str, int] = {}
            padded_datasets: list[Dataset] = []
            for ds in virtual_datasets:
                padded_ds, rep = _apply_pad_to_dataset(ds, targets, chunk_map)
                padded_datasets.append(padded_ds)
                full_report.update(rep)
            virtual_datasets = padded_datasets

            if pad_warn:
                _emit_pad_warning(full_report)

        elif combine == "nested":
            staged = _nested_combine_with_padding(
                virtual_datasets,
                concat_dims=cast("list[Any]", concat_dim),
                ids=ids,
                pad_scope=pad_scope,
                explicit_targets=explicit_targets,
                pad_warn=pad_warn,
                consolidate=consolidate,
                storage_options=_storage_options_from_registry(registry),
            )
            return staged

    # TODO add file closers

    # Combine all datasets, closing them in case of a ValueError
    try:
        if combine == "nested":
            # Combined nested list by successive concat and merge operations
            # along each dimension, using structure given by "ids"
            combined_vds = _nested_combine(
                virtual_datasets,
                concat_dims=concat_dim,
                compat=compat,
                data_vars=data_vars,
                coords=coords,
                ids=ids,
                join=join,
                combine_attrs=combine_attrs,
                fill_value=dtypes.NA,
            )
        elif combine == "by_coords":
            # Redo ordering from coordinates, ignoring how they were ordered
            # previously
            combined_vds = combine_by_coords(
                virtual_datasets,
                compat=compat,
                data_vars=data_vars,
                coords=coords,
                join=join,
                combine_attrs=combine_attrs,
            )
        else:
            raise ValueError(
                f"{combine} is an invalid option for the keyword argument `combine`"
            )
    except ValueError:
        for vds in virtual_datasets:
            vds.close()
        raise

    # combined_vds.set_close(partial(_multi_file_closer, closers))

    # read global attributes from the attrs_file or from the first dataset
    if attrs_file is not None:
        if isinstance(attrs_file, os.PathLike):
            attrs_file = cast(str, os.fspath(attrs_file))
        combined_vds.attrs = virtual_datasets[paths1d.index(attrs_file)].attrs

    # TODO should we just immediately close everything?
    # TODO If loadable_variables is eager then we should have already read everything we're ever going to read into memory at this point

    return combined_vds


def construct_fully_virtual_dataset(
    virtual_vars: Mapping[str, xr.Variable],
    coord_names: Iterable[str] | None = None,
    attrs: dict[str, Any] | None = None,
) -> xr.Dataset:
    """Construct a fully virtual Dataset from constituent parts."""

    data_vars, coords = separate_coords(
        vars=virtual_vars,
        indexes={},  # we specifically avoid creating any indexes yet to avoid loading any data
        coord_names=coord_names,
    )

    vds = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs=attrs,
    )

    return vds


def construct_virtual_dataset(
    manifest_store: ManifestStore,
    group: str | None = None,
    loadable_variables: Iterable[Hashable] | None = None,
    decode_times: bool | None = None,
    reader_options: Optional[dict] = None,
) -> xr.Dataset:
    """
    Construct a fully or partly virtual dataset from a ManifestStore
    containing the contents of one group.

    """

    # TODO: Remove private API `._group`
    if group:
        raise NotImplementedError("ManifestStore does not yet support nested groups")
    else:
        manifestgroup = manifest_store._group

    fully_virtual_ds = manifestgroup.to_virtual_dataset()

    with xr.open_zarr(
        manifest_store,
        group=group,
        consolidated=False,
        zarr_format=3,
        chunks=None,
        decode_times=decode_times,
    ) as loadable_ds:
        return replace_virtual_with_loadable_vars(
            fully_virtual_ds, loadable_ds, loadable_variables
        )


def construct_virtual_datatree(
    manifest_store: ManifestStore,
    group: str = "",
    *,
    loadable_variables: Iterable[str] | None = None,
    decode_times: bool | None = None,
) -> xr.DataTree:
    """
    Construct a fully or partly virtual datatree from a ManifestStore.
    """
    node = manifest_store._group[group] if group else manifest_store._group

    if isinstance(node, ManifestArray):
        node = ManifestGroup(arrays={group: node}, attributes={})

    fully_loadable_datatree = xr.open_datatree(
        manifest_store,  # type: ignore[arg-type]
        group=group,
        engine="zarr",
        consolidated=False,
        zarr_format=3,
        decode_times=decode_times,
    )

    partially_loaded_datasets = {
        name: replace_virtual_with_loadable_vars(
            virtual_node.to_dataset(),
            fully_loadable_datatree[name].to_dataset(),
            loadable_variables,
        )
        for name, virtual_node in node.to_virtual_datatree().subtree_with_keys
    }

    return xr.DataTree.from_dict(partially_loaded_datasets)


def replace_virtual_with_loadable_vars(
    fully_virtual_ds: xr.Dataset,
    loadable_ds: xr.Dataset,
    loadable_variables: Iterable[Hashable] | None = None,
) -> xr.Dataset:
    """
    Merge a fully virtual and the corresponding fully loadable dataset, keeping only `loadable_variables` from the latter (plus defaults needed for indexes).
    """

    var_names_to_load: list[Hashable]

    if isinstance(loadable_variables, list):
        var_names_to_load = list(loadable_variables)
    elif loadable_variables is None:
        # If `loadable_variables` is None, then we have to explicitly match default
        # behaviour of xarray, i.e., load and create indexes only for dimension
        # coordinate variables.  We already have all the indexes and variables
        # we should be keeping - we just need to distinguish them.
        var_names_to_load = [
            name for name, var in loadable_ds.variables.items() if var.dims == (name,)
        ]
    else:
        raise ValueError(
            "loadable_variables must be an iterable of string variable names,"
            f" or None, but got type {type(loadable_variables)}"
        )

    # this will automatically keep any IndexVariables needed for loadable 1D coordinates
    loadable_var_names_to_drop = set(loadable_ds.variables).difference(
        var_names_to_load
    )
    ds_loadable_to_keep = loadable_ds.drop_vars(
        loadable_var_names_to_drop, errors="ignore"
    )

    ds_virtual_to_keep = fully_virtual_ds.drop_vars(var_names_to_load, errors="ignore")

    # we don't need `compat` or `join` kwargs here because there should be no variables with the same name in both datasets
    return xr.merge(
        [
            ds_loadable_to_keep,
            ds_virtual_to_keep,
        ],
    )


# TODO this probably doesn't need to actually support indexes != {}
def separate_coords(
    vars: Mapping[str, xr.Variable],
    indexes: MutableMapping[str, xr.Index],
    coord_names: Iterable[str] | None = None,
) -> tuple[dict[str, xr.Variable], xr.Coordinates]:
    """
    Try to generate a set of coordinates that won't cause xarray to automatically build a pandas.Index for the 1D coordinates.

    Currently requires this function as a workaround unless xarray PR #8124 is merged.

    Will also preserve any loaded variables and indexes it is passed.
    """

    if coord_names is None:
        coord_names = []

    # split data and coordinate variables (promote dimension coordinates)
    data_vars = {}
    coord_vars: dict[
        str, tuple[Hashable, Any, dict[Any, Any], dict[Any, Any]] | xr.Variable
    ] = {}
    found_coord_names: set[str] = set()
    # Search through variable attributes for coordinate names
    for var in vars.values():
        if "coordinates" in var.attrs:
            found_coord_names.update(var.attrs["coordinates"].split(" "))
    for name, var in vars.items():
        if name in coord_names or var.dims == (name,) or name in found_coord_names:
            # use workaround to avoid creating IndexVariables described here https://github.com/pydata/xarray/pull/8107#discussion_r1311214263
            if len(var.dims) == 1:
                dim1d, *_ = var.dims
                coord_vars[name] = (dim1d, var.data, var.attrs, var.encoding)

                if isinstance(var, xr.IndexVariable):
                    # unless variable actually already is a loaded IndexVariable,
                    # in which case we need to keep it and add the corresponding indexes explicitly
                    coord_vars[str(name)] = var
                    # TODO this seems suspect - will it handle datetimes?
                    indexes[name] = xarray.indexes.PandasIndex(var, dim1d)
            else:
                coord_vars[name] = var
        else:
            data_vars[name] = var

    coords = xr.Coordinates(coord_vars, indexes=indexes)

    return data_vars, coords
