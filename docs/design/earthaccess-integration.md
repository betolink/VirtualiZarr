"""
Analysis: What earthaccess needs to support VirtualiZarr's mosaic/padding API

Background
----------
VirtualiZarr's ``open_virtual_mfdataset`` gained three new parameters in the
mosaic branch:

  - ``mosaic_dims: list[str]`` — spatial dimensions to union into a common grid
  - ``pad: str | dict`` — padding strategy for ragged arrays (TEMPO-style)
  - ``loadable_variables: list[str]`` — coords that must be materialised

earthaccess currently has two entry points that call VirtualiZarr:

  1. ``earthaccess.virtualize(granules, ...)`` — creates virtual datasets
     from NASA Earthdata granules.
  2. ``earthaccess.open_virtual(uri, load=False)`` — opens an *already*
     virtualized kerchunk/parquet store.

This document analyses what each entry point needs to change.

---

1. earthaccess.virtualize() — creating virtual datasets from granules
--------------------------------------------------------------------

Current signature (relevant params):

    def virtualize(
        granules,
        *,
        concat_dim: str | None = None,
        preprocess: Callable | None = None,
        parser: ParserType = "DMRPPParser",
        ...
        **xr_combine_kwargs,
    ) -> xr.Dataset:

Current internal call (``_open_virtual_mfdataset``):

    return vz.open_virtual_mfdataset(
        urls=urls,
        registry=registry,
        parser=parser,
        preprocess=preprocess,
        parallel=parallel,
        combine="nested",
        concat_dim=concat_dim,
        data_vars=data_vars,
        coords=coords,
        compat=compat,
        combine_attrs=combine_attrs,
        **xr_combine_kwargs,
    )

What's missing
~~~~~~~~~~~~~~
The wrapper does NOT pass:

  - ``loadable_variables``  → needed so mosaic_dims coordinates are materialised
  - ``mosaic_dims``         → needed for spatial mosaicking (ITS-LIVE)
  - ``pad``                 → needed for ragged-scanline padding (TEMPO)

Required changes
~~~~~~~~~~~~~~~~

a) Add parameters to ``virtualize()``:

    def virtualize(
        granules,
        *,
        concat_dim: str | None = None,
        mosaic_dims: list[str] | None = None,      # NEW
        pad: str | dict[str, Any] | None = None,  # NEW
        loadable_variables: list[str] | None = None,  # NEW
        ...
        **xr_combine_kwargs,
    ) -> xr.Dataset:

b) Forward them through ``_open_virtual_mfdataset``:

    return vz.open_virtual_mfdataset(
        urls=urls,
        registry=registry,
        parser=parser,
        preprocess=preprocess,
        parallel=parallel,
        combine="nested",
        concat_dim=concat_dim,
        mosaic_dims=mosaic_dims,          # NEW
        pad=pad,                           # NEW
        loadable_variables=loadable_variables,  # NEW
        data_vars=data_vars,
        coords=coords,
        compat=compat,
        combine_attrs=combine_attrs,
        **xr_combine_kwargs,
    )

Why ``loadable_variables`` matters for earthaccess
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``loadable_variables`` parameter tells VirtualiZarr which coordinate arrays
must be read in full (not left virtual). For mosaic'd data, the coordinates of
the ``mosaic_dims`` MUST be materialised so that ``_compute_mosaic_plan`` can:

  1. Read the numeric ``y`` and ``x`` arrays from each granule
  2. Compute the union grid extent
  3. Calculate per-granule offsets

If earthaccess users pass ``mosaic_dims=["y","x"]`` without also listing ``y``
and ``x`` in ``loadable_variables``, VirtualiZarr raises::

    ValueError: mosaic_dims: dimension 'y' has no coordinate variable in dataset 0.

earthaccess could auto-populate ``loadable_variables`` from ``mosaic_dims``,
OR simply document that users must include them::

    # Example usage
    vds = earthaccess.virtualize(
        granules,
        concat_dim="time",
        mosaic_dims=["y", "x"],
        pad="none",
        loadable_variables=["y", "x", "time"],
    )

---

2. earthaccess.open_virtual(load=False) — opening pre-virtualized stores
------------------------------------------------------------------------

Current implementation (``_open_virtual_via_virtualizarr``):

    # 1. Read inline coordinates from the kerchunk refs via fsspec+zarr
    kds = xr.open_zarr(store, consolidated=False)

    # 2. Re-open the same refs with VirtualiZarr, SKIP all coordinates
    parser = KerchunkParquetParser(skip_variables=list(kds.coords))
    vds = vz.open_virtual_dataset(ref_path, parser=parser, registry=registry)

    # 3. Re-attach the REAL coordinates from step 1
    for k in kds.coords:
        vds.coords[k] = kds[k]

Why this DOES work for mosaic'd data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After a mosaic pipeline has run and been serialized to kerchunk, the reference
store contains a **single unified coordinate system**:

    y  → shape (2560,)   ← UNION y, not per-granule
    x  → shape (2048,)   ← UNION x, not per-granule
    v  → shape (20, 2560, 2048)

The ``kds.coords`` read in step 1 are already the union coordinates. When
earthaccess re-attaches them in step 3, the shapes match the data arrays
perfectly. There is no "per-granule coordinate mismatch" problem here.

What about pad / mosaic for pre-virtualized stores?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NOT needed. ``open_virtual`` opens stores that have **already** been virtualized.
Padding and mosaicking happen at *virtualization time* (in
``earthaccess.virtualize()``), not at *open time*. The kerchunk refs already
contain the padded/mosaic'd layout.

Caveat: earthaccess pattern with ``open_virtual(load=True)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When ``load=True``, earthaccess does a kerchunk round-trip:

    vds = vz.open_virtual_mfdataset(...)
    vds.vz.to_kerchunk("refs.json")
    ds = xr.open_dataset("refs.json", engine="kerchunk")

This returns a concrete xarray Dataset (dask-backed). The coordinate handling
is done by xarray's kerchunk engine, which reads the inlined coords from the
refs. This also works because the refs contain the union coordinates.

---

3. Minimal patch for earthaccess
----------------------------------

The smallest change to support TEMPO / ITS-LIVE is in
``earthaccess/virtual/core.py``:

```python
def virtualize(
    granules: list[earthaccess.DataGranule],
    *,
    access: AccessType = "indirect",
    load: bool = False,
    group: str = "/",
    concat_dim: str | None = None,
    mosaic_dims: list[str] | None = None,          # ADD
    pad: str | dict[str, Any] | None = None,       # ADD
    loadable_variables: list[str] | None = None,   # ADD
    preprocess: Callable[[xr.Dataset], xr.Dataset] | None = None,
    data_vars: DataVarsType = "all",
    coords: str = "different",
    compat: CompatType = "no_conflicts",
    combine_attrs: CombineAttrsType = "drop_conflicts",
    parallel: ParallelType = "dask",
    parser: ParserType = "DMRPPParser",
    reference_dir: str | None = None,
    reference_format: ReferenceFormatType = "json",
    **xr_combine_kwargs: Any,
) -> xr.Dataset:
```

And in ``_open_virtual_mfdataset``:

```python
    return vz.open_virtual_mfdataset(
        urls=urls,
        registry=registry,
        parser=parser,
        preprocess=preprocess,
        parallel=parallel,
        combine="nested",
        concat_dim=concat_dim,
        mosaic_dims=mosaic_dims,
        pad=pad,
        loadable_variables=loadable_variables,
        data_vars=data_vars,
        coords=coords,
        compat=compat,
        combine_attrs=combine_attrs,
        **xr_combine_kwargs,
    )
```

That's it — about 10 lines of changes.

---

4. What earthaccess does NOT need to change
-------------------------------------------

| Component | Needs change? | Why |
|---|---|---|
| ``open_virtual(load=False)`` | ❌ No | Opens pre-virtualized stores; coords are already union |
| ``open_virtual(load=True)`` | ❌ No | Kerchunk round-trip works as-is |
| ``_open_virtual_via_virtualizarr`` | ❌ No | Skip-variables pattern works for union coords |
| ``_build_registry_for_url`` | ❌ No | Registry construction is independent |
| Parser selection / fallback | ❌ No | HDFParser already works |
| Kerchunk serialization | ❌ No | ``vds.vz.to_kerchunk`` supports missing chunks |

---

5. Example usage after the patch
----------------------------------

**TEMPO (ragged scanlines, multiple groups):**

```python
import earthaccess

granules = earthaccess.search_data(
    short_name="TEMPO_NO2_L2",
    temporal=("2024-03-28", "2024-03-29"),
)

# Group 1: product variables
product = earthaccess.virtualize(
    granules,
    group="product",
    concat_dim="time",
    loadable_variables=["time", "latitude", "longitude"],
    pad="auto",
)

# Group 2: geolocation variables
geolocation = earthaccess.virtualize(
    granules,
    group="geolocation",
    concat_dim="time",
    loadable_variables=["time", "latitude", "longitude"],
    pad="auto",
)

# Merge groups into one Dataset
import xarray as xr
result = xr.merge([product, geolocation])
result.vz.to_kerchunk("tempo_day.json", format="json")
```

**ITS-LIVE (spatial mosaic):**

```python
granules = earthaccess.search_data(
    short_name="ITS_LIVE",
    bounding_box=(-147.5, 59.0, -143.0, 61.5),
)

vds = earthaccess.virtualize(
    granules,
    concat_dim="time",
    mosaic_dims=["y", "x"],
    loadable_variables=["y", "x", "time"],
    pad="none",
)

vds.vz.to_kerchunk("itslive_mosaic.json", format="json")
```

**Opening the result with earthaccess:**

```python
# Works today — no changes needed
ds = earthaccess.open_virtual("tempo_day.json", load=False)
```

---

6. Backward compatibility
-------------------------

The patch is fully backward compatible:

  - Default ``mosaic_dims=None`` → no mosaicking, existing behaviour
  - Default ``pad=None`` → no padding, existing behaviour
  - Default ``loadable_variables=None`` → VirtualiZarr auto-detects, existing behaviour
  - All existing earthaccess code continues to work unchanged

---

Summary
-------

| Change | File | Lines | Description |
|---|---|---|---|
| Add params | ``earthaccess/virtual/core.py`` | ~6 | Add ``mosaic_dims``, ``pad``, ``loadable_variables`` to ``virtualize()`` |
| Forward params | ``earthaccess/virtual/core.py`` | ~3 | Pass new params through ``_open_virtual_mfdataset`` |
| Docs | ``earthaccess/virtual/core.py`` | ~10 | Update docstring with new params and examples |

Total: ~20 lines of code to unlock TEMPO + ITS-LIVE support in earthaccess.

No changes needed to ``open_virtual``, serialization, or parser selection.
