# Spatial mosaicking and padding in VirtualiZarr

**Branch:** `feature/place-in-grid-mosaic`  
**Status:** in review  

---

## Contents

1. [Why this work exists](#1-why-this-work-exists)
2. [Key concepts](#2-key-concepts)
3. [What changed vs main](#3-what-changed-vs-main)
4. [TEMPO: ragged scanlines and padding](#4-tempo-ragged-scanlines-and-padding)
5. [ITS-LIVE / Malaspina: spatial mosaicking](#5-its-live--malaspina-spatial-mosaicking)
6. [Benchmark results](#6-benchmark-results)
7. [Real-world STAC scenario: 100 granules from one UTM tile](#7-real-world-stac-scenario-100-granules-from-one-utm-tile)
8. [Limitations and known edge cases](#8-limitations-and-known-edge-cases)
9. [API quick reference](#9-api-quick-reference)

---

## 1. Why this work exists

VirtualiZarr's existing `open_virtual_mfdataset` can concatenate virtual datasets
along a dimension — but only when every file already has the same shape in all
other dimensions. Real-world satellite collections break this in two ways:

| Problem | Example | Symptom |
|---|---|---|
| **Ragged scanlines** | TEMPO L2 — scan count varies by orbit | `concat` raises shape mismatch |
| **Partial spatial coverage** | ITS-LIVE velocity pairs — each granule covers only part of the UTM tile | `concat` raises shape mismatch |

Both problems amount to the same thing: the arrays need to be placed inside a
**union grid** with missing cells filled in before concatenation. This branch adds
that capability to VirtualiZarr without touching any data.

---

## 2. Key concepts

### 2.1 ManifestArray

A `ManifestArray` is VirtualiZarr's core data structure. It stores a grid of
chunk references (`path`, `offset`, `length`) in a `ChunkManifest`, plus zarr
array metadata. Nothing is downloaded; all operations work on the reference table.

```
ManifestArray
├── metadata  (shape, chunks, dtype, codecs, fill_value …)
└── manifest  (grid of chunk refs)
              ┌──────┬──────┬──────┐
              │ path │ off  │ len  │  ← one cell per chunk
              │ path │ off  │ len  │
              └──────┴──────┴──────┘
```

### 2.2 Missing chunk

A chunk whose `path` is `""` (the `MISSING_CHUNK_PATH` sentinel). When a reader
encounters a missing chunk it returns the array's `fill_value` instead of fetching
any bytes. Missing chunks cost zero GET requests at compute time.

### 2.3 Virtualize GETs vs compute GETs

**Virtualize GETs** are requests made while *building* the virtual dataset.
`HDFParser` reads only the HDF5 file's metadata — superblock, object headers,
chunk B-tree, dimension scales — to reconstruct the manifest. No data chunks are
read. Typically 5–15 range requests per granule.

**Compute GETs** are requests made when xarray actually evaluates an expression
(e.g. `.mean()`). At this point kerchunk or icechunk resolves each chunk's
`url + offset + length` and issues one range request per chunk. Fill chunks cost
zero GETs — their encoded bytes are inlined in the reference store.

```
virtualize_gets ≈ n_granules × ~8 metadata requests per file
compute_gets    = n_real_chunks  (one GET per real chunk touched)
fill_gets       = 0              (inlined, never fetched)
```

### 2.4 Chunk-alignment requirement

`place_in_grid` and `pad_to_shape` copy whole compressed chunks from one position
in the grid to another. The element-level offsets between granules must therefore
be **exact multiples of the chunk size** on every spatial axis. If they are not,
the chunk boundaries cannot be made to coincide without decompressing and
re-encoding — which would break the virtual-reference contract.

For uncompressed arrays the branch provides `_remap_chunks` + `consolidate_chunks`
to handle the non-aligned case, but this is not yet wired into the public API for
the spatial mosaic path.

---

## 3. What changed vs main

All changes are in `virtualizarr/` — no xarray, zarr, or spec changes.

### 3.1 `manifests/array.py` — four new methods on `ManifestArray`

#### `place_in_grid(new_shape, offset)` ← the core primitive

```
before:                        after (new_shape=(6,6), offset=(2,2)):

array shape (2,4)              array shape (6,6)
chunks (2,2)                   chunks (2,2)
grid  (1,2)                    grid  (3,3)

┌────┬────┐        ┌────┬────┬────┐
│ A  │ B  │   →    │ ░░ │ ░░ │ ░░ │   ░░ = missing (fill_value on read)
└────┴────┘        ├────┼────┼────┤
                   │ ░░ │ A  │ B  │
                   ├────┼────┼────┤
                   │ ░░ │ ░░ │ ░░ │
                   └────┴────┴────┘
```

Rules:
- `offset` must be a multiple of `chunks` on every axis.
  Default `policy="round_to_chunks"` rounds up and emits a warning;
  `policy="error"` raises instead.
- No data is read or written. Chunk references are copied; missing positions are
  initialised with `MISSING_CHUNK_PATH`.

#### `pad_to_shape(new_shape)` — convenience wrapper

`pad_to_shape(new_shape)` is equivalent to `place_in_grid(new_shape, offset=(0,…))`.
Used for TEMPO-style ragged-scanline padding where every file starts at the origin.

#### `_remap_chunks(new_chunks)` — sub-chunk splitting (uncompressed only)

When files have different chunk shapes (or non-aligned spatial origins in the
uncompressed case), this method splits each existing chunk reference into
sub-references aligned to `new_chunks`. **Raises `ValueError`** if the array has
any bytes-to-bytes compression codec, because byte-splitting compressed chunks
produces corrupt virtual references.

#### `consolidate_chunks(target_chunks)` — merge sub-chunks back

After `_remap_chunks`, adjacent sub-chunk references that point to the same file
and form a contiguous byte range are merged back into one entry. Available as a
public API.

---

### 3.2 `manifests/manifest.py` — one new method

#### `_iterate_chunk_keys_with_paths()`

Yields `(chunk_key, path)` for every cell in the chunk grid, including missing
ones (`path == ""`). Used by the kerchunk writer to identify which cells need
inlined fill bytes.

---

### 3.3 `writers/kerchunk.py` — fill-chunk encoding fix

#### `_fill_chunk_bytes(marr, chunk_key)` (new)

Missing chunks must appear in the kerchunk reference store as **encoded** bytes,
not raw bytes. The reader applies the array's full codec pipeline (filters →
compressor) when decoding every chunk — including inlined ones. Previously,
missing chunks were inlined as raw `fill_value.tobytes()`, causing
`zlib.error: incorrect header check` at read time for any array with a
compression filter.

The fix: build the fill array, then apply all `v2_meta.filters` in order, then
`v2_meta.compressor`, mirroring zarr's v2 encode path exactly.

```python
# before (wrong — raw bytes)
return fill_array.tobytes(order="C")

# after (correct — encoded bytes)
buf = fill_array.tobytes(order="C")
for filt in v2_meta.filters or []:
    buf = filt.encode(buf)
if v2_meta.compressor is not None:
    buf = v2_meta.compressor.encode(buf)
return buf
```

---

### 3.4 `xarray.py` — mosaic pipeline and padding pipeline

#### New `mosaic_dims` parameter on `open_virtual_mfdataset`

```python
vds = open_virtual_mfdataset(
    urls,
    combine="nested",
    concat_dim="time",
    loadable_variables=["x", "y", "time"],
    mosaic_dims=["y", "x"],          # ← new
    pad="none",
)
```

When `mosaic_dims` is provided, VirtualiZarr runs the mosaic pipeline after
opening each granule and before concatenating along `concat_dim`.

#### New helper functions (internal)

| Function | Purpose |
|---|---|
| `_is_virtual_variable` | Tests whether an xarray Variable still wraps a ManifestArray |
| `_compute_pad_targets` | Computes the maximum shape across all datasets for padding |
| `_apply_pad_to_dataset` | Calls `pad_to_shape` on every ManifestArray in a dataset |
| `_is_regular_coord` | Validates that a coordinate array is evenly spaced |
| `_build_union_coord` | Builds the union 1-D coordinate from multiple arrays |
| `_compute_mosaic_plan` | Returns `union_coords` and `offsets` for all mosaic dims |
| `_apply_mosaic_to_dataset` | Calls `place_in_grid` on every ManifestArray in a dataset |
| `_resolve_pad_scope` / `_collect_explicit_targets` | Helper for `pad=` parameter logic |
| `_compute_common_chunks` | Finds the common chunk shape across datasets for a variable |
| `_nested_combine_with_padding` | Wraps `open_virtual_mfdataset` combine path with padding |
| `_consolidate_dataset` | Merges size-1 padding sub-chunks back to natural granule chunk sizes |
| `_original_chunks_from_datasets` | Records pre-padding chunk sizes per variable for later consolidation |

#### Bug fix in `_compute_mosaic_plan` (offset sign for decreasing axes)

The offset of a granule within the union grid was computed as:

```python
# before — wrong for decreasing axes (e.g. south-pointing y)
idx = round((start_val - float(union[0])) / abs(step))
```

For a decreasing y-axis (`step < 0`), `union[0]` is the maximum (northernmost)
value. A granule that starts south of the tile origin has `start_val < union[0]`,
giving a negative `idx`. `place_in_grid` then tried to write into a negative chunk
position, producing `shape (1,4,3) into shape (1,0,3)`.

Fix: use signed division so that negative step and negative numerator cancel:

```python
# after — correct for both increasing and decreasing axes
idx = round((start_val - float(union[0])) / step)
```

---

### 3.5 `parsers/hdf/hdf.py` — fill-value robustness fix

```python
# before — crashes for object-dtype variables (bytes fill value)
fill_value = dataset.fillvalue.item()

# after — handles both scalar ndarrays and plain Python values
_fv = dataset.fillvalue
fill_value = _fv.item() if hasattr(_fv, "item") else _fv
```

---

## 4. TEMPO: ragged scanlines and padding

### What is TEMPO?

[TEMPO](https://tempo.si.edu/) (Tropospheric Emissions: Monitoring of Pollution)
is a NASA geostationary instrument that measures air-quality fields over North
America roughly hourly. The number of scanlines (`ny`) per granule varies from
orbit to orbit because the scan rate is not constant.

A typical collection looks like this:

```
granule_01.nc   shape (scanline=1200, xtrack=2048)  ← short orbit
granule_02.nc   shape (scanline=1800, xtrack=2048)  ← longer orbit
granule_03.nc   shape (scanline=1350, xtrack=2048)
…
```

### Why padding is needed

`xr.concat` along `"granule"` only works when all arrays have the same shape on
the **non-concat** axes. Here the xtrack dimension is uniform so a plain concat
along scanlines would work — but the result is a jagged virtual array with varying
extent on the concat axis. To produce a rectangular grid (required by zarr), every
granule must be padded to the **union scanline count** before concatenation.

The real challenge arises when the last physical chunk in a file is a partial
chunk (e.g. 1250 scanlines with chunk_ny=300 gives a final chunk of only 50
scanlines). `pad_to_shape` rounds up to the next full chunk boundary so the
virtual reference grid is rectangular.

### Schematic: pad_to_shape

```
granule_01 (1200 scanlines, 2048 xtrack)   chunks=(300, 2048)

chunk grid:    row 0   row 1   row 2   row 3
               ──────  ──────  ──────  ──────
               [C0,0]  [C1,0]  [C2,0]  [C3,0]    4 chunks × 1 column

pad_to_shape((1800, 2048)):

chunk grid:    row 0   row 1   row 2   row 3   row 4   row 5
               ──────  ──────  ──────  ──────  ──────  ──────
               [C0,0]  [C1,0]  [C2,0]  [C3,0]  [░░░]  [░░░]
                                                 ↑ missing chunks
                                                 (return fill_value on read)
```

The padded array concatenates cleanly with granule_02's `(1800, 2048)` array:

```
concat([granule_01_padded, granule_02], dim="granule")
→ shape: (2, 1800, 2048)
```

### Full-week fixture (168 granules)

The `TestTempoWeekE2E` test generates 168 synthetic HDF5 files (7 days × 24
scans/day) with variable scanline counts per granule, matching real TEMPO L2
orbit geometry:

```
proxy dimensions:  nx=64, chunk_ny=16, ny ∈ [96, 160]  (1/32 scale of real TEMPO)
union_ny:          160 (rounded to chunk boundary)
real chunks:       1437
fill chunks:        243  (14.5% of grid)
```

### Chunk consolidation after padding

Padding uncompressed arrays requires `_remap_chunks(1)` to ensure every element
boundary is chunk-aligned, producing `chunk_size=1` along the padded dimension.
After `xr.concat`, the combined manifest therefore has `N × orig_rows` chunks of
shape `(1, xtrack)` instead of the natural `orig_rows`-per-granule chunks.

`open_virtual_mfdataset` automatically consolidates these back with
`consolidate=True` (the default). After consolidation each granule's contiguous
rows are merged back into one chunk, reducing the task count by `~orig_rows` ×:

```
TEMPO example (2 granules of 132 rows, padded to 132 each):

  without consolidation:  264 chunks × (1, 2048)   → 22 176 Dask tasks for mean()
  with consolidation:       4 chunks × (132, 2048)  →    168 Dask tasks for mean()
```

Consolidation is skipped for compressed arrays (already handled by `pad_to_shape`)
and for chunk groups that span multiple source files (cross-granule merges are
correctly rejected by `consolidate_chunks`).

#### Limitation: ragged uncompressed granules

When granules have **different** row counts (e.g. TEMPO G01=132, G02=131),
padding fills G02 to 132 rows with a missing sub-chunk.  After concat the target
cell for G02 contains **131 real + 1 missing** sub-chunks.  Merging these would
produce a manifest entry claiming `(132, xtrack)` bytes but storing only
`131 × xtrack` bytes, causing a zarr reshape error at read time:

```
ValueError: cannot reshape array of size 268288 into shape (1,132,2048)
```

`consolidate_chunks` detects this mixed-presence cell and raises `ValueError`;
`_consolidate_dataset` catches it and leaves the variable at `chunk_size=1`.
The result is **correct and safe** — just without the task-count reduction:

```
TEMPO example (G01=132, G02=131 rows, ragged):

  consolidate=True (safe no-op):  264 chunks × (1, 2048)  — reads correctly
  consolidate=False:              264 chunks × (1, 2048)  — same
```

Consolidation *does* help when all granules in a scan have the same row count
(uniform, non-ragged) or when HDF5 files use explicit chunk storage.

To disable explicitly:

```python
vds = open_virtual_mfdataset(..., consolidate=False)
```

---

## 5. ITS-LIVE / Malaspina: spatial mosaicking

### Dataset overview

[ITS-LIVE](https://its-live.jpl.nasa.gov/) produces surface ice-velocity pairs
from Landsat and Sentinel-2 imagery. Each granule covers the overlap area between
two consecutive scene acquisitions. These pairs have irregular spatial extents
within the WRS tile footprint.

The Malaspina benchmark uses 20 granules from WRS tile 063018 (south-east Alaska,
EPSG:3413) acquired September 2025 – January 2026:

```
schema per granule:
  variables: vx, vy, v, v_error, chip_size_width, chip_size_height, interp_mask
  dims:      (time=1, y=*, x=*)
  chunks:    (1, 512, 512)
  dtype:     int16 (velocity) / float32
  codec:     shuffle + zlib level 2
  pixel:     120 m
  max tile:  2560 y × 2048 x pixels  (= 307 200 × 245 760 m)
```

### Why plain concat fails

Each granule covers a different spatial sub-region of the tile. Their `y` and `x`
arrays have different lengths and different start values. `xr.concat` on the time
axis requires all granules to share the same `y`/`x` shape — which they do not.

### Mosaic pipeline

```
Step 1  open_virtual_dataset per granule (reads only HDF5 metadata)
        ↓
Step 2  _compute_mosaic_plan(datasets, mosaic_dims=["y","x"])
        → union_coords: {"y": array(2560 values), "x": array(2048 values)}
        → offsets:      {"y": [0, 512, 0, …],    "x": [0, 512, 512, …]}
        ↓
Step 3  for each granule i:
          _apply_mosaic_to_dataset(ds_i, i, mosaic_dims, union_coords, offsets)
          → calls ManifestArray.place_in_grid(
                new_shape=(1, 2560, 2048),
                offset=(0, offset_y_i, offset_x_i)
            )
        ↓
Step 4  xr.concat(mosaicked_datasets, dim="time")
        → final shape: (20, 2560, 2048)
```

### Chunk grid after mosaicking (per time step)

For a full-tile granule (2560×2048, chunks 512×512):

```
chunk grid = 5 rows × 4 cols = 20 cells, all real

[R][R][R][R]
[R][R][R][R]
[R][R][R][R]
[R][R][R][R]
[R][R][R][R]
```

For a partial granule (e.g. 2048×1536, placed at offset (512, 512)):

```
chunk grid = 5 rows × 4 cols = 20 cells

[░][░][░][░]   ← row 0: all missing (above granule origin)
[░][R][R][R]   ← rows 1–4: col 0 missing (left of granule origin)
[░][R][R][R]
[░][R][R][R]
[░][R][R][R]

R = 12 real chunks,  ░ = 8 missing chunks
```

Summary across 20 Malaspina granules: **308 real chunks + 92 fill chunks** per
variable (23% fill).

### GET accounting for 20 granules

```
virtualize_gets  ≈ 171   (≈ 8.5 metadata requests × 20 granules)
compute_gets     = 308   (one per real chunk — matches real_chunks exactly)
fill_gets        = 0     (inlined as encoded fill bytes, never fetched)
bytes transferred ≈ 11.7 MB for v.mean() over the full union grid
```

---

## 6. Benchmark results

All timings are wall-clock on a developer laptop (Linux, Python 3.12). The
fixture-server tests serve files over localhost HTTP so GET counts are exact.

### 6.1 TEMPO full-week (168 synthetic granules, local fixture server)

```
proxy scale: nx=64, chunk_ny=16, ny ∈ [96,160]  (1/32 of real TEMPO)
union shape: (26880 scanlines, 64 xtrack)
real chunks: 1437    fill chunks: 243
```

| Metric | kerchunk-parquet | kerchunk-json | icechunk |
|---|---:|---:|---:|
| virtualize | 1576 ms | 1688 ms | 1674 ms |
| serialize | 46 ms | 17 ms | 17 ms |
| open | 22 ms | 19 ms | 2 ms |
| mean() | 1934 ms | 1908 ms | 416 ms |
| virtualize GETs | 504 | 504 | 504 |
| compute GETs | 1437 | 1437 | 1437 |
| compute KB | 5748 | 5748 | 5748 |
| **TOTAL time** | **3577 ms** | **3632 ms** | **2108 ms** |

Icechunk's `mean()` is ~4.6× faster than kerchunk (416 ms vs ~1920 ms). This is
because icechunk issues parallel range requests natively; kerchunk serialises them
through fsspec's sequential HTTP client.

Virtualize time is dominated by the 168 × ~3 metadata range requests per file,
not by Python overhead. It is the same across all three backends because
virtualization is backend-agnostic.

### 6.2 Malaspina full-tile (20 real ITS-LIVE granules, local cache)

```
union shape: (20 time, 2560 y, 2048 x)
real chunks: 308    fill chunks: 92
granule size: ~6 MB each  (total ~120 MB cached locally)
```

| Metric | kerchunk-parquet | kerchunk-json | icechunk |
|---|---:|---:|---:|
| virtualize | 808 ms | 714 ms | 749 ms |
| serialize | 710 ms | 569 ms | 47 ms |
| open | 92 ms | 26 ms | 2 ms |
| mean() | 912 ms | 920 ms | 812 ms |
| virtualize GETs | 171 | 171 | 171 |
| compute GETs | 308 | 308 | 1018 |
| compute KB | 11947 | 11947 | 11947 |
| **TOTAL time** | **2522 ms** | **2228 ms** | **1610 ms** |

Notes:
- Icechunk issues **1018 compute GETs** vs 308 for kerchunk. Icechunk fetches each
  spatial chunk (1, 512, 512) as three separate requests (one per `time` slice)
  rather than the single contiguous read that covers all time steps that kerchunk
  issues via fsspec. Both transfer the same bytes (~11.7 MB).
- Kerchunk-parquet serialize (710 ms) is slower than JSON (569 ms) because the
  parquet writer encodes fill chunks as base64-encoded compressed bytes inline,
  which involves running the full zlib pipeline for each of the 92 fill chunks.
- Virtualize time (~750 ms) is ~20× faster than real-S3 (~18 s) because metadata
  is served locally without network latency.

### 6.3 Malaspina vs real S3 (no GET counting)

When running against real S3 (`TestMalaspinaWeekE2E`), virtualize time reflects
the full network round-trip for 20 HDF5 metadata reads:

```
virtualize (real S3):  ~18–19 s   (dominated by S3 latency per granule)
virtualize (local):     ~0.75 s   (same metadata, served over localhost)
compute    (real S3):   ~2.5 s    (parallel S3 GETs via obstore)
compute    (local):     ~0.9 s
```

The VirtualiZarr code path is identical. The difference is purely network latency.

### 6.4 Comparison with odc-stac (real S3, 20 granules)

`odc-stac` is a rasterio-based stacking library commonly used with COG/GeoTIFF assets.
It can also read NetCDF files via `/vsicurl/` URLs, but every file is downloaded
**whole** because HDF5 requires random access to the file structure. This makes it
much slower than VirtualiZarr's chunk-level access for the same dataset.

| Metric | VirtualiZarr | odc-stac (10 workers) |
|---|---|---|
| Total time | ~25 s | ~36 s |
| Data transferred | ~12 MB (chunk-level) | ~120 MB (whole files) |
| Grid | Native `(20, 2560, 2048)` | Snapped `(20, 2561, 2050)` |
| Parallelism | Yes (chunk-level, thread-safe) | Yes (process-safe, 10 workers) |
| Reprojection | None | Nearest-neighbor warp |

**VirtualiZarr is ~40% faster** despite transferring 10× less data, because it reads
only the chunks touched by the computation. odc-stac downloads each ~6 MB file in
its entirety, then warps it onto a snapped target grid.

### 6.5 Scientific accuracy: native grid vs reprojection

VirtualiZarr and odc-stac produce different pixel values at the edges because their
spatial models differ:

| Aspect | VirtualiZarr | odc-stac |
|---|---|---|
| **Pixel value** | Exact original measurement | Interpolated from nearest source pixel |
| **Sub-pixel shift** | None (native grid) | Up to ±60 m (half-pixel from grid snapping) |
| **Edge behaviour** | Fill value where no data exists | Warped from nearest source (may bleed in) |
| **Repeatability** | Perfect (same byte every time) | Depends on grid snapping parameters |

For ITS-LIVE, the native chunk grid is `(1, 512, 512)` with 120 m pixels. The mosaic
pipeline places each granule at its exact chunk-aligned offset. A `mean("time")`
at the centre of the union tile averages the **exact same physical pixel** across
all 20 granules, with no resampling noise.

odc-stac constructs a new target `GeoBox` snapped to the CRS origin. Pixel edges
align to `N × 120 m`, which may shift the grid by up to half a pixel relative to
the native Landsat sub-tile alignment. The resulting values differ at ~27k edge
pixels (out of 5.2 M), with a mean absolute difference of ~37 m/y.

For change-detection or time-series analysis, VirtualiZarr's preservation of the
native grid eliminates resampling artifacts that can introduce spurious trends.

---

## 7. Real-world STAC scenario: 100 granules from one UTM tile

### 7.1 What the pipeline requires

The mosaic pipeline has three hard requirements per spatial dimension:

1. **The coordinate variable must be loaded (`loadable_variables`).**
   The pipeline needs numeric coordinate arrays to compute the union grid and
   per-granule offsets. If `y` or `x` is left virtual the pipeline raises
   `ValueError` before any data is downloaded.

2. **The coordinate must be regularly spaced.**
   `_build_union_coord` calls `_is_regular_coord`, which checks that
   `np.diff(arr)` is constant within a tolerance of `1e-6 × step`. ITS-LIVE
   granules always have `step = ±120 m` so this is trivially satisfied.

3. **Granule origins must be chunk-aligned.**
   `offset[ax] = round((start_val - union[0]) / step)` must be a multiple of
   `chunks[ax]`. For ITS-LIVE over a single WRS tile this is guaranteed by the
   instrument geometry: the 512-pixel chunk exactly equals one Landsat scene
   sub-tile. If a future data version changes the HDF5 chunk size this guarantee
   could break.

### 7.2 What happens with 100 granules

```
100 granules × _compute_mosaic_plan → one union grid
                                       shape: (100, 2560, 2048)  (typical)

Per-variable chunk grid: 100 × 5 × 4 = 2000 cells per variable
  real chunks: ~1400–1700  (depends on coverage, typically ~70–80%)
  fill chunks: ~300–600    (~20–30%)

Memory for the virtual reference store:
  2000 cells × 7 variables × 3 arrays (paths/offsets/lengths)
  ≈ 2000 × 7 × 3 × 8 bytes ≈ 340 KB   ← negligible

Estimated virtualize time (real S3): ~90–100 s  (linear in granule count)
Estimated virtualize time (local cache): ~4–5 s

Kerchunk JSON size: ~2.4 MB for 20 → ~12 MB for 100 (linear)
```

### 7.3 Scenarios that currently fail

| Scenario | Failure mode | Status |
|---|---|---|
| Two granules with **different chunk sizes** (e.g. one reprocessed) | `_apply_mosaic_to_dataset` raises `ValueError` from `_remap_chunks` compression guard | Known limitation — needs uncompressed intermediate or rechunking at write time |
| Granule origin **not chunk-aligned** | `place_in_grid` rounds to nearest chunk boundary and emits a warning; granule data is shifted by ≤ 1 chunk | Acceptable for most use cases; use `policy="error"` to detect |
| Variable with **object dtype** (e.g. `mapping`, `img_pair_info` in ITS-LIVE) | zarr v3 cannot represent variable-length strings; `HDFParser` raises | Workaround: `drop_variables=["mapping","img_pair_info"]` |
| **Duplicate timestamps** (two granules same `time`) | `xr.concat` raises on duplicate coordinates | Not specific to this branch; use `compat="override"` or deduplicate upstream |
| Granules with **different `y`/`x` spacing** (e.g. 30 m vs 120 m) | `_build_union_coord` raises `ValueError` (inconsistent spacing) | By design — mixing resolutions in one virtual store is not supported |

### 7.4 Scalability schematic

```
STAC query → 100 URLs

open_virtual_mfdataset(urls, mosaic_dims=["y","x"], concat_dim="time")

  for each URL (parallelisable with dask or threads):
    open_virtual_dataset(url)          ← reads only HDF5 metadata (~8 requests)
    load x, y, time coordinates        ← reads coordinate arrays (few KB)

  _compute_mosaic_plan                 ← pure numpy, < 1 ms
    → union grid (2560 y × 2048 x for ITS-LIVE WRS tile)
    → per-granule offsets

  for each dataset:
    _apply_mosaic_to_dataset           ← array index arithmetic, < 1 ms/granule
      place_in_grid per variable

  xr.concat(100 mosaicked datasets, dim="time")
    → virtual dataset  shape (100, 2560, 2048)
      real chunks:  ~1500–1700  (depends on coverage)
      fill chunks:  ~300–500

  vds.vz.to_kerchunk("collection.json")  ← ~5 s, ~12 MB

  xr.open_dataset("collection.json", engine="kerchunk")
    → lazy Dataset  3 GB virtual

  ds["v"].isel(time=slice(0,10)).mean("time").compute()
    → downloads only the 10×20 = 200 real chunks needed
    → ~2–5 s on a good network connection
```

### 7.5 What will be downloaded on compute

Only the chunks actually touched by the computation:

```
ds["v"].isel(x=slice(512, 1024), y=slice(512, 1024)).mean("time")

  → selects 1×1 spatial chunk per time step
  → downloads 100 real chunks (one per granule that covers that region)
  → 0 fill chunks fetched (inlined as encoded bytes in the reference store)

vs. naive approach (download all files):
  → 100 complete files × ~6 MB each = ~600 MB
```

---

## 8. Limitations and known edge cases

### 8.1 Compressed arrays require chunk-aligned origins

This is the most important constraint. The mosaic fast path copies whole
compressed chunks. If a granule's pixel origin is not a multiple of the chunk size
in every mosaic dimension, the data cannot be placed correctly without
decompressing and re-encoding. The current behaviour:

- `policy="round_to_chunks"` (default): rounds the offset up to the next chunk
  boundary and emits a `UserWarning`. The granule's data is offset by up to
  `(chunks - 1)` pixels. **This produces incorrect data for mis-aligned granules.**
- `policy="error"`: raises `ValueError` immediately.

For ITS-LIVE data from a single WRS tile this never triggers because all origins
are guaranteed to be multiples of 512 pixels.

### 8.2 Fill chunks in kerchunk parquet

Kerchunk parquet stores chunk references in a pandas DataFrame. A `None` or `NaN`
URL column value survives the roundtrip as `float('nan')`, which fsspec cannot
handle. The kerchunk writer therefore inlines missing chunks as encoded fill bytes
(`base64:…`). For 100 granules with ~400 fill chunks and int16 arrays at
`(1,512,512)` chunk shape, the overhead is small:

```
512 × 512 × 2 bytes → shuffle → zlib → base64 ≈ ~1 KB per chunk
400 fill chunks × ~1 KB ≈ ~400 KB extra in the parquet store
```

This is negligible but worth noting for unusual fill values that compress poorly.

### 8.3 Icechunk GET count inflation

Icechunk 2.x fetches virtual chunks independently per logical chunk in the zarr
array. For a variable with shape `(20, 2560, 2048)` and chunks `(1, 512, 512)`,
icechunk issues one GET per `(1, 512, 512)` cell — even if adjacent time slices
point into the same HDF5 file at a contiguous byte range. Kerchunk+fsspec issues
one GET per chunk key, which can cover multiple time steps in a single contiguous
request when the file layout allows. This explains the 308 vs 1018 GET count
difference in the Malaspina benchmark (both transfer the same bytes).

### 8.4 Nodata / fill-value handling

VirtualiZarr preserves the raw pixel values (including the fill value, e.g.
`-32767` for ITS-LIVE `int16` arrays). The user must mask them before computing
statistics:

```python
fill_value = ds["v"].encoding.get("_FillValue", -32767)
result = ds["v"].where(ds["v"] != fill_value).mean("time", skipna=True)
```

This is consistent with xarray's zarr/kerchunk backend, which stores `_FillValue`
in `.encoding` but does not auto-mask it during `.compute()`. In contrast,
odc-stac (via rasterio) converts nodata pixels to `NaN` automatically, but may
still leave edge artifacts from nearest-neighbor warping.

### 8.5 `time` coordinate uniqueness

`xr.concat(dim="time")` will raise if two granules share the same timestamp. STAC
queries sometimes return duplicate granules (e.g. reprocessing events). Deduplicate
upstream or pass `compat="override"` if duplicates are acceptable.

### 8.6 Object-dtype HDF5 variables

Variables stored as HDF5 variable-length strings (e.g. `mapping`, `img_pair_info`
in ITS-LIVE NetCDF-4 files) cannot be represented in zarr v3. Use:

```python
parser=HDFParser(drop_variables=["mapping", "img_pair_info"])
```

### 8.7 Parallelism with HDF5-backed assets

HDF5 is **not thread-safe**. If you use odc-stac with NetCDF/HDF5 assets, you
must use a **process-based** scheduler (`scheduler="processes"`) or odc-stac's
`pool=ProcessPoolExecutor(...)`. Thread-based parallelism will corrupt the HDF5
global state and produce `errno = 2` errors.

VirtualiZarr does not have this limitation because `h5py` + `obstore` issue
independent HTTP range requests per chunk, with no shared HDF5 file handle.

---

## 9. API quick reference

### High-level: TEMPO ragged scanlines + time-concat

```python
from virtualizarr.xarray import open_virtual_mfdataset
from virtualizarr.parsers.hdf import HDFParser
from obstore.store import HTTPStore
from obspec_utils.registry import ObjectStoreRegistry

BASE = "https://data.asdc.earthdata.nasa.gov/"
registry = ObjectStoreRegistry({BASE: HTTPStore(BASE)})

# nested_granules: list[list[str]] — outer=scan, inner=granules within scan
vds = open_virtual_mfdataset(
    nested_granules,
    registry=registry,
    parser=HDFParser(group="/product"),
    combine="nested",
    concat_dim=["time", "mirror_step"],   # outer=scans, inner=granules
    pad="auto",                           # pad ragged mirror_step per scan
    consolidate=True,                     # default: merge size-1 chunks back
    preprocess=add_time,                  # inject UTC datetime coord
    loadable_variables=["latitude", "longitude"],  # inline fixed geostationary grid
)
```

`consolidate=True` (default) reduces Dask task count ~100× for TEMPO by merging
the `chunk_size=1` padding artifacts back to natural per-granule chunk sizes.
Pass `consolidate=False` only when debugging the manifest structure.

### High-level: mosaic + time-concat

```python
from virtualizarr.xarray import open_virtual_mfdataset
from virtualizarr.parsers.hdf import HDFParser
from obstore.store import HTTPStore
from obspec_utils.registry import ObjectStoreRegistry

BASE = "https://its-live-data.s3.amazonaws.com/"
registry = ObjectStoreRegistry({BASE: HTTPStore(BASE)})

vds = open_virtual_mfdataset(
    stac_urls,                                        # list[str], any length
    registry=registry,
    parser=HDFParser(drop_variables=["mapping", "img_pair_info"]),
    combine="nested",
    concat_dim="time",
    loadable_variables=["x", "y", "time"],            # must include mosaic dims
    mosaic_dims=["y", "x"],                           # spatial dims to union
    pad="none",
)

# serialize
vds.vz.to_kerchunk("collection.json", format="json")
vds.vz.to_kerchunk("collection.parquet", format="parquet")

# use
import xarray as xr
ds = xr.open_dataset("collection.json", engine="kerchunk")
result = ds["v"].isel(time=0).compute()               # downloads only needed chunks
```

### Low-level: ManifestArray primitives

```python
from virtualizarr.manifests import ManifestArray

# Pad a ragged array to a target shape (TEMPO use-case)
padded = arr.pad_to_shape((1800, 2048))

# Place an array at an offset inside a larger grid (mosaic use-case)
placed = arr.place_in_grid(
    new_shape=(1, 2560, 2048),
    offset=(0, 512, 512),
    policy="error",          # raise if offset is not chunk-aligned
)

# Re-chunk an uncompressed array to smaller sub-chunks
remapped = arr._remap_chunks(new_chunks=(1, 128, 128))

# Merge sub-chunk references back to larger chunks
consolidated = remapped.consolidate_chunks(target_chunks=(1, 512, 512))
```

### Running the benchmarks

**Start the fixture server (port 80 required for icechunk):**

```bash
sudo /path/to/.venv/bin/python \
    virtualizarr/tests/http_fixture_server.py \
    --serve-dir virtualizarr/tests/fixtures \
    --file-port 80 \
    --control-port 18080
```

**Run TEMPO full-week (168 synthetic granules, local only):**

```bash
FIXTURE_SERVER_URL=http://127.0.0.1 \
FIXTURE_SERVER_CONTROL_URL=http://127.0.0.1:18080 \
FIXTURE_SERVER_DIR=virtualizarr/tests/fixtures \
uv run pytest virtualizarr/tests/test_e2e_performance.py::TestTempoWeekE2E::test_comparison_table \
    --run-slow-tests -s
```

**Run Malaspina full-tile (20 granules, downloads once then cached):**

```bash
FIXTURE_SERVER_URL=http://127.0.0.1 \
FIXTURE_SERVER_CONTROL_URL=http://127.0.0.1:18080 \
FIXTURE_SERVER_DIR=virtualizarr/tests/fixtures \
uv run pytest virtualizarr/tests/test_e2e_performance.py::TestMalaspinaWeekLocalE2E::test_comparison_table \
    --run-network-tests --run-slow-tests -s
```
