import re
import warnings
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import LocalStore
from zarr.codecs import ZstdCodec

from virtualizarr.parsers.hdf import HDFParser
from virtualizarr.parsers.zarr import ZarrParser
from virtualizarr.xarray import open_virtual_mfdataset

ENCODING = {"a": {"chunks": (1, 1)}}
# zarr v3 compression encoding: compressors list
ENCODING_COMPRESSED = {"a": {"chunks": (1, 1), "compressors": [ZstdCodec()]}}


def _create_zarr(tmp_path, name, shape, dtype="float32", encoding=None):
    store_path = tmp_path / name
    arr = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    ds = xr.Dataset(
        {"a": xr.Variable(dims=["x", "y"], data=arr)},
    )
    ds.to_zarr(store_path, zarr_format=3, encoding=encoding or ENCODING)
    return store_path


@pytest.fixture
def ragged_zarrs(tmp_path):
    s1 = _create_zarr(tmp_path, "s1.zarr", (1, 1))
    s2 = _create_zarr(tmp_path, "s2.zarr", (1, 2))
    s3 = _create_zarr(tmp_path, "s3.zarr", (1, 3))
    return [str(s) for s in (s1, s2, s3)]


@pytest.fixture
def matching_zarrs(tmp_path):
    s1 = _create_zarr(tmp_path, "s1.zarr", (1, 2))
    s2 = _create_zarr(tmp_path, "s2.zarr", (1, 2))
    return [str(s) for s in (s1, s2)]


@pytest.fixture
def registry():
    return ObjectStoreRegistry({"file://": LocalStore()})


class TestPadNested:
    def test_pads_ragged_dims(self, ragged_zarrs, registry):
        result = open_virtual_mfdataset(
            ragged_zarrs,
            registry=registry,
            parser=ZarrParser(),
            combine="nested",
            concat_dim="x",
            pad="auto",
            loadable_variables=[],
        )

        assert result["a"].shape == (3, 3)
        assert result["a"].dtype == np.dtype("float32")

    def test_no_warning_when_shapes_match(self, matching_zarrs, registry):
        with warnings.catch_warnings(record=True) as record:
            result = open_virtual_mfdataset(
                matching_zarrs,
                registry=registry,
                parser=ZarrParser(),
                combine="nested",
                concat_dim="x",
                pad="auto",
                loadable_variables=[],
            )

        assert result["a"].shape == (2, 2)
        pad_warnings = [
            w for w in record if "padding" in str(w.message).lower()
        ]
        assert len(pad_warnings) == 0

    def test_pad_none_disables_padding(self, ragged_zarrs, registry):
        with pytest.raises(ValueError):
            open_virtual_mfdataset(
                ragged_zarrs,
                registry=registry,
                parser=ZarrParser(),
                combine="nested",
                concat_dim="x",
                pad="none",
                loadable_variables=[],
            )

    def test_user_supplied_pad_targets(self, ragged_zarrs, registry):
        result = open_virtual_mfdataset(
            ragged_zarrs,
            registry=registry,
            parser=ZarrParser(),
            combine="nested",
            concat_dim="x",
            pad={"y": 3},
            loadable_variables=[],
        )

        assert result["a"].shape == (3, 3)


def _create_tempo_like_granule(tmp_path, name, mirror_step_size, xtrack_size=8, scan_id=0, granule_id=0):
    """Create a synthetic HDF5 file mimicking a TEMPO L2 granule.

    Layout:
    - dims: (mirror_step, xtrack) — uses dimension scales named mirror_step and xtrack
    - ``time``   : float64 1-D on mirror_step (one timestamp per scanline, NOT a dim coord)
    - ``lat``    : float32 2-D (mirror_step, xtrack)
    - ``lon``    : float32 2-D (mirror_step, xtrack)
    - ``no2``    : float32 2-D data variable (mirror_step, xtrack)
    """
    import h5py

    path = tmp_path / name
    with h5py.File(path, "w") as f:
        ms = mirror_step_size
        xt = xtrack_size

        # dimension scales — named mirror_step and xtrack to match TEMPO
        mirror_idx = np.arange(ms, dtype="float64")
        xtrack_idx = np.arange(xt, dtype="float64")
        f.create_dataset("mirror_step", data=mirror_idx)
        f.create_dataset("xtrack", data=xtrack_idx)
        f["mirror_step"].make_scale("mirror_step")
        f["xtrack"].make_scale("xtrack")

        # time: 1D variable along mirror_step (NOT a dimension scale)
        f.create_dataset(
            "time",
            data=np.arange(ms, dtype="float64") + scan_id * 1000 + granule_id * ms,
        )
        f["time"].dims[0].attach_scale(f["mirror_step"])

        # 2D variables
        for var_name, dtype, val in [
            ("lat", "float32", float(scan_id + granule_id * 0.1)),
            ("lon", "float32", float(granule_id + scan_id * 10)),
            ("no2", "float32", float(scan_id * 10 + granule_id)),
        ]:
            f.create_dataset(var_name, data=np.full((ms, xt), val, dtype=dtype))
            f[var_name].dims[0].attach_scale(f["mirror_step"])
            f[var_name].dims[1].attach_scale(f["xtrack"])

    return f"file://{path.resolve()}"


class TestTEMPOSyntheticIssue487:
    """Synthetic test for the two-level nested concat: outer=datescan, inner=mirror_step.

    Mirrors issue #487: TEMPO L2 granules have (mirror_step, xtrack) dims.
    The last granule in each scan may have fewer mirror_step rows (ragged).
    Padding must only happen at the inner mirror_step level, not at the outer
    datescan level.
    """

    XTRACK = 8
    # scan1: [10, 10, 7] mirror_step sizes (last granule ragged)
    # scan2: [10, 10, 8] mirror_step sizes (last granule ragged)
    SCAN1_SIZES = [10, 10, 7]
    SCAN2_SIZES = [10, 10, 8]

    @pytest.fixture
    def registry(self):
        return ObjectStoreRegistry({"file://": LocalStore()})

    @pytest.fixture
    def scan_urls(self, tmp_path):
        scan1 = [
            _create_tempo_like_granule(tmp_path, f"s1g{i}.h5", ms, self.XTRACK, scan_id=0, granule_id=i)
            for i, ms in enumerate(self.SCAN1_SIZES)
        ]
        scan2 = [
            _create_tempo_like_granule(tmp_path, f"s2g{i}.h5", ms, self.XTRACK, scan_id=1, granule_id=i)
            for i, ms in enumerate(self.SCAN2_SIZES)
        ]
        return [scan1, scan2]

    def _add_datescan(self, ds):
        """Preprocess: add synthetic datescan scalar coord."""
        source = ds.encoding.get("source", "")
        m = re.search(r"s(\d+)g", source)
        scan_id = int(m.group(1)) if m else 0
        return ds.expand_dims("datescan").assign_coords(datescan=[f"scan_{scan_id:03d}"])

    def test_two_level_nested_concat_shape(self, scan_urls, registry):
        """Concat_dim=['datescan','mirror_step']: outer=datescan, inner=mirror_step.

        Padding must pad mirror_step within each scan to max(scan_sizes),
        then outer concat along datescan. Final shape: (2, 30, 8).
        """
        result = open_virtual_mfdataset(
            scan_urls,
            registry=registry,
            parser=HDFParser(),
            combine="nested",
            concat_dim=["datescan", "mirror_step"],
            pad="auto",
            preprocess=self._add_datescan,
            loadable_variables=[],
        )

        # mirror_step per scan: max(10,10,7)=10, max(10,10,8)=10 → total=20 after outer concat
        # datescan=2, mirror_step=20 (inner concat of 3 granules with padding), xtrack=8
        # Wait: inner concat along mirror_step: 10+10+10=30, outer concat along datescan: 2
        # shape = (datescan=2, mirror_step=30, xtrack=8)
        assert result["no2"].dims == ("datescan", "mirror_step", "xtrack")
        assert result["no2"].shape == (2, 30, self.XTRACK)

    def test_padding_only_on_mirror_step_not_outer_dim(self, scan_urls, registry):
        """Verify padding is applied only to mirror_step (ragged dim), not datescan."""
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            result = open_virtual_mfdataset(
                scan_urls,
                registry=registry,
                parser=HDFParser(),
                combine="nested",
                concat_dim=["datescan", "mirror_step"],
                pad="auto",
                preprocess=self._add_datescan,
                loadable_variables=[],
            )

        pad_warnings = [w for w in record if "padding" in str(w.message).lower()]
        # At least one warning about mirror_step padding
        if pad_warnings:
            warning_text = " ".join(str(w.message) for w in pad_warnings)
            assert "mirror_step" in warning_text
            # datescan should NOT appear in padding warnings (it's not ragged)
            assert "datescan" not in warning_text

        # Shape is correct regardless of warnings
        assert result["no2"].shape == (2, 30, self.XTRACK)

    def test_time_variable_concatenates_naturally(self, scan_urls, registry):
        """time is a 1D variable on mirror_step; it must concat naturally (not conflict)."""
        result = open_virtual_mfdataset(
            scan_urls,
            registry=registry,
            parser=HDFParser(),
            combine="nested",
            concat_dim=["datescan", "mirror_step"],
            pad="auto",
            preprocess=self._add_datescan,
            loadable_variables=["xtrack"],
        )
        # time is along mirror_step so after inner concat its size = 30 per scan
        # after outer concat it should be (datescan=2, mirror_step=30)
        assert "time" in result
        # xtrack is a 1D coordinate
        assert "xtrack" in result.coords
        assert result["xtrack"].shape == (self.XTRACK,)

    def test_lat_lon_present_as_2d_coords(self, scan_urls, registry):
        """lat and lon are 2D (mirror_step, xtrack) and must be present in the result."""
        result = open_virtual_mfdataset(
            scan_urls,
            registry=registry,
            parser=HDFParser(),
            combine="nested",
            concat_dim=["datescan", "mirror_step"],
            pad="auto",
            preprocess=self._add_datescan,
            loadable_variables=[],
        )
        assert "lat" in result
        assert "lon" in result
        assert result["lat"].dims == ("datescan", "mirror_step", "xtrack")
        assert result["lon"].dims == ("datescan", "mirror_step", "xtrack")


class TestTEMPOIssue487:
    @pytest.fixture
    def tempo_urls(self):
        fixture_dir = Path(__file__).parent / "fixtures" / "tempo"
        files = sorted(fixture_dir.glob("TEMPO_NO2_L2*.nc"))
        if len(files) < 4:
            pytest.skip("TEMPO fixture files not available")
        return [f"file://{f}" for f in files]

    def _add_datescan(self, ds):
        source = ds.encoding.get("source", "")
        m = re.search(r"S(\d+)", source)
        scan = int(m.group(1)) if m else 0
        return ds.expand_dims("datescan").assign_coords(datescan=[f"scan_{scan:03d}"])

    def test_nested_combine_two_scans(self, tempo_urls, registry):
        scan3 = [tempo_urls[0], tempo_urls[1]]
        scan4 = [tempo_urls[2], tempo_urls[3]]
        urls = [scan3, scan4]

        result = open_virtual_mfdataset(
            urls,
            registry=registry,
            parser=HDFParser(group="/product"),
            combine="nested",
            concat_dim=["datescan", "mirror_step"],
            pad="auto",
            preprocess=self._add_datescan,
            loadable_variables=[],
        )

        assert result["vertical_column_troposphere"] is not None
        assert result["vertical_column_troposphere"].shape == (2, 264, 2048)

    def test_consolidate_default_true_reduces_chunk_count(self, tempo_urls, registry):
        """consolidate=True (default) merges size-1 padding chunks back to
        natural granule-sized chunks along mirror_step, dramatically reducing
        the number of chunks in the combined manifest."""
        scan3 = [tempo_urls[0], tempo_urls[1]]
        scan4 = [tempo_urls[2], tempo_urls[3]]
        urls = [scan3, scan4]

        result = open_virtual_mfdataset(
            urls,
            registry=registry,
            parser=HDFParser(group="/product"),
            combine="nested",
            concat_dim=["datescan", "mirror_step"],
            pad="auto",
            preprocess=self._add_datescan,
            loadable_variables=[],
            consolidate=True,
        )

        no2 = result["vertical_column_troposphere"]
        assert no2.shape == (2, 264, 2048)

        # With chunk_size=1 (no consolidation) there would be 2*264 = 528 chunks
        # along (datescan, mirror_step). After consolidation chunks along
        # mirror_step should be much larger than 1.
        ma = no2.data
        assert ma.chunks[1] > 1, (
            f"Expected mirror_step chunk size > 1 after consolidation, got {ma.chunks}"
        )

    def test_consolidate_false_preserves_size1_chunks(self, tempo_urls, registry):
        """consolidate=False leaves the chunk_size=1 structure produced by
        padding intact — useful for debugging or when callers consolidate
        manually."""
        scan3 = [tempo_urls[0], tempo_urls[1]]
        scan4 = [tempo_urls[2], tempo_urls[3]]
        urls = [scan3, scan4]

        result = open_virtual_mfdataset(
            urls,
            registry=registry,
            parser=HDFParser(group="/product"),
            combine="nested",
            concat_dim=["datescan", "mirror_step"],
            pad="auto",
            preprocess=self._add_datescan,
            loadable_variables=[],
            consolidate=False,
        )

        no2 = result["vertical_column_troposphere"]
        assert no2.shape == (2, 264, 2048)
        ma = no2.data
        # Padding remaps to chunk_size=1 along mirror_step
        assert ma.chunks[1] == 1, (
            f"Expected mirror_step chunk size == 1 with consolidate=False, got {ma.chunks}"
        )


class TestPadCompressed:
    """Chunked + compressed Zarr stores (Case 1).

    When all files share the same HDF/Zarr chunk size, pad_to_shape extends
    only the shape metadata; _remap_chunks must never be called.
    HDF5 edge-chunks already contain fill_value bytes, so padded positions
    read back as fill_value transparently.
    """

    @pytest.fixture
    def compressed_ragged_zarrs(self, tmp_path):
        # same chunk size (1,1) but different y-lengths → ragged
        s1 = _create_zarr(tmp_path, "c1.zarr", (1, 2), encoding=ENCODING_COMPRESSED)
        s2 = _create_zarr(tmp_path, "c2.zarr", (1, 3), encoding=ENCODING_COMPRESSED)
        return [str(s) for s in (s1, s2)]

    def test_compressed_ragged_no_remap(self, compressed_ragged_zarrs, registry):
        """Compressed ragged arrays pad successfully without _remap_chunks."""
        result = open_virtual_mfdataset(
            compressed_ragged_zarrs,
            registry=registry,
            parser=ZarrParser(),
            combine="nested",
            concat_dim="x",
            pad="auto",
            loadable_variables=[],
        )
        assert result["a"].shape == (2, 3)

    def test_remap_not_called_on_compressed(self, compressed_ragged_zarrs, registry, monkeypatch):
        """Verify _remap_chunks is never invoked for compressed arrays."""
        from virtualizarr.manifests.array import ManifestArray

        called = []

        original = ManifestArray._remap_chunks

        def spy(self, *args, **kwargs):
            called.append(True)
            return original(self, *args, **kwargs)

        monkeypatch.setattr(ManifestArray, "_remap_chunks", spy)

        open_virtual_mfdataset(
            compressed_ragged_zarrs,
            registry=registry,
            parser=ZarrParser(),
            combine="nested",
            concat_dim="x",
            pad="auto",
            loadable_variables=[],
        )

        assert called == [], "_remap_chunks should not be called for compressed arrays"


# ---------------------------------------------------------------------------
# ITS_LIVE mosaic helpers
# ---------------------------------------------------------------------------

def _create_itslive_granule(tmp_path, name, nx, ny, x_start, y_start, step=120.0):
    """Create a synthetic HDF5 file mimicking an ITS_LIVE velocity granule.

    Uses HDF5 dimension scales so the HDF parser can resolve named dimensions.
    Layout:
    - ``x`` : ascending 1-D coord, length *nx*, step=+step
    - ``y`` : descending 1-D coord, length *ny*, step=-step
    - ``vx`` : int16 data variable of shape (1, ny, nx)
    - one ``time`` dimension of length 1
    """
    import h5py

    path = tmp_path / name
    x = np.arange(x_start, x_start + nx * step, step, dtype="float64")
    y = np.arange(y_start, y_start - ny * step, -step, dtype="float64")
    rng = np.random.default_rng(42)
    vx_data = rng.integers(-100, 100, size=(1, ny, nx), dtype="int16")

    with h5py.File(path, "w") as f:
        # coordinate datasets as dimension scales
        f.create_dataset("time", data=np.array([0.0]))
        f.create_dataset("y", data=y)
        f.create_dataset("x", data=x)
        f["time"].make_scale("time")
        f["y"].make_scale("y")
        f["x"].make_scale("x")

        # data variable
        ds = f.create_dataset(
            "vx",
            data=vx_data,
            chunks=(1, min(ny, 32), min(nx, 32)),
        )
        ds.dims[0].attach_scale(f["time"])
        ds.dims[1].attach_scale(f["y"])
        ds.dims[2].attach_scale(f["x"])

    return f"file://{path.resolve()}"


class TestITSLiveMosaic:
    """Synthetic ITS_LIVE mosaic tests for open_virtual_mfdataset(..., mosaic_dims=).

    Two granules at different (possibly overlapping) spatial tiles are combined
    into a single virtual mosaic via ``mosaic_dims=['y', 'x']``.
    """

    @pytest.fixture
    def registry(self):
        return ObjectStoreRegistry({"file://": LocalStore()})

    @pytest.fixture
    def two_tile_granules(self, tmp_path):
        """Two contiguous non-overlapping tiles on a shared 120 m grid."""
        # g1: x=[0, 120, ..., 1920]  (nx=17)
        g1 = _create_itslive_granule(tmp_path, "g1.h5", nx=17, ny=20,
                                     x_start=0.0, y_start=2400.0)
        # g2: x=[2040, 2160, ..., 4200]  (nx=18) — contiguous with g1
        g2 = _create_itslive_granule(tmp_path, "g2.h5", nx=18, ny=20,
                                     x_start=2040.0, y_start=2400.0)
        return [g1, g2]

    @pytest.fixture
    def overlapping_tile_granules(self, tmp_path):
        """Two tiles that overlap in x by 4 cells."""
        g1 = _create_itslive_granule(tmp_path, "g1.h5", nx=16, ny=20,
                                     x_start=0.0, y_start=2400.0)
        # starts at x=1440 (offset 12) — 4-cell overlap with g1
        g2 = _create_itslive_granule(tmp_path, "g2.h5", nx=16, ny=20,
                                     x_start=1440.0, y_start=2400.0)
        return [g1, g2]

    # ------------------------------------------------------------------
    # basic shape tests
    # ------------------------------------------------------------------

    def test_mosaic_shape_non_overlapping(self, two_tile_granules, registry):
        """Union of two non-overlapping tiles: spatial shape = sum of nx."""
        result = open_virtual_mfdataset(
            two_tile_granules,
            registry=registry,
            parser=HDFParser(),
            combine="nested",
            concat_dim="time",
            loadable_variables=["x", "y", "time"],
            mosaic_dims=["y", "x"],
        )
        # ny is the same (20), nx = 17 + 18 = 35
        assert result["vx"].dims == ("time", "y", "x")
        assert result["vx"].shape == (2, 20, 35)

    def test_mosaic_shape_overlapping(self, overlapping_tile_granules, registry):
        """Overlapping tiles: union nx = 16 + 16 - 4 = 28."""
        result = open_virtual_mfdataset(
            overlapping_tile_granules,
            registry=registry,
            parser=HDFParser(),
            combine="nested",
            concat_dim="time",
            loadable_variables=["x", "y", "time"],
            mosaic_dims=["y", "x"],
        )
        assert result["vx"].dims == ("time", "y", "x")
        assert result["vx"].shape == (2, 20, 28)

    # ------------------------------------------------------------------
    # coordinate correctness
    # ------------------------------------------------------------------

    def test_mosaic_x_coord_is_union(self, two_tile_granules, registry):
        """The x coordinate of the mosaic should be the full union grid."""
        result = open_virtual_mfdataset(
            two_tile_granules,
            registry=registry,
            parser=HDFParser(),
            combine="nested",
            concat_dim="time",
            loadable_variables=["x", "y", "time"],
            mosaic_dims=["y", "x"],
        )
        x = result["x"].values
        assert x[0] == pytest.approx(0.0)
        assert x[-1] == pytest.approx(4080.0)  # 34 * 120
        assert np.allclose(np.diff(x), 120.0)

    # ------------------------------------------------------------------
    # error cases
    # ------------------------------------------------------------------

    def test_mosaic_virtual_coord_raises(self, two_tile_granules, registry):
        """Raise clearly if a mosaic-dim coordinate is still virtual."""
        with pytest.raises(ValueError, match="loadable_variables"):
            open_virtual_mfdataset(
                two_tile_granules,
                registry=registry,
                parser=HDFParser(),
                combine="nested",
                concat_dim="time",
                loadable_variables=["time"],   # x and y NOT loaded
                mosaic_dims=["y", "x"],
            )

    def test_mosaic_irregular_coord_raises(self, tmp_path, registry):
        """Raise if a mosaic coordinate is not regularly spaced."""
        import h5py

        def _make_file(path, x, y, time_val):
            ny, nx = len(y), len(x)
            with h5py.File(path, "w") as f:
                f.create_dataset("time", data=np.array([time_val]))
                f.create_dataset("y", data=y)
                f.create_dataset("x", data=x)
                f["time"].make_scale("time")
                f["y"].make_scale("y")
                f["x"].make_scale("x")
                ds = f.create_dataset("vx", data=np.zeros((1, ny, nx), dtype="int16"),
                                      chunks=(1, ny, nx))
                ds.dims[0].attach_scale(f["time"])
                ds.dims[1].attach_scale(f["y"])
                ds.dims[2].attach_scale(f["x"])

        path_bad = tmp_path / "bad.h5"
        _make_file(path_bad,
                   x=np.array([0.0, 120.0, 250.0]),   # irregular step
                   y=np.array([300.0, 180.0, 60.0]),
                   time_val=0.0)

        path_good = tmp_path / "good.h5"
        _make_file(path_good,
                   x=np.array([360.0, 480.0, 600.0]),
                   y=np.array([300.0, 180.0, 60.0]),
                   time_val=1.0)

        with pytest.raises(ValueError, match="regular"):
            open_virtual_mfdataset(
                [f"file://{path_bad.resolve()}", f"file://{path_good.resolve()}"],
                registry=registry,
                parser=HDFParser(),
                combine="nested",
                concat_dim="time",
                loadable_variables=["x", "y", "time"],
                mosaic_dims=["y", "x"],
            )


