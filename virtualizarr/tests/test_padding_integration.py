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
            concat_dim=["mirror_step", "datescan"],
            pad="auto",
            preprocess=self._add_datescan,
            loadable_variables=[],
        )

        assert result["vertical_column_troposphere"] is not None
        assert result["vertical_column_troposphere"].shape == (2, 264, 2048)


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

