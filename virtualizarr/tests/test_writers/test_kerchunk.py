import numpy as np
import pandas as pd
import pytest
import xarray as xr
from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import LocalStore
from xarray import Dataset
from zarr.core.metadata.v2 import ArrayV2Metadata

from virtualizarr.manifests import ChunkManifest, ManifestArray
from virtualizarr.parsers import KerchunkJSONParser, KerchunkParquetParser
from virtualizarr.tests import requires_fastparquet, requires_kerchunk
from virtualizarr.utils import convert_v3_to_v2_metadata, kerchunk_refs_as_json


def test_deserialize_to_json():
    refs = {
        "version": 1,
        "refs": {
            ".zgroup": '{"zarr_format":2}',
            ".zattrs": "{}",
            "a/.zarray": '{"shape":[2,3],"chunks":[2,3],"fill_value":0,"order":"C","filters":null,"dimension_separator":".","compressor":null,"attributes":{},"zarr_format":2,"dtype":"<i8"}',
            "a/.zattrs": '{"_ARRAY_DIMENSIONS":["x","y"]}',
            "a/0.0": ["/test.nc", 6144, 48],
        },
    }
    json_expected = {
        "version": 1,
        "refs": {
            ".zgroup": {"zarr_format": 2},
            ".zattrs": {},
            "a/.zarray": {
                "shape": [2, 3],
                "chunks": [2, 3],
                "fill_value": 0,
                "order": "C",
                "filters": None,
                "dimension_separator": ".",
                "compressor": None,
                "attributes": {},
                "zarr_format": 2,
                "dtype": "<i8",
            },
            "a/.zattrs": {
                "_ARRAY_DIMENSIONS": ["x", "y"],
            },
            "a/0.0": ["/test.nc", 6144, 48],
        },
    }
    refs_as_json = kerchunk_refs_as_json(refs)
    assert refs_as_json == json_expected


@requires_kerchunk
class TestAccessor:
    def test_accessor_to_kerchunk_dict(self, array_v3_metadata):
        manifest = ChunkManifest(
            entries={"0.0": dict(path="file:///test.nc", offset=6144, length=48)}
        )
        arr = ManifestArray(
            chunkmanifest=manifest,
            metadata=array_v3_metadata(
                shape=(2, 3),
                data_type=np.dtype("<i8"),
                chunks=(2, 3),
                codecs=[],
                fill_value=None,
            ),
        )
        ds = Dataset({"a": (["x", "y"], arr)})

        expected_ds_refs = {
            "version": 1,
            "refs": {
                ".zgroup": '{"zarr_format":2}',
                ".zattrs": "{}",
                "a/.zarray": '{"shape":[2,3],"chunks":[2,3],"fill_value":0,"order":"C","filters":null,"dimension_separator":".","compressor":null,"attributes":{},"zarr_format":2,"dtype":"<i8"}',
                "a/.zattrs": '{"_ARRAY_DIMENSIONS":["x","y"]}',
                "a/0.0": ["/test.nc", 6144, 48],
            },
        }

        result_ds_refs = ds.vz.to_kerchunk(format="dict")
        assert kerchunk_refs_as_json(result_ds_refs) == kerchunk_refs_as_json(
            expected_ds_refs
        )

    def test_accessor_to_kerchunk_dict_empty(self, array_v3_metadata):
        """
        A manifest with no real chunks (all missing) should still emit inlined
        fill-value bytes for each chunk key.  This ensures kerchunk parquet can
        represent missing chunks without storing a None/NaN URL path (which
        round-trips through pandas as float NaN and crashes fsspec).
        """
        import base64

        manifest = ChunkManifest(entries={}, shape=(1, 1))
        arr = ManifestArray(
            chunkmanifest=manifest,
            metadata=array_v3_metadata(
                shape=(2, 3),
                data_type=np.dtype("<i8"),
                chunks=(2, 3),
                codecs=[],
                fill_value=None,
            ),
        )
        ds = Dataset({"a": (["x", "y"], arr)})

        result_ds_refs = ds.vz.to_kerchunk(format="dict")
        refs = result_ds_refs["refs"]

        # The single chunk key "0.0" must be present and inlined as fill bytes.
        assert "a/0.0" in refs, "missing chunk must be emitted as inlined fill bytes"
        val = refs["a/0.0"]
        assert isinstance(val, str) and val.startswith("base64:"), (
            f"expected base64-encoded fill bytes, got {val!r}"
        )
        # Decode and verify: fill_value=0 for int64, chunk shape (2,3) → 48 bytes
        raw = base64.b64decode(val[len("base64:"):])
        assert raw == np.zeros((2, 3), dtype="<i8").tobytes()

    def test_accessor_to_kerchunk_dict_empty_with_filters(self, tmp_path, array_v3_metadata):
        """
        Fill-value bytes for missing chunks must be encoded with the array's
        filters and compressor, so that zarr can decode them.  Previously,
        _fill_chunk_bytes returned raw uncompressed bytes, causing
        ``zlib.error: incorrect header check`` when zarr tried to apply the
        zlib filter on read.
        """
        # Build a ManifestArray with shuffle + zlib codecs (mirrors ITS_LIVE HDF5 layout)
        manifest = ChunkManifest(entries={}, shape=(1, 1))
        arr = ManifestArray(
            chunkmanifest=manifest,
            metadata=array_v3_metadata(
                shape=(1, 4),
                data_type=np.dtype("<i2"),
                chunks=(1, 4),
                codecs=[
                    {"name": "bytes", "configuration": {"endian": "little"}},
                    {"name": "numcodecs.shuffle", "configuration": {"elementsize": 2}},
                    {"name": "numcodecs.zlib", "configuration": {"level": 1}},
                ],
                fill_value=-32767,
            ),
        )
        ds = Dataset({"a": (["t", "x"], arr)})
        path = tmp_path / "filtered.json"
        ds.vz.to_kerchunk(path, format="json")

        # zarr must be able to open and decode the fill chunk without raising
        # ``zlib.error: incorrect header check``
        opened = xr.open_dataset(str(path), engine="kerchunk")
        values = opened["a"].values  # triggers actual decode
        assert values.shape == (1, 4)
        # fill_value=-32767 → xarray masks as NaN (float32)
        assert np.all(np.isnan(values))


        import ujson

        manifest = ChunkManifest(
            entries={"0.0": dict(path="file:///test.nc", offset=6144, length=48)}
        )
        arr = ManifestArray(
            chunkmanifest=manifest,
            metadata=array_v3_metadata(
                shape=(2, 3),
                data_type=np.dtype("<i8"),
                chunks=(2, 3),
                codecs=[],
                fill_value=None,
            ),
        )
        ds = Dataset({"a": (["x", "y"], arr)})

        filepath = tmp_path / "refs.json"

        ds.vz.to_kerchunk(filepath, format="json")

        with open(filepath) as json_file:
            loaded_refs = ujson.load(json_file)

        expected_ds_refs = {
            "version": 1,
            "refs": {
                ".zgroup": '{"zarr_format":2}',
                ".zattrs": "{}",
                "a/.zarray": '{"shape":[2,3],"chunks":[2,3],"fill_value":0,"order":"C","filters":null,"dimension_separator":".","compressor":null,"attributes":{},"zarr_format":2,"dtype":"<i8"}',
                "a/.zattrs": '{"_ARRAY_DIMENSIONS":["x","y"]}',
                "a/0.0": ["/test.nc", 6144, 48],
            },
        }
        assert kerchunk_refs_as_json(loaded_refs) == kerchunk_refs_as_json(
            expected_ds_refs
        )

    def test_write_inlined_chunks_to_dict(self, array_v3_metadata):
        # ManifestArray with mixed inlined+virtual chunks should serialize the
        # inlined positions as `base64:<b64>` strings and the virtual ones as
        # `[path, offset, length]` triples.
        manifest = ChunkManifest(
            entries={
                "0.0": {
                    "path": "",
                    "offset": 0,
                    "length": 8,
                    "data": b"\x01\x02\x03\x04\x05\x06\x07\x08",
                },
                "0.1": {"path": "file:///foo.nc", "offset": 100, "length": 8},
            }
        )
        arr = ManifestArray(
            chunkmanifest=manifest,
            metadata=array_v3_metadata(
                shape=(1, 2),
                data_type=np.dtype("<i4"),
                chunks=(1, 1),
                codecs=[],
                fill_value=None,
            ),
        )
        ds = Dataset({"a": (["x", "y"], arr)})

        result = ds.vz.to_kerchunk(format="dict")
        refs = result["refs"]
        assert refs["a/0.0"] == "base64:AQIDBAUGBwg="
        assert refs["a/0.1"] == ["/foo.nc", 100, 8]

    @requires_kerchunk
    def test_write_inlined_chunks_roundtrip(self, tmp_path, array_v3_metadata):
        # Write a manifest containing inlined chunks to JSON, then re-parse it
        # with KerchunkJSONParser; the inlined bytes should survive the round trip.
        inlined = b"\x01\x02\x03\x04\x05\x06\x07\x08"
        manifest = ChunkManifest(
            entries={
                "0.0": {"path": "", "offset": 0, "length": 8, "data": inlined},
                "0.1": {"path": "file:///foo.nc", "offset": 100, "length": 8},
            }
        )
        arr = ManifestArray(
            chunkmanifest=manifest,
            metadata=array_v3_metadata(
                shape=(1, 2),
                data_type=np.dtype("<i4"),
                chunks=(1, 1),
                codecs=[],
                fill_value=None,
            ),
        )
        ds = Dataset({"a": (["x", "y"], arr)})

        filepath = tmp_path / "refs.json"
        ds.vz.to_kerchunk(filepath, format="json")

        registry = ObjectStoreRegistry({"file://": LocalStore()})
        manifeststore = KerchunkJSONParser()(f"file://{filepath}", registry=registry)
        roundtripped = manifeststore._group._members["a"]
        assert roundtripped.manifest._inlined == {(0, 0): inlined}
        assert roundtripped.manifest.dict()["0.1"] == {
            "path": "file:///foo.nc",
            "offset": 100,
            "length": 8,
        }

    @requires_kerchunk
    @requires_fastparquet
    def test_write_inlined_chunks_roundtrip_parquet(self, tmp_path, array_v3_metadata):
        # As above but via the parquet serialization.
        inlined = b"\x01\x02\x03\x04\x05\x06\x07\x08"
        manifest = ChunkManifest(
            entries={
                "0.0": {"path": "", "offset": 0, "length": 8, "data": inlined},
                "0.1": {"path": "file:///foo.nc", "offset": 100, "length": 8},
            }
        )
        arr = ManifestArray(
            chunkmanifest=manifest,
            metadata=array_v3_metadata(
                shape=(1, 2),
                data_type=np.dtype("<i4"),
                chunks=(1, 1),
                codecs=[],
                fill_value=None,
            ),
        )
        ds = Dataset({"a": (["x", "y"], arr)})

        filepath = tmp_path / "refs"
        ds.vz.to_kerchunk(filepath, format="parquet")

        registry = ObjectStoreRegistry({"file://": LocalStore()})
        manifeststore = KerchunkParquetParser()(str(filepath), registry=registry)
        roundtripped = manifeststore._group._members["a"]
        assert roundtripped.manifest._inlined == {(0, 0): inlined}
        assert roundtripped.manifest.dict()["0.1"] == {
            "path": "file:///foo.nc",
            "offset": 100,
            "length": 8,
        }

    @requires_fastparquet
    def test_accessor_to_kerchunk_parquet(self, tmp_path, array_v3_metadata):
        import ujson

        chunks_dict = {
            "0.0": {"path": "file:///foo.nc", "offset": 100, "length": 100},
            "0.1": {"path": "file:///foo.nc", "offset": 200, "length": 100},
        }
        manifest = ChunkManifest(entries=chunks_dict)
        arr = ManifestArray(
            chunkmanifest=manifest,
            metadata=array_v3_metadata(
                shape=(2, 4),
                data_type=np.dtype("<i8"),
                chunks=(2, 2),
                codecs=[],
                fill_value=None,
            ),
        )
        ds = Dataset({"a": (["x", "y"], arr)})

        filepath = tmp_path / "refs"

        ds.vz.to_kerchunk(filepath, format="parquet", record_size=2)

        with open(tmp_path / "refs" / ".zmetadata") as f:
            meta = ujson.load(f)
            assert list(meta) == ["metadata", "record_size"]
            assert meta["record_size"] == 2

        df0 = pd.read_parquet(filepath / "a" / "refs.0.parq")

        assert df0.to_dict() == {
            "offset": {0: 100, 1: 200},
            "path": {
                0: "/foo.nc",
                1: "/foo.nc",
            },
            "size": {0: 100, 1: 100},
            "raw": {0: None, 1: None},
        }


@pytest.mark.parametrize(
    ["dtype", "endian", "expected_dtype_char"],
    [("i8", "little", "<"), ("i8", "big", ">"), ("i1", None, "|")],
)
def test_convert_v3_to_v2_metadata(
    array_v3_metadata, dtype: str, endian: str | None, expected_dtype_char: str
):
    shape = (5, 20)
    chunks = (5, 10)

    codecs = [
        {"name": "bytes", "configuration": {"endian": endian}},
        {
            "name": "numcodecs.delta",
            "configuration": {"dtype": f"{expected_dtype_char}{dtype}"},
        },
        {
            "name": "numcodecs.blosc",
            "configuration": {"cname": "zstd", "clevel": 5, "shuffle": 1},
        },
    ]

    v3_metadata = array_v3_metadata(
        data_type=np.dtype(dtype), shape=shape, chunks=chunks, codecs=codecs
    )
    v2_metadata = convert_v3_to_v2_metadata(v3_metadata)

    assert isinstance(v2_metadata, ArrayV2Metadata)
    assert v2_metadata.shape == shape
    expected_dtype = np.dtype(f"{expected_dtype_char}{dtype}")
    assert v2_metadata.dtype.to_native_dtype() == expected_dtype
    assert v2_metadata.chunks == chunks
    assert v2_metadata.fill_value == 0

    assert v2_metadata.filters
    filter_codec, compressor_codec = v2_metadata.filters
    compressor_config = compressor_codec.get_config()
    assert compressor_config["id"] == "blosc"
    assert compressor_config["cname"] == "zstd"
    assert compressor_config["clevel"] == 5
    assert compressor_config["shuffle"] == 1
    assert compressor_config["blocksize"] == 0

    filters_config = filter_codec.get_config()
    assert filters_config["id"] == "delta"
    expected_delta_dtype = f"{expected_dtype_char}{dtype}"
    assert filters_config["dtype"] == expected_delta_dtype
    assert filters_config["astype"] == expected_delta_dtype
    assert v2_metadata.attributes == {}


def test_warn_if_no_virtual_vars():
    non_virtual_ds = xr.Dataset({"foo": ("x", [10, 20, 30]), "x": ("x", [1, 2, 3])})
    with pytest.warns(UserWarning, match="non-virtual"):
        non_virtual_ds.vz.to_kerchunk()
