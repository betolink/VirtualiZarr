import dataclasses
import warnings
from typing import TYPE_CHECKING, Any, Callable, Literal, Union, cast

import numpy as np
import xarray as xr
from zarr.core.metadata.v3 import ArrayV3Metadata

import virtualizarr.manifests.utils as utils
from virtualizarr.manifests.array_api import (
    MANIFESTARRAY_HANDLED_ARRAY_FUNCTIONS,
    _isnan,
)
from virtualizarr.manifests.indexing import T_Indexer, index
from virtualizarr.manifests.manifest import MISSING_CHUNK_PATH, ChunkManifest
from virtualizarr.manifests.utils import (
    ChunkKeySeparator,
    copy_and_replace_metadata,
)
from virtualizarr.utils import ceildiv, determine_chunk_grid_shape

if TYPE_CHECKING:
    from zarr.core.metadata.v3 import RegularChunkGridMetadata
else:
    try:
        from zarr.core.metadata.v3 import RegularChunkGridMetadata  # zarr-python>3.1.6
    except ImportError:
        from zarr.core.metadata.v3 import (
            RegularChunkGrid as RegularChunkGridMetadata,  # zarr-python<=3.1.6
        )


class ManifestArray:
    """
    Virtualized array representation of the chunk data in a single Zarr Array.

    Supports concatenation / stacking, but only if the two arrays to be concatenated have the same codecs.

    Cannot be directly altered.

    Implements subset of the array API standard such that it can be wrapped by xarray.
    Doesn't store the zarr array name, zattrs or ARRAY_DIMENSIONS, as instead those can be stored on a wrapping xarray object.
    """

    _manifest: ChunkManifest
    _metadata: ArrayV3Metadata

    def __init__(
        self,
        metadata: ArrayV3Metadata | dict,
        chunkmanifest: dict | ChunkManifest,
    ) -> None:
        """
        Create a ManifestArray directly from the metadata of a zarr array and the manifest of chunks.

        Parameters
        ----------
        metadata : dict or ArrayV3Metadata
        chunkmanifest : dict or ChunkManifest
        """

        if isinstance(metadata, ArrayV3Metadata):
            _metadata = metadata
        else:
            # try unpacking the dict
            _metadata = ArrayV3Metadata(**metadata)

        if not isinstance(_metadata.chunk_grid, RegularChunkGridMetadata):
            raise NotImplementedError(
                f"Only RegularChunkGrid is currently supported for chunk size, but got type {type(_metadata.chunk_grid)}"
            )

        if isinstance(chunkmanifest, ChunkManifest):
            _chunkmanifest = chunkmanifest
        elif isinstance(chunkmanifest, dict):
            separator = cast(
                ChunkKeySeparator,
                getattr(_metadata.chunk_key_encoding, "separator", "."),
            )
            _chunkmanifest = ChunkManifest(entries=chunkmanifest, separator=separator)
        else:
            raise TypeError(
                f"chunkmanifest arg must be of type ChunkManifest or dict, but got type {type(chunkmanifest)}"
            )

        # TODO check that the metadata shape and chunkmanifest shape are consistent with one another
        # TODO also cover the special case of scalar arrays

        self._metadata = _metadata
        self._manifest = _chunkmanifest

    @property
    def manifest(self) -> ChunkManifest:
        return self._manifest

    @property
    def metadata(self) -> ArrayV3Metadata:
        return self._metadata

    @property
    def chunks(self) -> tuple[int, ...]:
        """
        Individual chunk size by number of elements.
        """
        return self._metadata.chunks

    @property
    def dtype(self) -> np.dtype:
        """The native dtype of the data (typically a numpy dtype)"""
        zdtype = self.metadata.data_type
        dtype = zdtype.to_native_dtype()
        return dtype

    @property
    def shape(self) -> tuple[int, ...]:
        """
        Array shape by number of elements along each dimension.
        """
        return self.metadata.shape

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        return int(np.prod(self.shape))

    def __repr__(self) -> str:
        return f"ManifestArray<shape={self.shape}, dtype={self.dtype}, chunks={self.chunks}>"

    @property
    def nbytes_virtual(self) -> int:
        """
        The total number of bytes required to hold these virtual references in memory in bytes.

        Notes
        -----
        This is not the size of the referenced array if it were actually loaded into memory (use `.nbytes`),
        this is only the size of the pointers to the chunk locations.
        If you were to load the data into memory it would be ~1e6x larger for 1MB chunks.
        """
        # note: we don't name this method `.nbytes` as we don't want xarray's repr to use it
        return self.manifest.nbytes

    def __array_function__(self, func, types, args, kwargs) -> Any:
        """
        Hook to teach this class what to do if np.concat etc. is called on it.

        Use this instead of __array_namespace__ so that we don't make promises we can't keep.
        """

        if func not in MANIFESTARRAY_HANDLED_ARRAY_FUNCTIONS:
            return NotImplemented

        # Note: this allows subclasses that don't override
        # __array_function__ to handle ManifestArray objects
        if not all(issubclass(t, ManifestArray) for t in types):
            return NotImplemented

        return MANIFESTARRAY_HANDLED_ARRAY_FUNCTIONS[func](*args, **kwargs)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs) -> Any:
        """We have to define this in order to convince xarray that this class is a duckarray, even though we will never support ufuncs."""
        if ufunc == np.isnan:
            return _isnan(self.shape)
        return NotImplemented

    def __array__(
        self, dtype: np.typing.DTypeLike | None = None, copy: bool | None = None
    ) -> np.ndarray:
        raise NotImplementedError(
            "ManifestArrays can't be converted into numpy arrays or pandas Index objects"
        )

    def __eq__(  # type: ignore[override]
        self,
        other: Union[int, float, bool, np.ndarray, "ManifestArray"],
    ) -> np.ndarray:
        """
        Element-wise equality checking.

        Returns a numpy array of booleans.
        """
        if isinstance(other, (int, float, bool, np.ndarray)):
            # TODO what should this do when comparing against numpy arrays?
            return np.full(shape=self.shape, fill_value=False, dtype=np.dtype(bool))
        elif not isinstance(other, ManifestArray):
            raise TypeError(
                f"Cannot check equality between a ManifestArray and an object of type {type(other)}"
            )

        if self.shape != other.shape:
            raise NotImplementedError("Unsure how to handle broadcasting like this")

        if not self.metadata.to_dict() == other.metadata.to_dict():
            return np.full(shape=self.shape, fill_value=False, dtype=np.dtype(bool))
        else:
            if self.manifest == other.manifest:
                return np.full(shape=self.shape, fill_value=True, dtype=np.dtype(bool))
            else:
                # TODO this doesn't yet do what it should - it simply returns all False if any of the chunk entries are different.
                # What it should do is return True for the locations where the chunk entries are the same.
                warnings.warn(
                    "__eq__ currently is over-cautious, returning an array of all False if any of the chunk entries don't match.",
                    UserWarning,
                )

                # do chunk-wise comparison
                equal_chunks = self.manifest.elementwise_eq(other.manifest)

                if not equal_chunks.all():
                    # TODO expand chunk-wise comparison into an element-wise result instead of just returning all False
                    return np.full(
                        shape=self.shape, fill_value=False, dtype=np.dtype(bool)
                    )
                else:
                    raise RuntimeWarning("Should not be possible to get here")

    def astype(self, dtype: np.dtype, /, *, copy: bool = True) -> "ManifestArray":
        """Cannot change the dtype, but needed because xarray will call this even when it's a no-op."""
        if not np.issubdtype(self.dtype, dtype):
            raise NotImplementedError()
        else:
            return self

    def __getitem__(
        self,
        key: T_Indexer,
        /,
    ) -> "ManifestArray":
        """
        Index into this ManifestArray, returning a new ManifestArray view over a subset of chunks.

        Supports only chunk-aligned selections. A ManifestArray only stores references to where
        each chunk's bytes live, never their decoded values, so any indexer that would split into
        the interior of a chunk would require loading the underlying data — which defeats the
        point of a virtual array. Selections that would do so raise ``SubChunkIndexingError``
        (a ``ValueError`` subclass); this is a permanent constraint, not a missing feature.

        Supported indexers (and tuples thereof):

        - ``Ellipsis`` and ``None`` — no-ops and new-axis insertion.
        - ``slice`` with ``step == 1`` whose start and stop land on chunk boundaries
          (``stop == axis_length`` is also allowed, so a partial final chunk can be selected).
          Slice indexers preserve the axis.
        - ``int`` — drops the indexed axis, following numpy / array-API semantics. Only legal
          when ``chunk_size == 1`` along that axis; otherwise picking a single element would
          require splitting a chunk.
        - Slice along the largest-stride storage axis of an **uncompressed** array that fits
          entirely within one source chunk — handled by rewriting the chunk reference's byte
          offset/length rather than splitting bytes. Useful for picking a single timestep from
          a multi-row chunk on a parser like the netCDF3 one. The eligible-axis is axis 0 for
          a plain ``[BytesCodec]`` array (C-order) or axis ``order[0]`` of a prepended
          ``[TransposeCodec(order=...), BytesCodec]`` (e.g. the last axis for F-order).

        Anything else — fancy indexing with arrays, misaligned slices, ``step != 1`` —
        raises ``SubChunkIndexingError`` or ``NotImplementedError``.

        Parameters
        ----------
        key
            A basic indexer or tuple of basic indexers, one per array axis (with ``Ellipsis``
            and ``None`` allowed as per the array API).

        Returns
        -------
        ManifestArray
            A new array whose ``ChunkManifest`` references only the selected chunks.
        """
        return index(self, key)

    def rename_paths(
        self,
        new: str | Callable[[str], str],
    ) -> "ManifestArray":
        """
        Rename paths to chunks in this array's manifest.

        Accepts either a string, in which case this new path will be used for all chunks, or
        a function which accepts the old path and returns the new path.

        Parameters
        ----------
        new
            New path to use for all chunks, either as a string, or as a function which accepts and returns strings.

        Returns
        -------
        ManifestArray

        See Also
        --------
        ChunkManifest.rename_paths

        Examples
        --------
        Rename paths to reflect moving the referenced files from local storage to an S3 bucket.

        >>> def local_to_s3_url(old_local_path: str) -> str:
        ...     from pathlib import Path
        ...
        ...     new_s3_bucket_url = "http://s3.amazonaws.com/my_bucket/"
        ...
        ...     filename = Path(old_local_path).name
        ...     return str(new_s3_bucket_url / filename)
        >>>
        >>> marr.rename_paths(local_to_s3_url)  # doctest: +SKIP
        """
        renamed_manifest = self.manifest.rename_paths(new)
        return ManifestArray(metadata=self.metadata, chunkmanifest=renamed_manifest)

    def with_fill_value_only(self, fill_value: Any) -> "ManifestArray":
        """
        Return a new ManifestArray with the same schema (shape, chunks, codecs,
        dimension names, attributes) as this one, but with an empty chunk
        manifest and the given ``fill_value``.

        Reads from any chunk in the result return ``fill_value`` (see the Zarr V3
        spec for missing-chunk semantics). This is useful as a typed placeholder
        for a variable that is absent from one source but present in others — e.g.
        concatenating with real data along a new axis without materializing chunks.

        Parameters
        ----------
        fill_value
            The scalar value to store on the metadata; every read from the
            resulting array returns this value.
        """
        # dataclasses.replace bypasses the to_dict/from_dict roundtrip used in
        # copy_and_replace_metadata, which can't accept raw NaN scalars (to_dict
        # serializes NaN to the JSON string "NaN")
        new_metadata = dataclasses.replace(self.metadata, fill_value=fill_value)
        empty_manifest = ChunkManifest(
            entries={}, shape=determine_chunk_grid_shape(self.shape, self.chunks)
        )
        return ManifestArray(metadata=new_metadata, chunkmanifest=empty_manifest)

    def place_in_grid(
        self,
        new_shape: tuple[int, ...],
        offset: tuple[int, ...],
        *,
        policy: Literal["round_to_chunks", "error"] = "round_to_chunks",
    ) -> "ManifestArray":
        """
        Place this ManifestArray at *offset* inside a larger *new_shape* grid.

        The existing chunk references are copied to their offset position; every
        other cell in the new grid becomes a missing chunk (returns ``fill_value``
        on read).

        ``place_in_grid(new_shape, offset=(0, 0, ...))`` is equivalent to
        ``pad_to_shape(new_shape)``.

        Parameters
        ----------
        new_shape
            Target shape of the returned array.
        offset
            Element-wise offset for each axis. Must be a multiple of
            ``chunks[ax]`` on each axis (verified by *policy*).
        policy
            ``"round_to_chunks"`` (default): round any non-aligned offset up to
            the next chunk boundary and emit a warning.
            ``"error"``: raise ``ValueError`` if any offset is not chunk-aligned.
        """
        chunks = self.chunks
        new_shape = tuple(new_shape)
        offset = tuple(offset)

        if len(new_shape) != self.ndim or len(offset) != self.ndim:
            raise ValueError(
                f"new_shape and offset must both have length {self.ndim}"
            )

        # --- align offsets to chunk boundaries ---
        aligned_offset = list(offset)
        any_rounded = False
        for ax in range(self.ndim):
            if offset[ax] % chunks[ax] != 0:
                if policy == "error":
                    raise ValueError(
                        f"offset[{ax}]={offset[ax]} is not chunk-aligned "
                        f"(chunk size={chunks[ax]}). Use policy='round_to_chunks'."
                    )
                aligned_offset[ax] = ceildiv(offset[ax], chunks[ax]) * chunks[ax]
                any_rounded = True
        if any_rounded:
            warnings.warn(
                f"place_in_grid rounded offset from {offset} to "
                f"{tuple(aligned_offset)} to align with chunk boundaries {chunks}"
            )
        offset = tuple(aligned_offset)

        # --- validate new_shape fits the array at the given offset ---
        for ax in range(self.ndim):
            needed = offset[ax] + self.shape[ax]
            if new_shape[ax] < needed:
                raise ValueError(
                    f"new_shape[{ax}]={new_shape[ax]} is too small: "
                    f"offset[{ax}]={offset[ax]} + shape[{ax}]={self.shape[ax]} "
                    f"= {needed} > {new_shape[ax]}"
                )
            if new_shape[ax] < self.shape[ax] and offset[ax] == 0:
                raise ValueError(
                    f"new_shape[{ax}]={new_shape[ax]} < shape[{ax}]={self.shape[ax]}; "
                    "shrinking is not supported"
                )

        old_grid = determine_chunk_grid_shape(self.shape, chunks)
        new_grid = determine_chunk_grid_shape(new_shape, chunks)
        chunk_offset = tuple(offset[ax] // chunks[ax] for ax in range(self.ndim))

        # fast path: nothing actually changes
        if chunk_offset == tuple(0 for _ in range(self.ndim)) and new_grid == old_grid:
            new_meta = copy_and_replace_metadata(self.metadata, new_shape=list(new_shape))
            return ManifestArray(metadata=new_meta, chunkmanifest=self.manifest)

        new_paths = np.full(new_grid, MISSING_CHUNK_PATH, dtype=np.dtypes.StringDType())
        new_offsets = np.zeros(new_grid, dtype=np.dtype("uint64"))
        new_lengths = np.zeros(new_grid, dtype=np.dtype("uint64"))

        dest = tuple(
            slice(chunk_offset[ax], chunk_offset[ax] + old_grid[ax])
            for ax in range(self.ndim)
        )
        new_paths[dest] = self.manifest._paths
        new_offsets[dest] = self.manifest._offsets
        new_lengths[dest] = self.manifest._lengths

        new_manifest = ChunkManifest.from_arrays(
            paths=new_paths,
            offsets=new_offsets,
            lengths=new_lengths,
            validate_paths=False,
            inlined=dict(self.manifest._inlined) if self.manifest._inlined else None,
        )
        new_meta = copy_and_replace_metadata(self.metadata, new_shape=list(new_shape))
        return ManifestArray(metadata=new_meta, chunkmanifest=new_manifest)

    def pad_to_shape(
        self,
        new_shape: tuple[int, ...],
        *,
        policy: Literal["round_to_chunks", "error"] = "round_to_chunks",
    ) -> "ManifestArray":
        """
        Pad this ManifestArray to a larger shape by expanding the chunk manifest.

        New chunk-grid cells are initialized as missing chunks (return ``fill_value``
        on read). Existing chunk references are copied unchanged.

        Parameters
        ----------
        new_shape
            Target array shape. Must be elementwise >= the current shape.
        policy
            ``"round_to_chunks"`` (default): round ``new_shape`` up to the next
            multiple of ``chunks[axis]`` on any axis where it is not chunk-aligned.
            ``"error"``: raise ``ValueError`` if ``new_shape[axis]`` is not a
            multiple of ``chunks[axis]``.

        Returns
        -------
        ManifestArray
        """
        old_shape = self.shape
        chunks = self.chunks
        new_shape = tuple(new_shape)

        if len(new_shape) != self.ndim:
            raise ValueError(
                f"new_shape length {len(new_shape)} must match ndim {self.ndim}"
            )
        for ax, (o, n) in enumerate(zip(old_shape, new_shape)):
            if n < o:
                raise ValueError(
                    f"new_shape[{ax}]={n} < old_shape[{ax}]={o}; "
                    "only padding (enlarging) is supported"
                )

        if policy == "round_to_chunks":
            rounded = list(new_shape)
            any_rounded = False
            for ax in range(self.ndim):
                if new_shape[ax] % chunks[ax] != 0:
                    rounded[ax] = ceildiv(new_shape[ax], chunks[ax]) * chunks[ax]
                    any_rounded = True
            if any_rounded:
                warnings.warn(
                    f"pad_to_shape rounded new_shape from {new_shape} to "
                    f"{tuple(rounded)} to align with chunk boundaries {chunks}"
                )
                new_shape = tuple(rounded)
        elif policy == "error":
            for ax in range(self.ndim):
                if new_shape[ax] % chunks[ax] != 0:
                    raise ValueError(
                        f"axis {ax}: length {new_shape[ax]} is not a multiple of "
                        f"chunk size {chunks[ax]}. Use policy='round_to_chunks'."
                    )
        else:
            raise ValueError(f"Unknown policy: {policy!r}")

        zero_offset = tuple(0 for _ in range(self.ndim))
        return self.place_in_grid(new_shape, zero_offset, policy="error")

    def _remap_chunks(self, new_chunks: tuple[int, ...]) -> "ManifestArray":
        """Return a new array with the same shape but remapped to *new_chunks*.

        Each existing chunk entry is split into sub-entries aligned with
        *new_chunks*.  Requires ``new_chunks[ax] <= chunks[ax]`` on every axis.

        Raises ``ValueError`` if the array has any bytes-to-bytes compression
        codec, because byte-splitting compressed chunks produces corrupt data.
        """
        from virtualizarr.codecs import is_uncompressed

        if not is_uncompressed(self):
            raise ValueError(
                "_remap_chunks cannot be called on a compressed ManifestArray. "
                "Byte-splitting compressed chunks would produce corrupt virtual "
                "references. Ensure chunk shapes are consistent across files, or "
                "use uncompressed storage."
            )
        old_chunks = self.chunks
        old_shape = self.shape
        new_grid = determine_chunk_grid_shape(old_shape, new_chunks)
        old_grid = determine_chunk_grid_shape(old_shape, old_chunks)
        elem_bytes = self.dtype.itemsize

        paths = np.full(new_grid, MISSING_CHUNK_PATH, dtype=np.dtypes.StringDType())
        offsets = np.zeros(new_grid, dtype=np.dtype("uint64"))
        lengths = np.zeros(new_grid, dtype=np.dtype("uint64"))

        for old_idx in np.ndindex(*old_grid):
            entry = self.manifest.get_entry(old_idx)
            if entry is None:
                continue

            start = tuple(old_idx[ax] * old_chunks[ax] for ax in range(self.ndim))
            end = tuple(
                min(start[ax] + old_chunks[ax], old_shape[ax])
                for ax in range(self.ndim)
            )
            dims = tuple(end[ax] - start[ax] for ax in range(self.ndim))

            i0 = tuple(start[ax] // new_chunks[ax] for ax in range(self.ndim))
            i1 = tuple(
                (end[ax] - 1) // new_chunks[ax] + 1 for ax in range(self.ndim)
            )

            for sub_idx in np.ndindex(
                *tuple(max(i1[ax] - i0[ax], 0) for ax in range(self.ndim))
            ):
                new_idx = tuple(i0[ax] + sub_idx[ax] for ax in range(self.ndim))
                seg_start = tuple(
                    max(start[ax], new_idx[ax] * new_chunks[ax])
                    for ax in range(self.ndim)
                )
                seg_end = tuple(
                    min(end[ax], (new_idx[ax] + 1) * new_chunks[ax])
                    for ax in range(self.ndim)
                )
                if any(seg_end[ax] <= seg_start[ax] for ax in range(self.ndim)):
                    continue

                rel = tuple(seg_start[ax] - start[ax] for ax in range(self.ndim))
                sz = tuple(seg_end[ax] - seg_start[ax] for ax in range(self.ndim))

                elem_off = 0
                stride = 1
                for ax in range(self.ndim - 1, -1, -1):
                    elem_off += rel[ax] * stride
                    stride *= dims[ax]
                elem_cnt = int(np.prod(sz))

                paths[new_idx] = entry["path"]
                offsets[new_idx] = int(entry["offset"]) + elem_off * elem_bytes
                lengths[new_idx] = elem_cnt * elem_bytes

        new_manifest = ChunkManifest.from_arrays(
            paths=paths, offsets=offsets, lengths=lengths, validate_paths=False
        )
        new_meta = copy_and_replace_metadata(
            self.metadata, new_shape=list(old_shape), new_chunks=list(new_chunks)
        )
        return ManifestArray(metadata=new_meta, chunkmanifest=new_manifest)

    def consolidate_chunks(
        self, target_chunks: tuple[int, ...]
    ) -> "ManifestArray":
        """Return a new ManifestArray whose chunk shape is *target_chunks*.

        Each target chunk must contain only sub-chunks that share the same
        source URL and form a single contiguous byte range (as produced by
        ``_remap_chunks`` followed by ``place_in_grid``).  When that condition
        holds the sub-chunks are merged back into one manifest entry.

        A target chunk cell is emitted as **missing** when every contributing
        sub-chunk in the current manifest is already missing.

        Raises ``ValueError`` if any target chunk contains sub-chunks from
        different source files, or sub-chunks whose byte ranges are not
        contiguous.

        Parameters
        ----------
        target_chunks
            The desired chunk shape after consolidation.  Must satisfy
            ``target_chunks[ax] >= self.chunks[ax]`` on every axis, and each
            ``target_chunks[ax]`` must be a multiple of ``self.chunks[ax]``.
        """
        if len(target_chunks) != self.ndim:
            raise ValueError(
                f"target_chunks must have length {self.ndim}, "
                f"but got {len(target_chunks)}"
            )
        for ax in range(self.ndim):
            if target_chunks[ax] < self.chunks[ax]:
                raise ValueError(
                    f"target_chunks[{ax}]={target_chunks[ax]} < "
                    f"current chunks[{ax}]={self.chunks[ax]}; "
                    "consolidate_chunks cannot shrink chunks"
                )
            if target_chunks[ax] % self.chunks[ax] != 0:
                raise ValueError(
                    f"target_chunks[{ax}]={target_chunks[ax]} is not a "
                    f"multiple of current chunks[{ax}]={self.chunks[ax]}"
                )

        # fast path: already at target
        if tuple(target_chunks) == self.chunks:
            return self

        old_chunks = self.chunks
        new_grid = determine_chunk_grid_shape(self.shape, target_chunks)
        # ratio: how many sub-chunks per target chunk along each axis
        ratio = tuple(target_chunks[ax] // old_chunks[ax] for ax in range(self.ndim))

        new_paths = np.full(new_grid, MISSING_CHUNK_PATH, dtype=np.dtypes.StringDType())
        new_offsets = np.zeros(new_grid, dtype=np.dtype("uint64"))
        new_lengths = np.zeros(new_grid, dtype=np.dtype("uint64"))

        src_paths = self.manifest._paths
        src_offsets = self.manifest._offsets
        src_lengths = self.manifest._lengths

        for new_idx in np.ndindex(*new_grid):
            # sub-chunk slice in the current grid that maps into this target cell
            src_slices = tuple(
                slice(new_idx[ax] * ratio[ax], (new_idx[ax] + 1) * ratio[ax])
                for ax in range(self.ndim)
            )
            sub_paths = src_paths[src_slices]
            sub_offsets = src_offsets[src_slices]
            sub_lengths = src_lengths[src_slices]

            # flatten to 1-D for easier scanning
            flat_paths = sub_paths.ravel()
            flat_offsets = sub_offsets.ravel().astype(np.int64)
            flat_lengths = sub_lengths.ravel().astype(np.int64)

            # filter out missing sub-chunks
            valid_mask = np.array([p != MISSING_CHUNK_PATH for p in flat_paths])
            if not valid_mask.any():
                # all missing → emit missing target chunk (already initialised)
                continue

            valid_paths = flat_paths[valid_mask]
            valid_offsets = flat_offsets[valid_mask]
            valid_lengths = flat_lengths[valid_mask]

            # all sub-chunks must point to the same file
            first_path = valid_paths[0]
            if not np.all(np.array([p == first_path for p in valid_paths])):
                raise ValueError(
                    f"Cannot consolidate target chunk {new_idx}: sub-chunks "
                    "span more than one source file."
                )

            # sort by offset so we can check contiguity regardless of array order
            order = np.argsort(valid_offsets)
            sorted_offsets = valid_offsets[order]
            sorted_lengths = valid_lengths[order]

            # check contiguity: each chunk's end == next chunk's start
            ends = sorted_offsets + sorted_lengths
            if not np.all(ends[:-1] == sorted_offsets[1:]):
                raise ValueError(
                    f"Cannot consolidate target chunk {new_idx}: byte ranges "
                    "are not contiguous."
                )

            merged_offset = int(sorted_offsets[0])
            merged_length = int(ends[-1] - sorted_offsets[0])

            new_paths[new_idx] = first_path
            new_offsets[new_idx] = np.uint64(merged_offset)
            new_lengths[new_idx] = np.uint64(merged_length)

        new_manifest = ChunkManifest.from_arrays(
            paths=new_paths,
            offsets=new_offsets,
            lengths=new_lengths,
            validate_paths=False,
        )
        new_meta = copy_and_replace_metadata(
            self.metadata,
            new_shape=list(self.shape),
            new_chunks=list(target_chunks),
        )
        return ManifestArray(metadata=new_meta, chunkmanifest=new_manifest)

    def to_virtual_variable(self) -> xr.Variable:
        """
        Create a "virtual" xarray.Variable containing the contents of one zarr array.

        The returned variable will be "virtual", i.e. it will wrap a single ManifestArray object.
        """

        # The xarray data model stores dimension names and arbitrary extra metadata outside of the wrapped array class,
        # so to avoid that information being duplicated we strip it from the ManifestArray before wrapping it.
        if self.metadata.dimension_names is not None:
            dims = self.metadata.dimension_names
        elif self.ndim == 0:
            dims = ()
        else:
            raise ValueError(
                f"Cannot create virtual variable from {self.ndim}-dimensional array without dimension names."
            )
        attrs = self.metadata.attributes
        stripped_metadata = utils.copy_and_replace_metadata(
            self.metadata, new_dimension_names=None, new_attributes={}
        )
        stripped_marr = ManifestArray(
            chunkmanifest=self.manifest, metadata=stripped_metadata
        )

        return xr.Variable(
            data=stripped_marr,
            dims=dims,
            attrs=attrs,
        )
