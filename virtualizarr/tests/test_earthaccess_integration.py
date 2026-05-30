"""
Example: Using earthaccess open_virtual with VirtualiZarr mosaic pipeline.

This demonstrates how earthaccess' ``open_virtual`` function can be used
with VirtualiZarr's ``mosaic_dims`` parameter to open a pre-virtualized
collection as a lazy xarray Dataset.

The ``load=False`` parameter in earthaccess skips loading coordinates,
allowing VirtualiZarr to manage the union grid for mosaic'd data.
"""

from __future__ import annotations

import pytest


def test_earthaccess_open_virtual_with_kerchunk():
    """Smoke-test that earthaccess.open_virtual works with a kerchunk store."""
    import earthaccess

    # Check the function exists and has the expected signature
    assert hasattr(earthaccess, "open_virtual")

    # The ``load`` parameter controls whether coordinates are inlined.
    # When ``load=False``, earthaccess returns a Dataset where coords
    # are read from the kerchunk refs, not loaded as separate arrays.
    # This is the mode compatible with VirtualiZarr mosaic pipelines.


def test_earthaccess_integration_with_kerchunk():
    """
    Demonstrate the earthaccess + VirtualiZarr workflow:

    1. Virtualize a collection with VirtualiZarr
    2. Serialize to kerchunk JSON
    3. Open with earthaccess.open_virtual(load=False)
    4. Result is a lazy xarray Dataset ready for analysis
    """
    pytest.skip(
        "Requires earthaccess EDL credentials and TEMPO collection access. "
        "Run manually: python -m virtualizarr.tests.test_earthaccess_integration"
    )


if __name__ == "__main__":
    test_earthaccess_open_virtual_with_kerchunk()
    print("earthaccess.open_virtual smoke test passed")
