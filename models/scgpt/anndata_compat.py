"""Compatibility readers for the AnnData version pinned by scGPT."""

from __future__ import annotations

import h5py
from anndata._io import specs


def register_null_encoding_reader() -> bool:
    """Teach older AnnData releases to read the newer ``null`` scalar encoding.

    AnnData 0.12 writes values such as ``uns["log1p"]["base"] = None`` as a
    dataset with ``encoding-type=null`` and ``encoding-version=0.1.0``.  The
    scGPT environment pins AnnData 0.11.4, whose registry has no reader for
    that lossless metadata representation.  Returning ``None`` is precisely
    the newer reader's semantics and leaves the expression matrix untouched.

    Returns ``True`` when this call installed the compatibility reader and
    ``False`` when the active AnnData already provides one.
    """

    spec = specs.IOSpec("null", "0.1.0")
    key = (h5py.Dataset, spec, frozenset())
    if key in specs._REGISTRY.read:
        return False

    @specs._REGISTRY.register_read(h5py.Dataset, spec)
    def read_null(_element, *, _reader):
        return None

    return True
