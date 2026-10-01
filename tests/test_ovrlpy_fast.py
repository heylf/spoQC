"""`install()` must patch exactly what it claims, and refuse any ovrlpy it was not written for.

The equivalence of the replacement itself is covered by `test_ovrlpy_sparse.py`.
This file is only about the binding.
"""
from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

ovrlpy = pytest.importorskip("ovrlpy")

from ovrlpy import _ovrlp, _subslicing, _utils  # noqa: E402

from spoqc._ovrlpy_fast import (  # noqa: E402
    SUPPORTED_OVRLPY_VERSIONS,
    _calculate_embedding_sparse,
    _message_passing_parallel,
    install,
)

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def restore_bindings(monkeypatch):
    """Let install() rebind ovrlpy, then put the stock function back for later tests."""
    monkeypatch.setattr(_utils, "_calculate_embedding", _utils._calculate_embedding)
    monkeypatch.setattr(_ovrlp, "_calculate_embedding", _ovrlp._calculate_embedding)
    monkeypatch.setattr(_subslicing, "_message_passing", _subslicing._message_passing)


def test_install_patches_both_bindings_on_the_pinned_version(restore_bindings):
    assert ovrlpy.__version__ in SUPPORTED_OVRLPY_VERSIONS
    install()
    assert _utils._calculate_embedding is _calculate_embedding_sparse
    assert _ovrlp._calculate_embedding is _calculate_embedding_sparse
    assert _subslicing._message_passing is _message_passing_parallel


@pytest.mark.parametrize("version", ["9.9.9", "unknown version", None])
def test_install_raises_on_an_unsupported_version(monkeypatch, restore_bindings, version):
    stock = _ovrlp._calculate_embedding
    stock_passing = _subslicing._message_passing
    monkeypatch.setattr(ovrlpy, "__version__", version, raising=False)
    with pytest.raises(RuntimeError, match="reproduces the internals"):
        install()
    assert _ovrlp._calculate_embedding is stock, "a refused install must not patch"
    assert _subslicing._message_passing is stock_passing, "a refused install must not patch"


def test_supported_version_is_the_pinned_version():
    """The shim, requirements.txt and pyproject.toml must name the same ovrlpy."""
    requirements = re.findall(
        r"^ovrlpy==(\S+)$", (REPO / "requirements.txt").read_text(), flags=re.M
    )
    project = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]
    pyproject = [d.split("==", 1)[1] for d in project["dependencies"] if d.startswith("ovrlpy==")]
    assert requirements == pyproject == list(SUPPORTED_OVRLPY_VERSIONS)


def test_only_the_accumulation_is_patched():
    """Nothing else is touched, so nothing else can regress runtime or memory.

    Process-parallel patch loops were measured and are NOT shipped: they reached
    1.97x end to end but took the full-scale tree peak from 24.63 GB to 31.50 GB.
    A change that costs 6.87 GB of peak is not an optimisation.
    """
    import spoqc._ovrlpy_fast as fast

    for name in ("compute_VSI_parallel", "_sample_expression_parallel",
                 "install_parallel_vsi", "install_parallel_sampling",
                 "_calculate_embedding_fast", "_calculate_embedding_batched"):
        assert not hasattr(fast, name), f"{name} must not ship"
