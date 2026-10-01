"""Shared whole-array, multi-threaded primitives reused across QC steps.

Submodules are not imported here: `spoqc.core.threads` must be importable before numpy,
numba or polars load (the entry point and each figure worker import it first).
"""
