from __future__ import annotations

try:
    from importlib.metadata import version as _version
    __version__ = _version("spoqc")
except Exception:  # pragma: no cover
    __version__ = "0.1.0"

__all__ = ["__version__"]