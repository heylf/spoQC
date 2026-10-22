from __future__ import annotations

# Headless backend for the CLI only; set before cli imports pyplot.
import matplotlib
matplotlib.use("Agg")

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())