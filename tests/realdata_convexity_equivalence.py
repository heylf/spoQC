"""Real-data differential check: original (db00d98) vs vectorised convexity QC.

Not collected by pytest. Usage:
    PYTHONPATH=<repo> python tests/realdata_convexity_equivalence.py <sdata.zarr> <threads> [--raw] [--skip-reference]

By default the geometries are first corrected with correct_for_valid_geometries,
which is the state cli.py hands to run_qc_cell; --raw compares on the geometries
as loaded. Prints wall time and cores busy (process CPU time delta / wall).
"""

import os
import sys
import time

import numpy as np
import plotly.graph_objects as go
import spatialdata as sd

sys.path.insert(0, os.path.dirname(__file__))
from test_convexity_equivalence import OUTPUT_COLUMNS, copy_sdata, reference_calc_convexity  # noqa: E402

from spoqc import helperfuncs  # noqa: E402
from spoqc.core import spatial  # noqa: E402
from spoqc.general.valid_geometries import correct_for_valid_geometries  # noqa: E402
from spoqc.metrics.segmentation import convexity  # noqa: E402


def timed(label, func):
    cpu, wall = os.times(), time.perf_counter()
    func()
    wall = time.perf_counter() - wall
    cpu_after = os.times()
    cpu = (cpu_after.user - cpu.user) + (cpu_after.system - cpu.system)
    print(f"[TIME] {label}: wall {wall:.3f} s, cpu {cpu:.3f} s, cores busy {cpu / wall:.2f}", flush=True)


def main(path, threads, raw, skip_reference):
    assert convexity.__file__.startswith(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    helperfuncs.plot_scatter = lambda *a, **k: None
    helperfuncs.plot_scatter_density = lambda *a, **k: None
    go.Figure.write_image = lambda *a, **k: None

    full = sd.read_zarr(path, selection=('shapes', 'tables'))
    sdata = {'cell_boundaries': full['cell_boundaries'], 'nucleus_boundaries': full['nucleus_boundaries'],
             'table': full['table']}
    if not raw:
        correct_for_valid_geometries(sdata)
    print(f"[NOTE] {len(sdata['cell_boundaries'])} cells, {len(sdata['nucleus_boundaries'])} nuclei, "
          f"{'raw' if raw else 'corrected'} geometries, threads={threads}", flush=True)

    actual = copy_sdata(sdata)
    timed(f"new calc_convexity (threads={threads})",
          lambda: convexity.calc_convexity(actual, os.environ.get('TMPDIR', '/tmp'), threads))
    cells, nuclei = sdata['cell_boundaries'], sdata['nucleus_boundaries']
    timed(f"new spatial.polygons_containing (threads={threads})",
          lambda: spatial.polygons_containing(np.asarray(cells.geometry.values),
                                              np.asarray(nuclei.geometry.centroid.values), threads))
    timed(f"new convexity_metrics (cells, threads={threads})",
          lambda: convexity.convexity_metrics(np.asarray(cells.geometry.values), threads))
    if skip_reference:
        return

    expected = copy_sdata(sdata)
    timed("original calc_convexity numerics", lambda: reference_calc_convexity(expected))
    for column in OUTPUT_COLUMNS:
        exp, act = expected['table'].obs[column], actual['table'].obs[column]
        assert exp.dtype == act.dtype, f"{column}: dtype {exp.dtype} != {act.dtype}"
        if column == 'nuclei_idxs':
            assert act.tolist() == exp.tolist(), column
            assert {type(x) for labels in act for x in labels} == {type(x) for labels in exp for x in labels}, column
            n_pairs = sum(len(labels) for labels in exp)
            print(f"[EQUAL] {column}: {len(exp)} lists, {n_pairs} (cell, nucleus) pairs identical", flush=True)
        else:
            assert act.to_numpy().tobytes() == exp.to_numpy().tobytes(), column
            print(f"[EQUAL] {column}: {exp.dtype}, {len(exp)} values bit-identical", flush=True)
    print("[PASS] all outputs identical", flush=True)


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]), '--raw' in sys.argv, '--skip-reference' in sys.argv)
