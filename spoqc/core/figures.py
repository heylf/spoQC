"""The one figure-writing path: save_figure() hands a finished figure to a worker process.

The figure is pickled at the call site and a worker renders and writes every requested file, so
the main thread goes straight on computing. Rendering (Agg rasterising, PDF path serialisation,
kaleido export) is 70-95% of figure time and holds the GIL, so it needs processes, not threads.

start(n) opens the pool; without it save_figure writes synchronously in this process.
wait() blocks until every submitted figure is written and re-raises the first worker error: call
it before anything lists, moves or reads a figure file. stop() waits and closes the pool;
abort() cancels queued writes on the error path. Only figure files go through here; no computed
data is touched. imshow() draws a dense per-pixel image already reduced, so matplotlib never
copies the full array on the main thread.

CPU budget: each worker is one render process plus, for plotly, its kaleido Chromium; the two run
in turn, so a worker keeps about one core busy. start(threads) gets the run's whole budget.
While the main thread computes, THREADS // THREADS_PER_FIGURE_WORKER background workers write
figures, and the main thread's own image kernels (imshow, image reduction) run on the budget
less the workers busy at the call. A drain pool of the remaining workers takes figures only
while the main thread is blocked:
- in wait() or stop(), all of it, so the figure pools use every core of the budget. Nothing is
  in flight when they return.
- in save_figure at MAX_PENDING_BYTES, all of it while the cap holds. Once the cap clears the
  drain pool takes no more figures, and save_figure still waits until one of its workers is
  free: the main thread resumes with at most THREADS - 1 workers busy (background workers only
  replace one another and drain renders are not replaced), so its core is its own.
"""

import collections
import gc
import io
import logging
import multiprocessing
import os
import pickle
import threading
import warnings
from concurrent.futures import ProcessPoolExecutor

import matplotlib
import matplotlib.pyplot as plt
import numba
import numpy as np
from matplotlib.collections import Collection, QuadMesh
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from numba import njit, prange

from spoqc.core import figure_worker

# Threads of the run's budget per figure worker (see the CPU budget note above).
THREADS_PER_FIGURE_WORKER = 4
# Collections with at least this many elements are rasterised in PDF output. Vector PDF costs
# ~0.1 ms per marker (44 s and 92 MB for the 410k-point doublet 3D scatter); rasterised it is
# 5 s and 0.3 MB, drawn at the savefig dpi like the PNG. Axes and text stay vector.
RASTERIZE_MIN_ELEMENTS = 10_000
# Images with more samples than output pixels are reduced to this many samples per output pixel
# (per axis) in the pickled copy; the renderer antialiases the rest down to output pixels.
# Scalar images are coloured first and the colours averaged (premultiplied by alpha), which is
# what matplotlib's own antialiasing of a downsampled image does. Measured on the hqtr metric set
# at full-scale density (18 samples/px) against matplotlib's full-resolution render: 1.1-3.3/255
# mean at 2 samples/px, where averaging the values instead was up to 11/255 (LBP) even at 4.
# Integer and bool images are coloured the same way, so a label image blends its labels' colours
# (what matplotlib's 'rgba' interpolation stage shows), never the colour of a mean label.
# 'nearest'/'none' interpolation, the 'data' interpolation stage and norms other than a plain
# linear Normalize are never reduced.
IMAGE_SAMPLES_PER_PIXEL = 2
# Pickled figure bytes submitted but not yet written before save_figure blocks. Each blob is held
# about twice (here and in the pipe to the worker). A single larger figure is still submitted.
# While save_figure blocks, the drain pool writes held figures too (see the CPU budget note).
MAX_PENDING_BYTES = 2 * 1024**3

# mplot3d warns that set_rasterized on its collections "will be ignored", but Axes3D composites
# them rasterised all the same: the 410k-point doublet 3D PDF goes from 92 MB to 0.3 MB.
warnings.filterwarnings(
    "ignore", message="Rasterization of .*Path3DCollection", category=UserWarning
)

_executor = None  # background pool: figures written while the main thread computes
_drain = None  # the rest of the thread budget: used only while the main thread is blocked
_budget = None  # start()'s thread budget
_workers = {}  # pool -> worker count
# Figures are handed to a pool only when it has an idle worker, so none sits in a busy pool's
# queue when wait() opens the drain pool. Guarded by _state; pool callbacks run on the pools'
# manager threads.
_held = collections.deque()  # (the _write arguments, pickled bytes), not yet in a pool
_in_flight = {}  # future -> (pool, pickled bytes)
_reserved = {}  # token -> (pool, pickled bytes): taken from _held, being submitted
_errors = []  # worker exceptions, re-raised on the main thread
_draining = False  # in wait(): the whole drain pool takes figures
_cap_wait = None  # bytes of the figure save_figure is blocked on at the cap, else None
_state = threading.Condition(threading.RLock())
_log = logging.getLogger(__name__)


def _element_count(collection):
    if isinstance(collection, QuadMesh):
        return collection.get_coordinates().size // 2
    return max(len(collection.get_offsets()), len(collection.get_paths()))


def _block_mean(a, k):
    """Mean over k x k blocks (the last block per axis may be smaller), masked samples excluded."""
    starts = [np.arange(0, n, k) for n in a.shape[:2]]

    def block_sum(x):
        return np.add.reduceat(
            np.add.reduceat(x, starts[0], axis=0, dtype=np.float64), starts[1], axis=1
        )

    data = np.ma.getdata(a)
    if np.ma.getmask(a) is np.ma.nomask:
        sizes = [np.diff(np.append(s, n)) for s, n in zip(starts, a.shape[:2])]
        count = np.multiply.outer(*sizes).reshape(
            len(sizes[0]), len(sizes[1]), *([1] * (a.ndim - 2))
        )
        return _as_dtype(block_sum(data) / count, a.dtype)
    valid = ~np.ma.getmaskarray(a)
    count = block_sum(valid)
    mean = block_sum(np.where(valid, data, 0)) / np.maximum(count, 1)
    return np.ma.masked_array(_as_dtype(mean, a.dtype), mask=count == 0)


def _as_dtype(mean, dtype):
    """Block means in the image's dtype; integer (e.g. uint8 RGB) means are rounded, not truncated."""
    return (mean if np.issubdtype(dtype, np.floating) else np.rint(mean)).astype(dtype)


@njit(parallel=True)
def _colour_blocks(data, mask, has_mask, vmin, vmax, clip, lut, n_colours, k):
    """Colour each sample like Normalize + Colormap.__call__, then average k x k blocks of
    premultiplied colours (the last block per axis may be smaller); returns uint8 RGBA."""
    rows, cols = data.shape
    out_rows, out_cols = (rows + k - 1) // k, (cols + k - 1) // k
    out = np.empty((out_rows, out_cols, 4), np.uint8)
    i_under, i_over, i_bad = n_colours, n_colours + 1, n_colours + 2
    for block_row in prange(out_rows):
        acc = np.zeros((out_cols, 4))
        count = np.zeros(out_cols)
        for r in range(block_row * k, min(rows, block_row * k + k)):
            for c in range(cols):
                x = data[r, c]
                if (has_mask and mask[r, c]) or not np.isfinite(x):  # matplotlib masks non-finite
                    i = i_bad
                else:
                    v = 0.0 if vmin == vmax else (x - vmin) / (vmax - vmin)
                    if clip:
                        v = min(max(v, 0.0), 1.0)
                    v *= n_colours
                    if v == n_colours:
                        v = n_colours - 1
                    if v < 0:
                        i = i_under
                    elif v >= n_colours:
                        i = i_over
                    else:
                        i = int(v)
                j = c // k
                alpha = lut[i, 3]
                acc[j, 0] += lut[i, 0] * alpha
                acc[j, 1] += lut[i, 1] * alpha
                acc[j, 2] += lut[i, 2] * alpha
                acc[j, 3] += alpha
                count[j] += 1
        for j in range(out_cols):
            weight = acc[j, 3]
            for ch in range(3):
                out[block_row, j, ch] = int(acc[j, ch] / weight * 255 + 0.5) if weight > 0 else 0
            out[block_row, j, 3] = int(weight / count[j] * 255 + 0.5)
    return out


@njit(parallel=True)
def _finite_range(data):
    """(min, max) of the finite values of a 2-D array, rows in parallel: the vmin/vmax matplotlib's
    autoscale takes from the array safe_masked_invalid masks. (inf, -inf) when none is finite."""
    rows, cols = data.shape
    lows = np.full(rows, np.inf)
    highs = np.full(rows, -np.inf)
    for r in prange(rows):
        low, high = np.inf, -np.inf
        for c in range(cols):
            x = data[r, c]
            if np.isfinite(x):
                low = min(low, x)
                high = max(high, x)
        lows[r] = low
        highs[r] = high
    return lows.min(), highs.max()


def _parallel(kernel, *args):
    """kernel(*args) on the numba threads the figure workers leave free: the thread budget less
    the workers busy now, at least 1. The kernels here are row-parallel with row-local results,
    so the thread count never changes what they return."""
    if _executor is None:
        return kernel(*args)
    with _state:
        busy = len(_in_flight) + len(_reserved)
    previous = numba.get_num_threads()
    numba.set_num_threads(max(1, min(numba.config.NUMBA_NUM_THREADS, _budget - busy)))
    try:
        return kernel(*args)
    finally:
        numba.set_num_threads(previous)


def _numba_data(data):
    # numba takes integer data as it is (a 913 Mpx int8 mask is 7.3 GB as float64) and bool data
    # as its 0/1 bytes, but has no float16 / longdouble
    if data.dtype == bool:
        return data.view(np.uint8)
    return data.astype(np.float64) if data.dtype in (np.float16, np.longdouble) else data


def _colour(data, mask, cmap, vmin, vmax, clip, k):
    """Block-averaged uint8 RGBA of scalar `data` through Normalize(vmin, vmax, clip) and `cmap`;
    mask None: no sample is masked."""
    lut = np.vstack([cmap(np.arange(cmap.N)), [cmap.get_under(), cmap.get_over(), cmap.get_bad()]])
    has_mask = mask is not None
    return _parallel(_colour_blocks, _numba_data(data), mask if has_mask else np.zeros((1, 1), bool),
                     has_mask, float(vmin), float(vmax), bool(clip), lut, cmap.N, k)


def _colour_average(image, k):
    """Block-averaged uint8 RGBA of a scalar image, through its own norm and colormap."""
    a, norm = image.get_array(), image.norm
    mask = np.ma.getmaskarray(a) if np.ma.getmask(a) is not np.ma.nomask else None
    return _colour(np.ma.getdata(a), mask, image.get_cmap(), norm.vmin, norm.vmax, norm.clip, k)


def _output_dpi(fig, kwargs):
    dpi = kwargs.get("dpi", matplotlib.rcParams["savefig.dpi"])
    return fig.dpi if dpi == "figure" else dpi


def _block_size(image, shape, dpi):
    """k for the k x k blocks that bring a `shape` array shown as `image` down to about
    IMAGE_SAMPLES_PER_PIXEL samples per output pixel at `dpi` (k <= 1: it has no more)."""
    # Display extent of the whole image through its own transform (world or pixel units,
    # zoomed or translated alike), at the figure dpi; the output is at `dpi`.
    box = image.get_window_extent()
    scale = dpi / image.get_figure(root=True).dpi
    samples_per_pixel = min(
        shape[1] / (abs(box.width) * scale),
        shape[0] / (abs(box.height) * scale),
    )
    return int(round(samples_per_pixel / IMAGE_SAMPLES_PER_PIXEL, 6))  # display extents carry float noise


def _dense_images(fig, dpi):
    """[(image, k)] for the plain AxesImages of `fig` that k x k blocks (k > 1) would bring down
    to IMAGE_SAMPLES_PER_PIXEL samples per output pixel at `dpi`."""
    dense = []
    for image in fig.findobj(lambda artist: type(artist) is AxesImage):
        k = _block_size(image, image.get_array().shape, dpi)
        if k > 1:
            dense.append((image, k))
    return dense


def imshow(ax, data, *, dpi=None, cmap=None, **kwargs):
    """ax.imshow(data, cmap=cmap, **kwargs) for a scalar per-pixel image that will be saved at
    `dpi` (None: the savefig default), without matplotlib's full-size copy of a dense one.

    matplotlib's imshow copies the whole array on the calling thread (safe_masked_invalid: a
    copy, an isfinite mask and a masked array), which save_figure then reduces anyway. An image
    denser than IMAGE_SAMPLES_PER_PIXEL samples per output pixel of the axes as they are now (a
    colorbar added later only shrinks them) is coloured and block-averaged here instead, in
    parallel and straight from `data`, as save_figure reduces one: vmin/vmax are the finite range
    of the whole array (matplotlib's autoscale) and non-finite samples take the bad colour. The
    image keeps the colormap and a Normalize(vmin, vmax), so a colorbar shows the data's scale.
    Other images, masked arrays, RGB(A) data and explicit norms, limits or interpolation go to
    ax.imshow unchanged. Like plt.imshow, the image becomes the current one (plt.colorbar()).
    """
    image = _imshow(ax, data, dpi, cmap, kwargs)
    plt.sci(image)
    return image


def _imshow(ax, data, dpi, cmap, kwargs):
    if (
        np.ma.isMaskedArray(data)
        or np.ndim(data) != 2
        or {"norm", "vmin", "vmax", "interpolation", "interpolation_stage"} & kwargs.keys()
    ):
        return ax.imshow(data, cmap=cmap, **kwargs)
    data = np.asarray(data)
    if kwargs.get("extent") is None:  # imshow's own extent, for origin 'upper' or 'lower'
        rows, cols = data.shape
        lower = kwargs.get("origin", matplotlib.rcParams["image.origin"]) == "lower"
        kwargs["extent"] = (-0.5, cols - 0.5, *((-0.5, rows - 0.5) if lower else (rows - 0.5, -0.5)))
    # A 1 x 1 stand-in with the same extent gives the display box (and sets the axes limits).
    stand_in = ax.imshow(np.zeros((1, 1)), cmap=cmap, **kwargs)
    ax.apply_aspect()
    k = _block_size(stand_in, data.shape, _output_dpi(ax.figure, {} if dpi is None else {"dpi": dpi}))
    stand_in.remove()
    vmin, vmax = _parallel(_finite_range, _numba_data(data)) if k > 1 else (np.inf, -np.inf)
    if vmin > vmax:  # not dense, or nothing finite: matplotlib's own path
        return ax.imshow(data, cmap=cmap, **kwargs)
    image = ax.imshow(_colour(data, None, plt.get_cmap(cmap), vmin, vmax, False, k), **kwargs)
    image.set_cmap(cmap)
    image.set_norm(Normalize(vmin, vmax))
    return image


def _reduced_images(fig, dpi):
    """{id(per-pixel array): reduced copy} for images far denser than the output.

    Every per-pixel array of an image is reduced with the same blocks: the data (scalar data is
    coloured and colour-averaged, RGB(A) data is averaged; masked samples count as the bad
    colour or are excluded; integer and bool data are coloured like floats) and an array alpha.
    Extent (fixed by imshow), norm and clim are not
    per-pixel; the colorbar keeps using the image's norm and colormap. Only plain AxesImage:
    NonUniformImage/PcolorImage carry per-pixel coordinate arrays and are left alone.
    """
    reduced = {}
    for image, k in _dense_images(fig, dpi):
        a = image.get_array()
        if (
            image.get_interpolation() in ("nearest", "none")
            or image.get_interpolation_stage() == "data"  # colours of resampled values, not averages
        ):
            continue
        if a.ndim == 2 and type(image.norm) is not Normalize:
            continue
        # the norm keeps the vmin/vmax imshow took from the full array
        reduced[id(a)] = _colour_average(image, k) if a.ndim == 2 else _block_mean(a, k)
        alpha = image.get_alpha()
        if np.ndim(alpha) > 0:  # e.g. ovrlpy's signal-faded integrity map
            reduced[id(alpha)] = _block_mean(np.asarray(alpha, dtype=np.float64), k)
    return reduced


def _same(obj):
    return obj


class _SubstitutingPickler(pickle.Pickler):
    """Pickles `replacements[id(obj)]` in place of obj, leaving the caller's figure untouched."""

    def __init__(self, file, replacements):
        super().__init__(file, protocol=pickle.HIGHEST_PROTOCOL)
        self._replacements = replacements

    def reducer_override(self, obj):
        if id(obj) in self._replacements:
            return _same, (self._replacements[id(obj)],)
        return NotImplemented


def _pickle(fig, exact, kwargs):
    if exact or not isinstance(fig, Figure):
        return pickle.dumps(fig, protocol=pickle.HIGHEST_PROTOCOL)
    replacements = _reduced_images(fig, _output_dpi(fig, kwargs))
    if not replacements:
        return pickle.dumps(fig, protocol=pickle.HIGHEST_PROTOCOL)
    buffer = io.BytesIO()
    _SubstitutingPickler(buffer, replacements).dump(fig)
    return buffer.getvalue()


def _write(blob, rc, paths, exact, kwargs):
    fig = pickle.loads(blob)
    if rc is None:  # plotly
        for path in paths:
            fig.write_image(path, **kwargs)
        return
    # Unpickling a pyplot figure registers it with pyplot as the current figure: close it, or a
    # worker keeps every figure it renders, and without a pool the caller's plt.close() would
    # close this copy instead of its own figure.
    try:
        with matplotlib.rc_context(rc):
            for path in paths:
                if not exact and os.fspath(path).lower().endswith(".pdf"):
                    for collection in fig.findobj(Collection):
                        if _element_count(collection) >= RASTERIZE_MIN_ELEMENTS:
                            collection.set_rasterized(True)  # no effect on the Agg PNG
                fig.savefig(path, **kwargs)
    finally:
        plt.close(fig)


def save_figure(fig, *paths, exact=False, **kwargs):
    """Write a matplotlib or plotly `fig` to each of `paths` (format from the extension).

    `kwargs` go to every savefig / write_image call. With a pool, this returns once the figure is
    pickled and the files exist after wait(); the caller's figure is never modified.
    `exact=True` writes the figure as savefig would, without image reduction or PDF
    rasterising: use it for files that are read back as data.
    """
    global _cap_wait
    rc = dict(matplotlib.rcParams) if isinstance(fig, Figure) else None
    if rc is not None and _dense_images(fig, _output_dpi(fig, kwargs)):
        # A closed figure is a reference cycle holding its image arrays (8 B/px for float64) until
        # a full collection, which CPython rarely runs: on a 913 Mpx slide hqtr's closed figures
        # held ~58 GB. Free them before this dense figure is pickled.
        gc.collect()
    blob = _pickle(fig, exact, kwargs)
    if _executor is None:
        _write(blob, rc, paths, exact, kwargs)
        return
    with _state:
        _raise_worker_error()
        # Over the cap the main thread waits, so the drain pool takes held figures too (_reserve).
        _cap_wait = len(blob)
        picks = _reserve()
    try:
        _submit(picks)
        with _state:
            while _blocked(len(blob)) and not _errors:
                _state.wait()
            _raise_worker_error()
    finally:
        with _state:
            _cap_wait = None
    with _state:
        _held.append(((blob, rc, paths, exact, kwargs), len(blob)))
        picks = _reserve()
    _submit(picks)


def _over_cap(nbytes):
    """Under _state: whether adding `nbytes` would take the pending figures past the cap."""
    return bool(_held or _in_flight) and _pending_bytes() + nbytes > MAX_PENDING_BYTES


def _blocked(nbytes):
    """Under _state: whether save_figure of `nbytes` must wait: past the cap, or every drain
    worker still busy after a cap wait, which would leave the main thread no core."""
    drain_busy = sum(pool is _drain for pool, _ in (*_in_flight.values(), *_reserved.values()))
    return _over_cap(nbytes) or (_drain is not None and drain_busy >= _workers[_drain])


def _drain_slots():
    """Under _state: how many drain workers may be busy now (see the CPU budget note)."""
    if _drain is None:
        return 0
    if _draining or (_cap_wait is not None and _over_cap(_cap_wait)):
        return _workers[_drain]
    return 0


def _pending_bytes():
    return sum(n for _, n in _held) + sum(n for _, n in (*_in_flight.values(), *_reserved.values()))


def _log_errors(errors, what):
    for error in errors:
        _log.error("figure write failed (%s)", what, exc_info=error)


def _raise_worker_error():
    """Raise the first worker error; log the others, which would otherwise be lost."""
    if _errors:
        first, rest = _errors[0], _errors[1:]
        _errors.clear()
        _held.clear()  # the run fails: queued figures are not written
        _log_errors(rest, "not re-raised: an earlier figure failed first")
        raise first


def _reserve():
    """Under _state: take the held figures that fit an idle worker and reserve its slot (the
    drain pool only while the main thread is blocked: _drain_slots). Nothing is taken once a
    write has failed.

    Returns (token, pool, item) for _submit(), which the caller runs after releasing _state.
    """
    picks = []
    while _held and not _errors:
        busy = collections.Counter(pool for pool, _ in (*_in_flight.values(), *_reserved.values()))
        slots = {_executor: _workers[_executor], _drain: _drain_slots()}
        pool = next((p for p, n in slots.items() if p is not None and busy[p] < n), None)
        if pool is None:
            break
        item = _held.popleft()
        token = object()
        _reserved[token] = (pool, item[1])
        picks.append((token, pool, item))
    return picks


def _submit(picks):
    """Submit reserved figures. Must run WITHOUT _state held.

    submit() takes the pool's shutdown lock, and a breaking pool holds that lock while its
    futures' callbacks (_finished) take _state: holding _state here would invert that order.
    A submit that fails puts its figure and the ones after it back, so no reservation leaks,
    and records the error. A BaseException (e.g. KeyboardInterrupt) is re-raised instead of
    recorded; a later wait() submits those figures again.
    """
    for n, (token, pool, (args, nbytes)) in enumerate(picks):
        try:
            future = pool.submit(_write, *args)
        except BaseException as error:  # e.g. BrokenProcessPool, KeyboardInterrupt
            with _state:
                for t, _, item in reversed(picks[n:]):
                    del _reserved[t]
                    _held.appendleft(item)
                _state.notify_all()
                if not isinstance(error, Exception):
                    raise
                _errors.append(error)
            return
        with _state:
            del _reserved[token]
            _in_flight[future] = (pool, nbytes)
        future.add_done_callback(_finished)


def _finished(future):
    """Done-callback, on a pool's manager thread (or the submitting thread, if already done).

    A pool that breaks runs this for its futures while holding its own shutdown lock, so this
    never calls submit() then: only a successful write hands out the next figure, and that
    submit happens after _state is released. Failed and cancelled futures record and notify.
    """
    picks = []
    with _state:
        try:
            del _in_flight[future]
            if future.cancelled():
                return
            if future.exception() is not None:
                _errors.append(future.exception())
                return
            picks = _reserve()
        finally:
            _state.notify_all()
    _submit(picks)


def _pool(workers):
    # figure_worker.init runs each worker single-threaded and ends it when this process dies.
    pool = ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=figure_worker.init,
        initargs=(os.getpid(),),
    )
    _workers[pool] = workers
    return pool


def start(threads_budget):
    """Open the pools for save_figure(); `threads_budget` is the run's thread count.

    The drain pool's processes are started now, each with a no-op task from this module so it
    imports matplotlib and the rest here, and wait() does not pay that start-up; they idle,
    without figure data, until the main thread waits.
    """
    global _executor, _drain, _budget
    _budget = threads_budget
    background = max(1, threads_budget // THREADS_PER_FIGURE_WORKER)
    _executor = _pool(background)
    if threads_budget > background:
        _drain = _pool(threads_budget - background)
        for _ in range(threads_budget - background):
            _drain.submit(_same, None)


def wait():
    """Block until every submitted figure is written; re-raise the first worker error.

    The main thread is idle meanwhile, so held figures also go to the whole drain pool, and the
    figure pools use the whole thread budget.
    """
    global _draining
    with _state:
        _draining = True
        picks = _reserve()
    try:
        _submit(picks)
        with _state:
            while _in_flight or _reserved or (_held and not _errors):
                _state.wait()
    finally:
        with _state:
            _draining = False
    with _state:
        _raise_worker_error()


def _close(cancel):
    global _executor, _drain
    pools = [p for p in (_executor, _drain) if p is not None]
    _executor = _drain = None
    for pool in pools:  # outside _state: shutdown joins the threads that run _finished
        pool.shutdown(cancel_futures=cancel)
        del _workers[pool]


def stop():
    """wait(), then shut the pools down."""
    try:
        wait()
    finally:
        _close(cancel=False)


def abort():
    """On the error path: drop queued writes, let running ones finish, close the pools.

    Errors of those writes are logged, not kept for a later start().
    """
    with _state:
        _held.clear()
        dropped = list(_errors)
        _errors.clear()
    _close(cancel=True)
    with _state:
        dropped += _errors
        _errors.clear()
    _log_errors(dropped, "while aborting")
