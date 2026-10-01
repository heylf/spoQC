import gc
import io
import os
import pickle
import subprocess
import sys
import threading
import time
import types
import weakref
from concurrent.futures import Future
from concurrent.futures.process import BrokenProcessPool

import matplotlib
import matplotlib.pyplot as plt
import numba
import numpy as np
import pandas as pd
import plotly.express as px
import pytest
from matplotlib.transforms import Affine2D
from anndata import AnnData

from spoqc import helperfuncs
from spoqc.core import figure_worker, figures
from spoqc.core.figures import save_figure

WORKERS = 2


@pytest.fixture
def pool():
    figures.start(WORKERS)
    yield figures
    figures.stop()


@pytest.fixture(autouse=True)
def _close_figures():
    """Close every pyplot figure a test leaves open: later code draws on the current figure
    (plt.gcf(), seaborn), and a leftover one, e.g. an extent-less NonUniformImage, breaks it."""
    yield
    plt.close("all")


def _until(condition, timeout=60):
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.05)


def _scatter(n=100, seed=0):
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots()
    ax.scatter(rng.random(n), rng.random(n), s=1)
    return fig


def _adata(n=500, seed=0):
    rng = np.random.default_rng(seed)
    adata = AnnData(np.zeros((n, 1), dtype=np.float32))
    adata.obsm["spatial"] = rng.random((n, 2)) * 1000
    adata.obs["celltype"] = pd.Categorical(rng.choice(["a", "b", "c"], n))
    adata.obs["score"] = rng.random(n)
    return adata


class TestWritesFiles:
    def test_matplotlib_figure_is_written_to_every_path(self, pool, tmp_path):
        fig = _scatter()
        save_figure(
            fig, tmp_path / "a.png", f"{tmp_path}/a.pdf", dpi=50, bbox_inches="tight"
        )
        plt.close(fig)
        pool.wait()
        assert (tmp_path / "a.png").read_bytes().startswith(b"\x89PNG")
        assert (tmp_path / "a.pdf").read_bytes().startswith(b"%PDF")

    def test_plotly_figure_is_written_to_every_path(self, pool, tmp_path):
        fig = px.scatter(x=[1, 2, 3], y=[3, 1, 2])
        save_figure(fig, f"{tmp_path}/p.png", f"{tmp_path}/p.pdf", scale=1)
        pool.wait()
        assert (tmp_path / "p.png").read_bytes().startswith(b"\x89PNG")
        assert (tmp_path / "p.pdf").read_bytes().startswith(b"%PDF")

    def test_many_figures_beyond_the_pending_bound_all_land(self, pool, tmp_path, monkeypatch):
        monkeypatch.setattr(figures, "MAX_PENDING_BYTES", 1)
        n = WORKERS * 6
        for i in range(n):
            fig = _scatter(seed=i)
            save_figure(fig, tmp_path / f"{i}.png", dpi=20)
            plt.close(fig)
        pool.wait()
        assert sorted(os.listdir(tmp_path)) == sorted(f"{i}.png" for i in range(n))

    def test_figure_changed_after_save_is_written_as_it_was_saved(self, pool, tmp_path):
        fig = _scatter()
        fig.axes[0].set_title("saved")
        with plt.rc_context({"svg.fonttype": "none"}):  # text as text; also checks rcParams reach the worker
            save_figure(fig, tmp_path / "a.svg")
        fig.axes[0].set_title("changed afterwards")
        pool.wait()
        svg = (tmp_path / "a.svg").read_text()
        assert "saved" in svg and "changed afterwards" not in svg

    def test_real_site_plot_scatter_writes_png_and_pdf(self, pool, tmp_path):
        helperfuncs.plot_scatter(
            _adata(), str(tmp_path), "x", None, "celltype", None, "title"
        )
        pool.wait()
        for ext in ["png", "pdf"]:
            assert (tmp_path / f"scatterplot_x.{ext}").stat().st_size > 0


class TestErrors:
    def test_worker_error_is_raised_by_wait(self, pool, tmp_path):
        save_figure(_scatter(), tmp_path / "missing_dir" / "a.png")
        with pytest.raises(FileNotFoundError):
            pool.wait()

    def test_worker_error_is_raised_by_the_next_save(self, pool, tmp_path):
        save_figure(_scatter(), tmp_path / "missing_dir" / "a.png")
        _until(lambda: figures._errors)  # the failing write has finished
        with pytest.raises(FileNotFoundError):
            save_figure(_scatter(), tmp_path / "b.png")

    def test_worker_error_is_raised_by_stop(self, tmp_path):
        figures.start(WORKERS)
        save_figure(
            px.scatter(x=[1, 2], y=[1, 2]), f"{tmp_path}/p.png", format="no-such-format"
        )
        with pytest.raises(ValueError):
            figures.stop()


class TestDataUnchanged:
    def test_obs_and_obsm_are_identical_after_plotting(self, pool, tmp_path):
        adata = _adata()
        obs, spatial = adata.obs.copy(), adata.obsm["spatial"].copy()
        helperfuncs.plot_scatter(
            adata, str(tmp_path), "x", None, "celltype", None, None
        )
        helperfuncs.plot_scatter_density(
            adata, str(tmp_path), "y", "celltype", "score", None, None
        )
        pool.wait()
        pd.testing.assert_frame_equal(adata.obs, obs, check_exact=True)
        np.testing.assert_array_equal(adata.obsm["spatial"], spatial)

    def test_large_collection_is_rasterised_only_in_the_worker_pdf(
        self, pool, tmp_path
    ):
        fig = _scatter(n=figures.RASTERIZE_MIN_ELEMENTS)
        save_figure(fig, tmp_path / "big.pdf")
        pool.wait()
        assert b"/Subtype /Image" in (tmp_path / "big.pdf").read_bytes()
        assert not fig.axes[0].collections[0].get_rasterized()

    def test_small_collection_stays_vector_in_pdf(self, pool, tmp_path):
        save_figure(_scatter(n=100), tmp_path / "small.pdf")
        pool.wait()
        assert b"/Subtype /Image" not in (tmp_path / "small.pdf").read_bytes()


class TestWorkers:
    def test_worker_count_is_respected(self, pool, tmp_path):
        for i in range(WORKERS * 4):
            fig = _scatter(seed=i)
            save_figure(fig, tmp_path / f"{i}.png", dpi=20)
            plt.close(fig)
        pool.wait()
        background = max(1, WORKERS // figures.THREADS_PER_FIGURE_WORKER)
        assert pool._executor._max_workers == background
        assert 0 < len(pool._executor._processes) <= background
        assert background + figures._workers[figures._drain] == WORKERS


class TestPyplotState:
    def test_worker_closes_every_figure_it_renders(self, pool, tmp_path):
        for i in range(5):
            fig = _scatter(seed=i)
            save_figure(fig, tmp_path / f"{i}.png", dpi=20)
            plt.close(fig)
        _until(lambda: not figures._held and not figures._in_flight)  # all on the background worker
        assert figures._workers[figures._executor] == 1
        assert figures._executor.submit(plt.get_fignums).result() == []

    def test_without_a_pool_plt_close_closes_the_callers_figure(self, tmp_path):
        plt.close("all")
        fig = _scatter()
        save_figure(fig, tmp_path / "a.png", dpi=20)
        assert plt.gcf() is fig  # the written copy is closed and fig is current again
        plt.close()
        assert plt.get_fignums() == []


class TestImageReduction:
    """Output: 2 in x 2 in figure at 50 dpi, axes filling it, so the image covers 100 x 100 px."""

    def _imshow(self, data, **kwargs):
        fig = plt.figure(figsize=(2, 2), dpi=50)
        ax = fig.add_axes((0, 0, 1, 1))
        ax.imshow(data, **kwargs)
        return fig, ax

    def _reduced_shape(self, fig):
        reduced = figures._reduced_images(fig, 50)
        return None if not reduced else next(iter(reduced.values())).shape[:2]

    @staticmethod
    def _side(samples_per_pixel, n=4000):
        """Reduced side for `n` samples at `samples_per_pixel` per output pixel."""
        k = int(samples_per_pixel / figures.IMAGE_SAMPLES_PER_PIXEL)
        return -(-n // k)

    def test_float_image_is_block_averaged_to_the_output_density(self):
        fig, _ = self._imshow(np.zeros((4000, 4000), dtype=np.float32))
        # 40 samples per output pixel
        assert self._reduced_shape(fig) == (self._side(40),) * 2

    def test_one_pixel_lines_survive_block_averaging(self):
        data = np.zeros((4000, 4000), dtype=np.float32)
        data[::97, :] = 1.0  # 1-px lines, never aligned to a block
        fig, _ = self._imshow(data)
        reduced = next(iter(figures._reduced_images(fig, 50).values())).astype(int)
        background = reduced[-1, -1]  # a block with no line (4000 - 1 is not a multiple of 97)
        rows_with_line = (np.abs(reduced - background).sum(axis=-1) > 0).any(axis=1)
        assert rows_with_line.sum() == len(range(0, 4000, 97))

    def _png(self, fig, exact):
        buffer = io.BytesIO()
        pickle.loads(figures._pickle(fig, exact, {"dpi": 50})).savefig(buffer, format="png", dpi=50)
        buffer.seek(0)
        return plt.imread(buffer)

    @pytest.mark.parametrize("dtype", [np.uint8, np.int8, np.int32, bool])
    def test_integer_and_bool_images_are_reduced_to_rgba(self, dtype):
        data = np.zeros((4000, 4000), dtype=dtype)
        data[::97, :] = 1
        fig, _ = self._imshow(data, cmap="gray")
        reduced = next(iter(figures._reduced_images(fig, 50).values()))
        assert reduced.dtype == np.uint8 and reduced.shape == (self._side(40), self._side(40), 4)

    def test_integer_image_reduction_blends_label_colours_not_labels(self):
        # labels 0 and 2 side by side in every block: tab10's colours 0 and 2 averaged, not colour 1
        data = np.zeros((4000, 4000), dtype=np.int32)
        data[:, 1::2] = 2
        fig, ax = self._imshow(data, cmap="tab10")
        reduced = next(iter(figures._reduced_images(fig, 50).values()))
        colours = ax.images[0].to_rgba(np.array([[0, 2]]))[0]
        np.testing.assert_allclose(reduced[5, 5] / 255, colours.mean(axis=0), atol=1 / 255)

    @pytest.mark.parametrize("cmap", ["gray", "tab10"])
    def test_reduced_integer_image_renders_as_the_full_image_up_to_resampling(self, cmap):
        # Exactly the resampling difference the float reduction has. Random 10-px labels are the
        # worst case for it: at 2 samples/px measured 10.6/255 mean (gray) and 8.5/255 (tab10).
        rng = np.random.default_rng(0)
        labels = np.repeat(np.repeat(rng.integers(0, 10, (400, 400)), 10, 0), 10, 1)
        errors = []
        for data in (labels, labels.astype(np.float32)):
            fig, _ = self._imshow(data, cmap=cmap)
            errors.append(np.abs(self._png(fig, exact=False) - self._png(fig, exact=True)).mean())
            plt.close(fig)
        assert errors[0] == pytest.approx(errors[1], abs=1e-6) and errors[0] < 12 / 255, np.array(errors) * 255

    def test_integer_image_with_array_alpha_is_reduced_with_its_alpha_like_a_float_image(self):
        data = np.zeros((4000, 4000), dtype=np.uint8)
        alpha = np.ones((4000, 4000))
        fig, _ = self._imshow(data, alpha=alpha)
        reduced = figures._reduced_images(fig, 50)
        assert {k: v.shape[:2] for k, v in reduced.items()} == {
            id(fig.axes[0].images[0].get_array()): (self._side(40),) * 2,
            id(alpha): (self._side(40),) * 2,
        }

    def test_integer_image_at_the_data_interpolation_stage_is_not_reduced(self):
        fig, _ = self._imshow(np.zeros((4000, 4000), dtype=np.uint8), interpolation_stage="data")
        assert self._reduced_shape(fig) is None

    def test_integer_rgb_block_means_are_rounded(self):
        rgb = np.array([[0, 1], [1, 1]], dtype=np.uint8)[..., None].repeat(3, axis=2)
        assert figures._block_mean(rgb, 2).tolist() == [[[1, 1, 1]]]  # 0.75, not truncated to 0

    @pytest.mark.parametrize("exact", [False, True])
    def test_dead_dense_figures_do_not_accumulate_across_saves(self, tmp_path, exact):
        # exact figures count too: qc_wsi's exact figure is dense
        plt.close("all")
        images = []
        gc.disable()  # only save_figure may collect
        try:
            for i in range(4):
                fig, ax = self._imshow(np.zeros((4000, 4000), dtype=np.float32))
                images.append(weakref.ref(ax.images[0]))
                save_figure(fig, tmp_path / f"{i}.png", dpi=50, exact=exact)
                plt.close(fig)
                del fig, ax
            alive = [i for i, image in enumerate(images) if image() is not None]
        finally:
            gc.enable()
        assert alive == [3]  # a closed figure is a reference cycle until the next dense save

    @pytest.mark.parametrize("interpolation", ["nearest", "none"])
    def test_nearest_and_none_interpolation_are_never_reduced(self, interpolation):
        fig, _ = self._imshow(np.zeros((4000, 4000), dtype=np.float32), interpolation=interpolation)
        assert self._reduced_shape(fig) is None

    def test_world_unit_extent_translated_far_from_origin(self):
        fig, _ = self._imshow(np.zeros((4000, 4000), dtype=np.float32), extent=(1e5, 1e5 + 50, 2e5, 2e5 + 50))
        assert self._reduced_shape(fig) == (self._side(40),) * 2

    def test_image_with_its_own_transform(self):
        # spatialdata-plot style: pixel-unit extent, world units via the image transform
        fig = plt.figure(figsize=(2, 2), dpi=50)
        ax = fig.add_axes((0, 0, 1, 1))
        ax.imshow(np.zeros((4000, 4000), dtype=np.float32),
                  transform=Affine2D().scale(0.25).translate(300, 700) + ax.transData)
        ax.set_xlim(300, 1300)
        ax.set_ylim(1700, 700)
        assert self._reduced_shape(fig) == (self._side(40),) * 2

    def test_zoomed_in_image_is_reduced_less(self):
        fig, ax = self._imshow(np.zeros((4000, 4000), dtype=np.float32))
        ax.set_xlim(0, 1000)  # 4x zoom: 10 samples per output pixel
        ax.set_ylim(1000, 0)
        assert self._reduced_shape(fig) == (self._side(10),) * 2

    def test_strongly_zoomed_image_keeps_full_resolution(self):
        fig, ax = self._imshow(np.zeros((4000, 4000), dtype=np.float32))
        ax.set_xlim(0, 50)
        ax.set_ylim(50, 0)
        assert self._reduced_shape(fig) is None

    def test_masked_samples_are_excluded_from_block_means(self):
        data = np.ma.masked_array(np.full((8, 8), 2.0), mask=np.zeros((8, 8), bool))
        data[:4, :4] = np.ma.masked
        data[4:, 4:] = 6.0
        reduced = figures._block_mean(data, 4)
        assert reduced.mask.tolist() == [[True, False], [False, False]]
        assert reduced[1, 1] == 6.0

    def test_callers_figure_is_not_modified_and_colour_scale_is_kept(self, pool, tmp_path):
        data = np.zeros((4000, 4000), dtype=np.float32)
        data[1, 1] = 7.0
        fig, ax = self._imshow(data, cmap="viridis")
        image = ax.images[0]
        before = image.get_array()
        save_figure(fig, tmp_path / "img.png", tmp_path / "img.pdf", dpi=50)
        pool.wait()
        assert image.get_array() is before and before.shape == (4000, 4000)
        assert (image.norm.vmin, image.norm.vmax) == (0.0, 7.0)
        assert (tmp_path / "img.png").read_bytes().startswith(b"\x89PNG")
        assert (tmp_path / "img.pdf").read_bytes().startswith(b"%PDF")

    def test_exact_writes_what_savefig_writes(self, pool, tmp_path):
        rng = np.random.default_rng(0)
        fig, ax = self._imshow(rng.random((2000, 2000)).astype(np.float32))
        ax.scatter(rng.random(figures.RASTERIZE_MIN_ELEMENTS) * 2000, rng.random(figures.RASTERIZE_MIN_ELEMENTS) * 2000, s=1)
        metadata = {"Software": None}
        save_figure(fig, tmp_path / "exact.png", exact=True, metadata=metadata)
        save_figure(fig, tmp_path / "exact.pdf", exact=True)
        pool.wait()
        fig.savefig(tmp_path / "direct.png", metadata=metadata)
        assert (tmp_path / "exact.png").read_bytes() == (tmp_path / "direct.png").read_bytes()
        assert b"/Subtype /Image" in (tmp_path / "exact.pdf").read_bytes()  # the imshow itself
        assert (tmp_path / "exact.pdf").read_bytes().count(b"/Subtype /Image") == 1  # scatter stays vector


class TestReadBacks:
    def test_sort_files_moves_every_figure_even_while_writes_are_in_flight(self, pool, tmp_path):
        names = [f"{prefix}_{i}.png" for prefix in ("umap", "barplot", "violin") for i in range(4)]
        for name in names:
            fig = _scatter(n=20_000, seed=len(name))
            save_figure(fig, tmp_path / name, dpi=150)
            plt.close(fig)
        helperfuncs.sort_files(str(tmp_path), "prefix", ["res.txt", "done.txt"])
        assert sorted(p.name for p in tmp_path.iterdir()) == ["barplot", "umap", "violin"]
        assert sorted(p.name for p in tmp_path.glob("*/*.png")) == sorted(names)

    def test_stripe_thickness_reads_the_figure_after_it_lands_and_matches_serial(self, tmp_path):
        from spoqc.subworkflows import qc_wsi

        def thickness(out):
            out.mkdir()
            rng = np.random.default_rng(0)
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(rng.random((600, 600)).astype(np.float32) > 0.7, cmap="viridis")
            ax.axis("off")
            save_figure(fig, out / "input.png", exact=True, bbox_inches="tight")
            plt.close(fig)
            return qc_wsi.measure_stripe_thickness_and_black_area(str(out / "input.png"), np.array([68, 1, 84]), str(out))

        serial = thickness(tmp_path / "serial")
        figures.start(WORKERS)
        try:
            pooled = thickness(tmp_path / "pooled")
        finally:
            figures.stop()
        assert pooled == serial


class TestPoolLifecycle:
    def test_without_start_figures_are_written_before_save_returns(self, tmp_path):
        assert figures._executor is None
        save_figure(_scatter(), tmp_path / "sync.png")
        assert (tmp_path / "sync.png").read_bytes().startswith(b"\x89PNG")

    def test_without_start_worker_errors_raise_at_the_call(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            save_figure(_scatter(), tmp_path / "missing_dir" / "a.png")

    def test_abort_cancels_queued_writes_and_closes_the_pool(self, tmp_path):
        figures.start(1)
        for i in range(12):
            save_figure(_scatter(n=50_000, seed=i), tmp_path / f"{i}.png", dpi=200)
        figures.abort()
        assert figures._executor is None and not figures._held and not figures._in_flight
        assert len(list(tmp_path.glob("*.png"))) < 12

    def test_pending_bytes_stay_under_the_cap(self, pool, tmp_path, monkeypatch):
        blob = len(pickle.dumps(_scatter(n=5_000), protocol=pickle.HIGHEST_PROTOCOL))
        monkeypatch.setattr(figures, "MAX_PENDING_BYTES", 3 * blob)
        for i in range(20):
            save_figure(_scatter(n=5_000, seed=i), tmp_path / f"{i}.png", dpi=50)
            assert figures._pending_bytes() <= 3 * blob
        pool.wait()
        assert len(list(tmp_path.glob("*.png"))) == 20


class TestCli:
    def _main(self, monkeypatch, tmp_path, run, calls):
        from spoqc import cli
        from spoqc.cli_args import build_parser
        from spoqc.core import threads

        # the entry point would have run threads.configure(12); numpy is already loaded here
        monkeypatch.setattr(threads, "N", 12)
        monkeypatch.setattr(cli.figures, "start", lambda n: calls.append(("start", n)))
        monkeypatch.setattr(cli.figures, "stop", lambda: calls.append(("stop",)))
        monkeypatch.setattr(cli.figures, "abort", lambda: calls.append(("abort",)))
        monkeypatch.setattr(cli, "run", run)
        cli.main(build_parser().parse_args(["-i", str(tmp_path), "-o", str(tmp_path), "-t", str(tmp_path), "-n", "12"]))

    def test_pool_gets_the_thread_budget_and_is_stopped(self, monkeypatch, tmp_path):
        calls = []
        self._main(monkeypatch, tmp_path, lambda CONST: None, calls)
        assert calls == [("start", 12), ("stop",)]

    def test_a_failing_step_aborts_the_pool_and_propagates(self, monkeypatch, tmp_path):
        def run(CONST):
            raise RuntimeError("step failed")

        calls = []
        with pytest.raises(RuntimeError, match="step failed"):
            self._main(monkeypatch, tmp_path, run, calls)
        assert calls == [("start", 12), ("abort",)]


def _png(path):
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGBA")).astype(int)


class TestPerPixelAttributes:
    """Everything per-pixel must be reduced with the data, or the worker's render breaks."""

    SHAPE = (3000, 2000)

    def _smooth(self, channels=None):
        yy, xx = np.mgrid[0 : self.SHAPE[0], 0 : self.SHAPE[1]]
        base = (np.sin(xx / 300) * np.cos(yy / 400) + 1) / 2
        return base if channels is None else np.stack([base ** (i + 1) for i in range(channels)], axis=-1)

    def _reduced_vs_exact(self, tmp_path, draw):
        """Save the same figure reduced and exact; return (number of reduced arrays, mean |diff|)."""
        fig = plt.figure(figsize=(3, 2), dpi=100)
        ax = fig.add_axes((0.1, 0.1, 0.8, 0.8))
        draw(ax)
        n_reduced = len(figures._reduced_images(fig, 100))
        save_figure(fig, tmp_path / "reduced.png", tmp_path / "reduced.pdf")
        save_figure(fig, tmp_path / "exact.png", exact=True)
        plt.close(fig)
        assert (tmp_path / "reduced.pdf").read_bytes().startswith(b"%PDF")
        return n_reduced, np.abs(_png(tmp_path / "reduced.png") - _png(tmp_path / "exact.png")).mean()

    def test_array_alpha_is_reduced_with_the_data(self, tmp_path):
        # ovrlpy._plot_signal_integrity: imshow(integrity, alpha=(signal / t).clip(0, 1) ** 2)
        alpha = self._smooth() ** 2
        n, diff = self._reduced_vs_exact(
            tmp_path, lambda ax: ax.imshow(self._smooth(), alpha=alpha, vmin=0, vmax=1, origin="lower")
        )
        assert n == 2 and diff < 2

    def test_array_alpha_with_a_pool_writes_png_and_pdf_and_leaves_the_caller_alone(self, pool, tmp_path):
        fig, ax = plt.subplots(figsize=(3, 2), dpi=100)
        alpha = self._smooth()
        image = ax.imshow(self._smooth().astype(np.float32), alpha=alpha)
        save_figure(fig, tmp_path / "a.png", tmp_path / "a.pdf")
        pool.wait()
        assert image.get_alpha() is alpha and image.get_array().shape == self.SHAPE
        assert (tmp_path / "a.png").read_bytes().startswith(b"\x89PNG")
        assert (tmp_path / "a.pdf").read_bytes().startswith(b"%PDF")

    @pytest.mark.parametrize("channels", [3, 4])
    def test_rgb_and_rgba_float_images(self, tmp_path, channels):
        n, diff = self._reduced_vs_exact(tmp_path, lambda ax: ax.imshow(self._smooth(channels)))
        assert n == 1 and diff < 2

    def test_rgba_float_image_with_array_alpha(self, tmp_path):
        n, diff = self._reduced_vs_exact(tmp_path, lambda ax: ax.imshow(self._smooth(4), alpha=self._smooth()))
        assert n == 2 and diff < 2

    def test_default_extent_and_clim_keep_their_place(self, tmp_path):
        def draw(ax):
            ax.imshow(self._smooth(), cmap="magma").set_clim(0.2, 0.8)

        n, diff = self._reduced_vs_exact(tmp_path, draw)
        assert n == 1 and diff < 2

    def test_masked_image(self, tmp_path):
        data = np.ma.masked_less(self._smooth(), 0.3)
        n, diff = self._reduced_vs_exact(tmp_path, lambda ax: ax.imshow(data))
        assert n == 1 and diff < 3

    def test_non_uniform_image_is_never_reduced(self):
        from matplotlib.image import NonUniformImage

        fig, ax = plt.subplots(figsize=(3, 2), dpi=100)
        image = NonUniformImage(ax)
        image.set_data(np.linspace(0, 1, 2000), np.linspace(0, 1, 3000), self._smooth())
        ax.add_image(image)
        assert figures._reduced_images(fig, 100) == {}


@pytest.fixture(scope="module")
def small_ovrlp():
    """A real ovrlpy analysis on 60k synthetic transcripts over 1500 x 1500 um (about 10 s)."""
    ovrlpy = pytest.importorskip("ovrlpy")
    rng = np.random.default_rng(0)
    n, side = 60_000, 1500.0
    x, y = rng.random(n) * side, rng.random(n) * side
    domain = (x // 500).astype(int) + 3 * (y // 500).astype(int)
    genes = np.array([f"g{i}" for i in range(12)])
    df = pd.DataFrame({"gene": genes[(domain % 4) * 3 + rng.integers(0, 3, n)], "x": x, "y": y, "z": rng.normal(5, 1, n)})
    ovrlp = ovrlpy.Ovrlp(df, min_distance=8, n_components=3, n_workers=2, random_state=0)
    ovrlp.analyse()
    return ovrlpy, ovrlp


class TestOvrlpyFigures:
    def test_region_of_interest_with_a_dense_integrity_map_is_written(self, pool, small_ovrlp, tmp_path):
        # Regression: doublet_score.py doublet_case_*_zoomed crashed in the worker with
        # "operands could not be broadcast" because the array alpha was not reduced.
        ovrlpy, ovrlp = small_ovrlp
        fig = ovrlpy.plot_region_of_interest(ovrlp, 750, 750, window_size=750, figsize=(6, 4))
        assert len(figures._reduced_images(fig, 100)) == 2  # the integrity data and its alpha
        save_figure(fig, tmp_path / "roi.png", tmp_path / "roi.pdf")
        pool.wait()
        assert (tmp_path / "roi.png").read_bytes().startswith(b"\x89PNG")
        assert (tmp_path / "roi.pdf").read_bytes().startswith(b"%PDF")

    def test_signal_integrity_map_is_written(self, pool, small_ovrlp, tmp_path):
        ovrlpy, ovrlp = small_ovrlp
        fig = ovrlpy.plot_signal_integrity(ovrlp, signal_threshold=2)
        save_figure(fig, tmp_path / "map.png", tmp_path / "map.pdf")
        pool.wait()
        assert (tmp_path / "map.png").read_bytes().startswith(b"\x89PNG")
        assert (tmp_path / "map.pdf").read_bytes().startswith(b"%PDF")


def _worker_thread_pools():
    import cv2
    import numba
    import polars
    from threadpoolctl import threadpool_info

    from spoqc.core import threads

    return {
        "N": threads.N,
        "polars": polars.thread_pool_size(),
        "numba": numba.config.NUMBA_NUM_THREADS,
        "cv2": cv2.getNumThreads(),
        "native": sorted({p["num_threads"] for p in threadpool_info()}),
    }


def test_figure_workers_run_single_threaded(pool):
    report = pool._executor.submit(_worker_thread_pools).result()
    assert report == {"N": 1, "polars": 1, "numba": 1, "cv2": 1, "native": [1]}


class TestColourAverage:
    def test_kernel_matches_matplotlib_colours_averaged_premultiplied(self):
        rng = np.random.default_rng(0)
        data = np.ma.masked_array(rng.normal(0.5, 0.4, (37, 53)), mask=rng.random((37, 53)) < 0.1)
        cmap = plt.get_cmap("hot").with_extremes(under="blue", over="green", bad=(1, 0, 0, 0.5))
        for clip in (False, True):
            fig, ax = plt.subplots()
            image = ax.imshow(data, cmap=cmap, norm=matplotlib.colors.Normalize(0.1, 0.9, clip=clip))
            got = figures._colour_average(image, 4).astype(float) / 255
            rgba = image.to_rgba(data)  # matplotlib's own colours, bad/under/over included
            rgba[..., :3] *= rgba[..., 3:]
            starts = [np.arange(0, n, 4) for n in data.shape]
            sums = np.add.reduceat(np.add.reduceat(rgba, starts[0], axis=0), starts[1], axis=1)
            counts = np.multiply.outer(*[np.diff(np.append(s, n)) for s, n in zip(starts, data.shape)])
            want = np.concatenate([sums[..., :3] / sums[..., 3:], (sums[..., 3] / counts)[..., None]], axis=-1)
            np.testing.assert_allclose(got, want, atol=0.5 / 255 + 1e-9)
            plt.close(fig)

    def test_non_linear_norm_is_never_reduced(self):
        fig = plt.figure(figsize=(2, 2), dpi=50)
        ax = fig.add_axes((0, 0, 1, 1))
        ax.imshow(np.random.default_rng(0).random((4000, 4000)) + 0.1, norm=matplotlib.colors.LogNorm())
        assert figures._reduced_images(fig, 50) == {}

    def test_noisy_texture_looks_like_the_full_resolution_render(self, tmp_path):
        # LBP-like per-pixel noise at the full-scale hqtr density (18 samples per output pixel):
        # averaging values would draw it as one flat mid colour; averaging colours does not.
        codes = np.random.default_rng(0).integers(0, 102, (1800, 1800)).astype(np.float64)
        fig = plt.figure(figsize=(2, 2), dpi=50)
        ax = fig.add_axes((0, 0, 1, 1))
        ax.imshow(codes, cmap="hot")
        save_figure(fig, tmp_path / "reduced.png")
        save_figure(fig, tmp_path / "exact.png", exact=True)
        plt.close(fig)
        assert np.abs(_png(tmp_path / "reduced.png") - _png(tmp_path / "exact.png")).mean() < 3


class TestDrain:
    def _slow_figures(self, tmp_path, n):
        for i in range(n):
            fig = _scatter(n=100_000, seed=i)
            save_figure(fig, tmp_path / f"{i}.png", dpi=150)
            plt.close(fig)

    def test_wait_moves_queued_figures_to_a_drain_pool_of_the_rest_of_the_budget(self, tmp_path, monkeypatch):
        pools = []
        real_pool = figures._pool
        monkeypatch.setattr(figures, "_pool", lambda n: pools.append(n) or real_pool(n))
        figures.start(4)
        in_flight = []
        real_submit = figures._submit
        monkeypatch.setattr(figures, "_submit", lambda picks: real_submit(picks) or in_flight.append(len(figures._in_flight)))
        try:
            self._slow_figures(tmp_path, 8)
            computing = max(in_flight)
            drain_submit = figures._drain.submit
            moved = []
            monkeypatch.setattr(figures._drain, "submit", lambda *a: moved.append(a) or drain_submit(*a))
            figures.wait()
        finally:
            figures.stop()
        assert pools == [1, 3]  # background 4 // THREADS_PER_FIGURE_WORKER, drain the other 3
        assert computing == 1  # while the main thread computes: the background worker only
        assert max(in_flight) == 4 and moved  # while it waits: the whole budget
        assert sorted(p.name for p in tmp_path.iterdir()) == sorted(f"{i}.png" for i in range(8))

    def test_a_budget_of_one_thread_has_no_drain_pool(self, tmp_path, monkeypatch):
        pools = []
        real_pool = figures._pool
        monkeypatch.setattr(figures, "_pool", lambda n: pools.append(n) or real_pool(n))
        figures.start(1)
        try:
            self._slow_figures(tmp_path, 3)
        finally:
            figures.stop()
        assert pools == [1]
        assert len(list(tmp_path.glob("*.png"))) == 3

    def test_drain_pool_errors_propagate_from_wait(self, tmp_path):
        figures.start(4)
        try:
            self._slow_figures(tmp_path, 3)
            for i in range(3):
                save_figure(_scatter(), tmp_path / "missing_dir" / f"{i}.png")
            with pytest.raises(FileNotFoundError):
                figures.wait()
        finally:
            figures.abort()


class _DieInWorker:
    """Kills the worker process that unpickles it."""

    def __reduce__(self):
        return os._exit, (1,)


class _SleepInWorker:
    def __init__(self, seconds):
        self.seconds = seconds

    def __reduce__(self):
        return time.sleep, (self.seconds,)


def _wait_with_timeout(timeout=60):
    """figures.wait() on a thread; returns the exception it raised. A hang fails the test."""
    result = {}

    def run():
        try:
            figures.wait()
            result["error"] = None
        except BaseException as error:  # noqa: BLE001 - handed to the test
            result["error"] = error

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(timeout)
    assert "error" in result, "figures.wait() hung"
    return result["error"]


class TestBrokenPools:
    def test_a_dead_worker_raises_broken_process_pool_and_does_not_hang(self):
        figures.start(8)
        try:
            save_figure(_DieInWorker())
            for _ in range(20):
                save_figure(_SleepInWorker(0.5))
            assert isinstance(_wait_with_timeout(), BrokenProcessPool)
        finally:
            figures.abort()

    def test_a_failing_worker_initializer_raises_broken_process_pool(self, monkeypatch):
        monkeypatch.setattr(figures, "figure_worker", types.SimpleNamespace(init=sys.exit))
        figures.start(4)
        try:
            save_figure(_SleepInWorker(0.1))
            assert isinstance(_wait_with_timeout(), BrokenProcessPool)
        finally:
            figures.abort()

    def test_a_figure_whose_submit_fails_stays_held_and_the_error_is_raised(self, pool, monkeypatch):
        def broken_submit(*args):
            raise BrokenProcessPool("pool broke")

        monkeypatch.setattr(figures._executor, "submit", broken_submit)
        save_figure(_SleepInWorker(0))
        assert len(figures._held) == 1  # not lost
        with pytest.raises(BrokenProcessPool, match="pool broke"):
            figures.wait()
        monkeypatch.undo()

    def test_a_base_exception_in_submit_leaks_no_reservation(self, pool, monkeypatch, tmp_path):
        class Interrupt(BaseException):
            pass

        def interrupted_submit(*args):
            raise Interrupt

        monkeypatch.setattr(figures._executor, "submit", interrupted_submit)
        with pytest.raises(Interrupt):
            save_figure(_scatter(), tmp_path / "kept.png")
        assert not figures._reserved and len(figures._held) == 1  # not lost
        assert not figures._errors  # an interrupt is re-raised, not recorded as a write error
        monkeypatch.undo()
        assert _wait_with_timeout() is None  # a later wait() submits it again and returns
        assert (tmp_path / "kept.png").stat().st_size > 0


class TestErrorBookkeeping:
    def test_secondary_errors_are_logged_not_dropped(self, caplog):
        figures._errors.extend([ValueError("first"), ValueError("second")])
        with caplog.at_level("ERROR", logger="spoqc.core.figures"), pytest.raises(ValueError, match="first"):
            with figures._state:
                figures._raise_worker_error()
        assert any(r.exc_info and "second" in str(r.exc_info[1]) for r in caplog.records)
        assert figures._errors == []

    def test_abort_logs_a_late_error_and_leaves_none_for_the_next_start(self, tmp_path, caplog):
        figures.start(2)
        save_figure(_scatter(n=200_000), tmp_path / "missing_dir" / "late.png", dpi=150)  # fails after rendering
        _until(lambda: figures._in_flight)
        with caplog.at_level("ERROR", logger="spoqc.core.figures"):
            figures.abort()
        assert figures._errors == [] and not figures._held and not figures._in_flight
        assert any(r.exc_info and isinstance(r.exc_info[1], FileNotFoundError) for r in caplog.records)
        figures.start(2)
        try:
            save_figure(_scatter(), tmp_path / "next.png")
        finally:
            figures.stop()  # raises if the late error had leaked into this run
        assert (tmp_path / "next.png").exists()


class TestImageKinds:
    def _fig(self, data, **kwargs):
        fig = plt.figure(figsize=(2, 2), dpi=50)
        fig.add_axes((0, 0, 1, 1)).imshow(data, **kwargs)
        return fig

    def test_float16_image_is_reduced(self):
        data = np.log10(np.random.default_rng(0).integers(0, 255, (4000, 4000), dtype=np.uint8) + np.float16(1))
        assert data.dtype == np.float16
        reduced = figures._reduced_images(self._fig(data), 50)
        assert [v.shape for v in reduced.values()] == [(200, 200, 4)]

    def test_data_stage_interpolation_is_never_reduced(self):
        fig = self._fig(np.zeros((4000, 4000), np.float32), interpolation_stage="data")
        assert figures._reduced_images(fig, 50) == {}


class TestOrphanedWorkers:
    def test_workers_exit_when_the_parent_process_dies(self, tmp_path):
        script = f"""
import os, sys
from spoqc.core import figures
import matplotlib.pyplot as plt
import numba
figures.start(2)
fig, ax = plt.subplots(); ax.plot([0, 1])
figures.save_figure(fig, {str(tmp_path / 'a.png')!r})
figures.wait()
pids = [p.pid for pool in (figures._executor, figures._drain) for p in pool._processes.values()]
print(' '.join(map(str, pids)), flush=True)
os._exit(0)  # die without shutting the pools down
"""
        out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=300, check=True)
        pids = [int(p) for p in out.stdout.split()]
        assert len(pids) == 2

        def alive(pid):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return False
            return True

        _until(lambda: not any(alive(p) for p in pids), timeout=10 * figure_worker.PARENT_POLL_SECONDS + 10)


class _DieInWorkerAfter:
    """Kills the worker `seconds` after it unpickles this (builtins only, so the worker imports
    nothing first and the timing is exact)."""

    def __init__(self, seconds):
        self.seconds = seconds

    def __reduce__(self):
        return eval, (f"__import__('time').sleep({self.seconds}) or __import__('os')._exit(1)",)


def _slow_submit(monkeypatch, pool, seconds, when):
    """Make `pool.submit` sleep `seconds` first whenever `when()` is true (widens the window in
    which the pool can break while a submit is under way); returns the list of delayed calls."""
    real, delayed = pool.submit, []

    def submit(*args):
        if when():
            delayed.append(threading.current_thread().name)
            time.sleep(seconds)
        return real(*args)

    monkeypatch.setattr(pool, "submit", submit)
    return delayed


class TestLockOrder:
    """A pool breaking while another thread is inside submit() must raise, not deadlock."""

    def test_caller_submit_while_the_pool_breaks(self, monkeypatch):
        # save_figure's own submit is delayed 3 s while the in-flight worker dies after 1 s: the
        # pool breaks (its manager thread holds the shutdown lock and runs _finished) meanwhile.
        figures.start(8)  # 2 background workers
        try:
            for _ in range(2):
                save_figure(_SleepInWorker(0))
            figures.wait()  # both background workers are up
            save_figure(_DieInWorkerAfter(1.0))
            result = {}

            def run():
                try:
                    save_figure(_SleepInWorker(0))
                    figures.wait()
                    result["error"] = None
                except BaseException as error:  # noqa: BLE001 - handed to the test
                    result["error"] = error

            caller = threading.Thread(target=run, daemon=True, name="caller")
            armed = [True]

            def when():
                hit = armed[0] and threading.current_thread() is caller
                if hit:
                    armed[0] = False
                return hit

            delayed = _slow_submit(monkeypatch, figures._executor, 3.0, when)
            caller.start()
            caller.join(60)
            assert "error" in result, "deadlock: save_figure/wait hung while the pool broke"
            assert delayed == ["caller"]
            assert isinstance(result["error"], BrokenProcessPool)
        finally:
            figures.abort()

    def test_drain_callback_submit_to_the_background_pool_while_it_breaks(self, monkeypatch):
        # A drain write finishes; its callback (on the drain pool's manager thread) hands the next
        # held figure to the background pool, whose submit is delayed 3 s while its in-flight
        # worker dies after 1 s: the background pool breaks meanwhile.
        figures.start(8)  # 2 background workers, 6 drain workers
        try:
            for _ in range(2):
                save_figure(_SleepInWorker(0))
            figures.wait()
            save_figure(_DieInWorkerAfter(1.0))  # one background worker busy, one idle
            done = Future()
            done.set_result(None)
            with figures._state:
                figures._in_flight[done] = (figures._drain, 0)  # a drain write that just finished
                figures._held.append(((pickle.dumps(_SleepInWorker(0)), None, (), False, {}), 0))
            callback = threading.Thread(target=figures._finished, args=(done,), daemon=True, name="drain-callback")
            delayed = _slow_submit(monkeypatch, figures._executor, 3.0, lambda: threading.current_thread() is callback)
            callback.start()
            callback.join(60)
            assert not callback.is_alive(), "deadlock: the drain callback's submit hung while the pool broke"
            assert delayed == ["drain-callback"]
            assert isinstance(_wait_with_timeout(), BrokenProcessPool)
        finally:
            figures.abort()


class TestImshow:
    """figures.imshow: a dense scalar image is coloured and reduced before matplotlib copies it.
    Output: a 12 x 6 in figure at 50 dpi with a colorbar, as plot_pixels draws (at 300 dpi)."""

    DPI = 50

    @staticmethod
    def _data(dtype, shape=(1500, 2250), seed=0):
        rng = np.random.default_rng(seed)
        yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
        f = np.sin(xx / 60.0) * np.cos(yy / 40.0) * 10 + rng.normal(0, 1, shape)
        if dtype == "float":
            f[100:200, 100:900] = np.nan
            f[700:720, :] = np.inf
            f[900:910, :] = -np.inf
            return f
        return {"uint8": (f > 3).astype(np.uint8), "bool": f > 0, "int64": np.rint(f).astype(np.int64)}[dtype]

    def _figure(self, data, new, **kwargs):
        plt.figure(figsize=(12, 6))
        kwargs = {"cmap": "hot", "extent": [0, 90, 0, 60], "aspect": "equal", **kwargs}
        if new:
            image = figures.imshow(plt.gca(), data, dpi=self.DPI, **kwargs)
        else:
            image = plt.imshow(data, **kwargs)
        plt.colorbar()
        return plt.gcf(), image

    def _png(self, fig, tmp_path, name):
        save_figure(fig, tmp_path / name, bbox_inches="tight", dpi=self.DPI)
        return _png(tmp_path / name)

    @pytest.mark.parametrize("layout", ["contiguous", "slice", "flipud"])
    @pytest.mark.parametrize("dtype", ["float", "uint8", "bool", "int64"])
    def test_renders_as_matplotlib_and_save_figure_render_the_full_array(self, dtype, layout, tmp_path):
        # slice: a bounding-box subfigure of the full image; flipud: plot_pixels(flip=True)
        data = {"contiguous": lambda a: a, "slice": lambda a: a[5:1495, 7:2240], "flipud": np.flipud}[layout](
            self._data(dtype, shape=(1500, 2400))
        )
        expected = self._png(self._figure(data, new=False)[0], tmp_path, "plain.png")
        fig, image = self._figure(data, new=True)
        assert image.get_array().ndim == 3  # reduced by figures.imshow
        actual = self._png(fig, tmp_path, "new.png")
        assert actual.shape == expected.shape
        assert np.abs(actual - expected).mean() <= 0.5 / 255, np.abs(actual - expected).mean()

    def test_matplotlib_never_masks_the_full_array(self, monkeypatch):
        sizes = []
        masked_invalid = matplotlib.cbook.safe_masked_invalid

        def record(x, *args, **kwargs):
            sizes.append(int(np.prod(np.shape(x)[:2])))  # samples, not channels
            return masked_invalid(x, *args, **kwargs)

        monkeypatch.setattr(matplotlib.cbook, "safe_masked_invalid", record)
        data = self._data("float")
        _, image = self._figure(data, new=True)
        assert image.get_array().dtype == np.uint8 and image.get_array().ndim == 3
        assert sizes and max(sizes) < data.size / 4

    def test_colorbar_and_norm_keep_the_finite_range_of_the_full_array(self):
        data = self._data("float")
        _, image = self._figure(data, new=True)
        finite = data[np.isfinite(data)]
        assert (image.norm.vmin, image.norm.vmax) == (finite.min(), finite.max())
        assert plt.gci() is image and image.get_cmap().name == "hot"
        _, plain = self._figure(data, new=False)
        assert (plain.norm.vmin, plain.norm.vmax) == (image.norm.vmin, image.norm.vmax)

    def test_image_not_denser_than_the_output_goes_to_matplotlib(self):
        data = self._data("float", shape=(150, 225))
        _, image = self._figure(data, new=True)
        assert np.ma.isMaskedArray(image.get_array()) and image.get_array().shape == data.shape

    @pytest.mark.parametrize("origin", ["upper", "lower"])
    def test_without_extent_the_axes_limits_are_matplotlibs(self, origin):
        data = self._data("float")
        fig, image = self._figure(data, new=True, extent=None, origin=origin)
        new_limits = (fig.axes[0].get_xlim(), fig.axes[0].get_ylim())
        assert image.get_array().ndim == 3  # reduced
        fig, _ = self._figure(data, new=False, extent=None, origin=origin)
        assert new_limits == (fig.axes[0].get_xlim(), fig.axes[0].get_ylim())

    def test_saved_dense_figure_is_small_and_needs_no_collection(self, monkeypatch, tmp_path):
        collected = []
        monkeypatch.setattr(figures.gc, "collect", lambda *a: collected.append(1))
        fig, _ = self._figure(self._data("float"), new=True)
        assert not figures._dense_images(fig, self.DPI)
        save_figure(fig, tmp_path / "f.png", bbox_inches="tight", dpi=self.DPI)
        assert not collected


class TestCapWait:
    def test_main_thread_blocked_at_the_cap_lets_the_drain_pool_write(self, tmp_path, monkeypatch):
        figures.start(4)  # 1 background worker, 3 drain workers
        drain = figures._drain
        try:
            submitted = []
            submit = figures._submit
            monkeypatch.setattr(figures, "_submit", lambda picks: (submitted.extend(p for _, p, _ in picks), submit(picks)))
            # room for 3 figures: while the background worker writes one, the next ones are held
            blob = len(pickle.dumps(_scatter(n=20_000), protocol=pickle.HIGHEST_PROTOCOL))
            monkeypatch.setattr(figures, "MAX_PENDING_BYTES", 3 * blob)
            for i in range(8):
                save_figure(_scatter(n=20_000, seed=i), tmp_path / f"{i}.png", dpi=50)
                assert figures._cap_wait is None
            before_wait = list(submitted)
            figures.wait()
        finally:
            figures.stop()
        assert drain in before_wait, "no figure went to the drain pool while save_figure waited"
        assert len(list(tmp_path.glob("*.png"))) == 8

    def test_drain_at_the_cap_stops_when_it_clears_and_the_main_thread_resumes_with_a_free_core(self, monkeypatch):
        budget = 4  # 1 background worker, 3 drain workers
        figures.start(budget)
        drain_workers = figures._workers[figures._drain]
        drain_picks = []  # (cap still holding, wait() draining) for each drain reservation
        returns = []  # (drain workers busy, all workers busy) as each save_figure returns
        computing = []  # (save_figure call, drain busy, all busy) while the main thread "computes"
        in_save = threading.Event()
        done = threading.Event()

        def counts():
            pools = [pool for pool, _ in (*figures._in_flight.values(), *figures._reserved.values())]
            return pools.count(figures._drain), len(pools)

        reserve = figures._reserve

        def recording_reserve():
            picks = reserve()
            cap_holds = figures._cap_wait is not None and figures._over_cap(figures._cap_wait)
            drain_picks.extend((cap_holds, figures._draining) for _, p, _ in picks if p is figures._drain)
            return picks

        def sample():
            while not done.is_set():
                with figures._state:
                    if not in_save.is_set() and returns:
                        computing.append((len(returns) - 1, *counts()))
                time.sleep(0.001)

        monkeypatch.setattr(figures, "_reserve", recording_reserve)
        sampler = threading.Thread(target=sample, daemon=True)
        try:
            blob = len(pickle.dumps(_SleepInWorker(0.2), protocol=pickle.HIGHEST_PROTOCOL))
            # room for 5: more held figures than the background worker and the drain pool take
            monkeypatch.setattr(figures, "MAX_PENDING_BYTES", 5 * blob)
            sampler.start()
            for _ in range(20):
                in_save.set()
                save_figure(_SleepInWorker(0.2))
                with figures._state:
                    returns.append(counts())
                    in_save.clear()
                time.sleep(0.02)  # the main thread computes the next figure
            done.set()
            sampler.join()
            loop_picks = list(drain_picks)
            figures.wait()
        finally:
            done.set()
            figures.stop()
        assert loop_picks, "no figure went to the drain pool while save_figure waited at the cap"
        # the drain pool is handed figures only while a save_figure is blocked past the cap
        assert all(cap_holds and not draining for cap_holds, draining in loop_picks), loop_picks
        # right after a cap wait: a drain worker is free, so the main thread has its core
        assert all(drain < drain_workers and total <= budget - 1 for drain, total in returns), returns
        # while it computes, drain renders finish without being replaced and the budget holds
        assert computing
        assert all(drain <= returns[call][0] and total <= budget - 1 for call, drain, total in computing), computing

    def test_a_write_error_while_blocked_at_the_cap_is_raised_not_waited_on(self, monkeypatch):
        class Interrupt(BaseException):
            pass

        def interrupted_submit(*args):
            raise Interrupt

        def broken_submit(*args):
            raise BrokenProcessPool("pool broke")

        figures.start(4)
        try:
            # a figure held with nothing in flight and no error recorded
            monkeypatch.setattr(figures._executor, "submit", interrupted_submit)
            with pytest.raises(Interrupt):
                save_figure(_SleepInWorker(0))
            assert len(figures._held) == 1 and not figures._in_flight and not figures._errors
            # the next figure is over the cap; handing the held one out fails, so nothing will notify
            monkeypatch.setattr(figures._executor, "submit", broken_submit)
            monkeypatch.setattr(figures._drain, "submit", broken_submit)
            monkeypatch.setattr(figures, "MAX_PENDING_BYTES", 1)
            result = {}

            def blocked_save():
                try:
                    save_figure(_SleepInWorker(0))
                    result["error"] = None
                except BaseException as error:  # noqa: BLE001 - handed to the test
                    result["error"] = error

            thread = threading.Thread(target=blocked_save, daemon=True)
            thread.start()
            thread.join(30)
            assert "error" in result, "save_figure hung at the cap after a write error"
            assert isinstance(result["error"], BrokenProcessPool)
            assert figures._cap_wait is None
        finally:
            monkeypatch.undo()
            figures.abort()


class TestMainThreadKernels:
    def test_image_kernels_run_on_the_budget_less_the_busy_workers_with_the_same_result(self, monkeypatch):
        data = np.random.default_rng(0).random((3000, 4000))

        def draw():
            fig = plt.figure(figsize=(2, 2), dpi=50)
            return figures.imshow(fig.add_axes((0, 0, 1, 1)), data, dpi=50).get_array().copy()

        alone = draw()
        figures.start(4)
        try:
            save_figure(_SleepInWorker(3))
            _until(lambda: figures._in_flight)  # the background worker is busy
            used = []
            for name in ("_finite_range", "_colour_blocks"):
                kernel = getattr(figures, name)
                monkeypatch.setattr(figures, name, lambda *a, k=kernel: used.append(numba.get_num_threads()) or k(*a))
            before = numba.get_num_threads()
            beside_a_worker = draw()
            after = numba.get_num_threads()
        finally:
            figures.stop()
        assert used == [max(1, min(numba.config.NUMBA_NUM_THREADS, 4 - 1))] * 2
        assert after == before
        assert np.array_equal(alone, beside_a_worker)
