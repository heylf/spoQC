
# In[]
import numpy as np
import time
import matplotlib.pyplot as plt
import zarr
import os

from numcodecs import Blosc
from numba import njit, prange

from .. import helperfuncs

# -------- Numba kernels (pure compute; no I/O) --------

@njit(parallel=True, fastmath=False)
def compute_unary_tile_numba(p, eps, u_out):
    """
    p: (H, W) float32
    u_out: (H, W, 2) float32 (preallocated)
    """
    H, W = p.shape
    for i in prange(H):
        for j in range(W):
            val = p[i, j]
            # log-space unaries
            u0 = -np.log(1.0 - val + eps)
            u1 = -np.log(val + eps)
            u_out[i, j, 0] = u0
            u_out[i, j, 1] = u1


@njit(parallel=True, fastmath=False)
def update_messages_tile_numba(
    u, up, down, left, right, pairwise, direction_idx,
    alpha, normalize_mode, eps_norm,
    cur, out_updated
):
    """
    All arrays are (H, W, 2) float32 except pairwise (2,2) float32.
    normalize_mode: 0 -> 'min', 1 -> 'total'
    Returns: max |updated - cur| on this tile (float32)
    """
    H, W, _ = u.shape
    row_max = np.zeros(H, dtype=np.float32)  # per-row local maxima

    for i in prange(H):
        m = np.float32(0.0)  # local max for this row
        for j in range(W):
            # Compose accumulator (skip the message in the send direction)
            if direction_idx == 0:  # sending UP, don't use up
                a0 = u[i, j, 0] + down[i, j, 0] + left[i, j, 0] + right[i, j, 0]
                a1 = u[i, j, 1] + down[i, j, 1] + left[i, j, 1] + right[i, j, 1]
            elif direction_idx == 1:  # sending DOWN, don't use down
                a0 = u[i, j, 0] + up[i, j, 0] + left[i, j, 0] + right[i, j, 0]
                a1 = u[i, j, 1] + up[i, j, 1] + left[i, j, 1] + right[i, j, 1]
            elif direction_idx == 2:  # sending LEFT, don't use left
                a0 = u[i, j, 0] + up[i, j, 0] + down[i, j, 0] + right[i, j, 0]
                a1 = u[i, j, 1] + up[i, j, 1] + down[i, j, 1] + right[i, j, 1]
            else:  # 3: sending RIGHT, don't use right
                a0 = u[i, j, 0] + up[i, j, 0] + down[i, j, 0] + left[i, j, 0]
                a1 = u[i, j, 1] + up[i, j, 1] + down[i, j, 1] + left[i, j, 1]

            # Min-sum with 2x2 pairwise
            t00 = a0 + pairwise[0, 0]; t01 = a1 + pairwise[0, 1]
            msg0 = t00 if t00 < t01 else t01
            t10 = a0 + pairwise[1, 0]; t11 = a1 + pairwise[1, 1]
            msg1 = t10 if t10 < t11 else t11

            # Normalize
            if normalize_mode == 0:  # "min"
                mn = msg0 if msg0 < msg1 else msg1
                msg0 -= mn; msg1 -= mn
                den = msg0 + msg1
                if den <= 0.0:
                    den = eps_norm
                msg0 /= den; msg1 /= den
            else:  # "total"
                den = msg0 + msg1
                if den == 0.0:
                    den = eps_norm
                msg0 /= den; msg1 /= den

            # Damping vs current tile "cur"
            c0 = cur[i, j, 0]; c1 = cur[i, j, 1]
            up0 = (1.0 - alpha) * c0 + alpha * msg0
            up1 = (1.0 - alpha) * c1 + alpha * msg1

            out_updated[i, j, 0] = up0
            out_updated[i, j, 1] = up1

            # Delta vs current "cur"
            d0 = up0 - c0;  d0 = -d0 if d0 < 0.0 else d0
            d1 = up1 - c1;  d1 = -d1 if d1 < 0.0 else d1
            d  = d0 if d0 > d1 else d1
            if d > m:
                m = d

        row_max[i] = m

    # Sequential reduction to a single scalar
    max_delta = np.float32(0.0)
    for i in range(H):
        if row_max[i] > max_delta:
            max_delta = row_max[i]

    return max_delta


@njit(parallel=True, fastmath=False)
def beliefs_and_labels_tile_numba(u, up, down, left, right, eps_norm, b_out, labels_out):
    """
    Computes normalized beliefs and argmin labels per pixel.
    u, up, down, left, right: (H, W, 2) float32
    b_out: (H, W, 2) float32, labels_out: (H, W) int8
    """
    H, W, _ = u.shape
    for i in prange(H):
        for j in range(W):
            b0 = u[i, j, 0] + up[i, j, 0] + down[i, j, 0] + left[i, j, 0] + right[i, j, 0]
            b1 = u[i, j, 1] + up[i, j, 1] + down[i, j, 1] + left[i, j, 1] + right[i, j, 1]
            den = b0 + b1
            if den == 0.0:
                den = eps_norm
            nb0 = b0 / den
            nb1 = b1 / den
            b_out[i, j, 0] = nb0
            b_out[i, j, 1] = nb1
            labels_out[i, j] = 0 if nb0 <= nb1 else 1


def first_version_loopy_belief_propagation_parallel(
        prob_map_np,
        spoqc_tmp_folder,
        modality,
        beta=1.0,
        alpha=0.3,
        max_iter=20,
        normalize='min',
        tolerance=1e-8,
        flip_tolerance=1e-6,
        flip_check=10,
    ):
    timer = helperfuncs.Timer()
    timer.start()

    # Tile tuning
    chunk_read = 2048   # read-heavy
    chunk_update = 1024 # update-heavy
    tile = (chunk_update, chunk_update)

    n, m = prob_map_np.shape
    shape = (n, m, 2)
    n_pad, m_pad = n + 2, m + 2

    pairwise = np.array([[0.0, beta], [beta, 0.0]], dtype=np.float32)

    # -----------------------
    # Zarr store setup (float32)
    # -----------------------
    compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)
    store = zarr.DirectoryStore(f"{spoqc_tmp_folder}/lbp_store_{modality}_zarr")
    root = zarr.group(store=store, overwrite=True)

    # Prob map (input) in Zarr
    prob_map = root.create_dataset(
        "prob_map",
        shape=(n, m),
        chunks=(min(n, chunk_read), min(m, chunk_read)),
        compressor=compressor,
        dtype="f4",
    )
    prob_map[:] = prob_map_np.astype(np.float32, copy=False)  # remove if already on disk

    # Unary (n, m, 2)
    unary = root.create_dataset(
        "unary",
        shape=shape,
        chunks=(min(n, chunk_read), min(m, chunk_read), 2),
        compressor=compressor,
        dtype="f4",
    )

    # Beliefs & labels
    beliefs = root.create_dataset(
        "beliefs",
        shape=shape,
        chunks=(min(n, chunk_read), min(m, chunk_read), 2),
        compressor=compressor,
        dtype="f4",
    )
    labels = root.create_dataset(
        "labels",
        shape=(n, m),
        chunks=(min(n, chunk_read), min(m, chunk_read)),
        dtype="int8",
        compressor=compressor,
    )

    # -----------------------
    # Messages via memmap (fast, uncompressed)
    # shape: (4, n+2, m+2, 2) with padding like original
    # -----------------------
    mm_path = os.path.join(spoqc_tmp_folder, f"lbp_messages_{modality}.mmap")
    messages = np.memmap(mm_path, mode="w+", dtype=np.float32,
                         shape=(4, n_pad, m_pad, 2))
    messages[:] = 0.0

    # -----------------------
    # Initialize unary in tiles (Numba kernel)
    # -----------------------
    eps = np.float32(1e-8)
    for i0 in range(0, n, tile[0]):
        i1 = min(n, i0 + tile[0])
        for j0 in range(0, m, tile[1]):
            j1 = min(m, j0 + tile[1])

            p = prob_map[i0:i1, j0:j1]  # (Ti, Tj), avoid [:]
            u = np.empty((i1 - i0, j1 - j0, 2), dtype=np.float32)

            compute_unary_tile_numba(p, eps, u)
            unary[i0:i1, j0:j1, :] = u

    # Normalize mode for Numba (avoid strings in kernels)
    if normalize == "min":
        normalize_mode = 0
    elif normalize == "total":
        normalize_mode = 1
    else:
        raise SystemExit("[ERROR] Normalization not supported")

    eps_norm = np.float32(1e-8)  # small guard to avoid 0-division in kernels

    # -----------------------
    # LBP iterations (tile-wise, delta vs current tile; no snapshot copy)
    # -----------------------
    early_flipping_stop = 0.0

    for it in range(max_iter):
        start = time.time()
        print(it)
        change = 0.0

        for direction_idx in range(4):
            for i0 in range(0, n, tile[0]):
                i1 = min(n, i0 + tile[0])
                for j0 in range(0, m, tile[1]):
                    j1 = min(m, j0 + tile[1])

                    # Load incoming messages for this tile (NumPy arrays)
                    up    = messages[0, i0: i1,     j0+1: j1+1, :]
                    down  = messages[1, i0+2: i1+2, j0+1: j1+1, :]
                    left_ = messages[2, i0+1: i1+1, j0:   j1,   :]
                    right_= messages[3, i0+1: i1+1, j0+2: j1+2, :]

                    u = unary[i0:i1, j0:j1, :]  # (Ti, Tj, 2)

                    # Target slice for this direction update
                    cur  = messages[direction_idx, i0+1:i1+1, j0+1:j1+1, :]  # (Ti, Tj, 2)
                    updated = np.empty_like(cur)

                    delta_tile = update_messages_tile_numba(
                        u, up, down, left_, right_, pairwise, direction_idx,
                        np.float32(alpha), normalize_mode, eps_norm,
                        cur, updated
                    )

                    # Write back the updated messages (no extra axis gymnastics)
                    messages[direction_idx, i0+1:i1+1, j0+1:j1+1, :] = updated

                    if float(delta_tile) > change:
                        change = float(delta_tile)

        flipping_change = abs((early_flipping_stop / flip_check) - change)
        if it % flip_check == 0:
            print(f"flipping {flipping_change:.3e}")

        if change < tolerance:
            print(f"[NOTE] LBP converged after {it} iterations with {change:.3e} change")
            break
        elif flipping_change < flip_tolerance and it % flip_check == 0:
            print(f"[NOTE] LBP converged after {it} iterations with {flipping_change:.3e} flipping change")
            break
        else:
            # No global copy; just update the rolling sum for flipping heuristic
            early_flipping_stop += change
            if it % flip_check == 0:
                early_flipping_stop = 0.0

        end = time.time()
        print(f"[Time] {end - start:.3f} seconds")

    # Ensure memmap contents are flushed
    messages.flush()

    # -----------------------
    # Beliefs & labels (tile-wise via kernel)
    # -----------------------
    for i0 in range(0, n, tile[0]):
        i1 = min(n, i0 + tile[0])
        for j0 in range(0, m, tile[1]):
            j1 = min(m, j0 + tile[1])

            up    = messages[0, i0: i1,     j0+1: j1+1, :]
            down  = messages[1, i0+2: i1+2, j0+1: j1+1, :]
            left_ = messages[2, i0+1: i1+1, j0:   j1,   :]
            right_= messages[3, i0+1: i1+1, j0+2: j1+2, :]

            u = unary[i0:i1, j0:j1, :]

            b = np.empty_like(u)
            lab = np.empty((i1 - i0, j1 - j0), dtype=np.int8)

            beliefs_and_labels_tile_numba(u, up, down, left_, right_, eps_norm, b, lab)

            beliefs[i0:i1, j0:j1, :] = b
            labels[i0:i1, j0:j1] = lab

    print(f"Done. Zarr store at: {spoqc_tmp_folder}/lbp_store_{modality}_zarr")
    print(f"Messages memmap at: {mm_path}")
    timer.stop()
    # beliefs[:,:,0] is prob for good quality
    return beliefs[:,:,0], labels

def visualize_markov_calculation(average_cell_probability_image, labels, figure_path, flip=False):
    # Automatically scale figure size based on image resolution
    img_h, img_w = average_cell_probability_image.shape
    scale = 0.001
    if ( img_h > 35_000 or img_w > 35_000 ):
        scale = 0.0001
    if ( img_h > 350_000 or img_w > 350_000 ):
        scale = 0.00001
    ax_width = img_w * scale
    ax_height = img_h * scale
    ncols = 3
    spacing = 3
    fig_width = ncols * ax_width + (ncols - 1) * spacing
    fig_height = ax_height
    plt.figure(figsize=(fig_width, fig_height))

    plt.subplot(1, 3, 1)
    plt.title("Predicted Probabilities")
    plt.imshow(average_cell_probability_image, cmap='viridis')
    plt.colorbar(fraction=0.046, pad=0.04)
    if ( flip ):
        plt.gca().invert_yaxis()

    plt.subplot(1, 3, 2)
    t = 0.6
    plt.title(f"Predicted Probabilities (binary > {t})")
    plt.imshow((average_cell_probability_image > t).astype(np.uint8), cmap='gray')
    helperfuncs.add_manual_legend(legend_dict={"high Q": "#FFFFFF", "low Q": "#000000"})
    if ( flip ):
        plt.gca().invert_yaxis()

    plt.subplot(1, 3, 3)
    plt.title("Inferred Labels (LBP + Early Stop)")
    plt.imshow(labels, cmap='gray')
    helperfuncs.add_manual_legend(legend_dict={"mask": "#FFFFFF", "low Q": "#000000"})
    if ( flip ):
        plt.gca().invert_yaxis()
    plt.savefig(f'{figure_path}/markov_random_field_calculations.png', bbox_inches='tight')
    plt.savefig(f'{figure_path}/markov_random_field_calculations.pdf', bbox_inches='tight')
    plt.close()

