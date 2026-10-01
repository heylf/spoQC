
# In[]
import numpy as np
import time
import matplotlib.pyplot as plt
from numba import njit, prange

from .. import helperfuncs
from spoqc.core import figures
from spoqc.core.figures import save_figure

# -------- Numba kernels (pure compute; no I/O) --------
# Every kernel parallelises over image rows with prange, so the thread count is
# numba's pool size (NUMBA_NUM_THREADS, set by spoqc.core.threads.configure).

@njit(parallel=True, fastmath=False)
def compute_unary_numba(p, eps, u_out):
    """
    p: (H, W) probabilities, any float dtype (cast to float32 per pixel)
    u_out: (H, W, 2) float32 (preallocated)
    """
    H, W = p.shape
    for i in prange(H):
        for j in range(W):
            val = np.float32(p[i, j])
            # log-space unaries
            u0 = -np.log(1.0 - val + eps)
            u1 = -np.log(val + eps)
            u_out[i, j, 0] = u0
            u_out[i, j, 1] = u1


@njit(parallel=True, fastmath=False)
def update_messages_numba(
    u, up, down, left, right, pairwise, direction_idx,
    alpha, normalize_mode, eps_norm,
    cur
):
    """
    All arrays are (H, W, 2) float32 except pairwise (2,2) float32.
    normalize_mode: 0 -> 'min', 1 -> 'total'
    cur is updated in place: pixel (i, j) reads cur[i, j] before writing it,
    and no other pixel reads it. There is one pointer to cur, so no aliasing.
    Returns: max |updated - cur| over the image (float32)
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

            cur[i, j, 0] = up0
            cur[i, j, 1] = up1

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
def beliefs_and_labels_numba(u, up, down, left, right, eps_norm, b0_out, labels_out):
    """
    Computes the normalized belief of label 0 and the argmin label per pixel.
    u, up, down, left, right: (H, W, 2) float32
    b0_out: (H, W) float32, labels_out: (H, W) int8
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
            b0_out[i, j] = nb0
            labels_out[i, j] = 0 if nb0 <= nb1 else 1


def first_version_loopy_belief_propagation_parallel(
        prob_map_np,
        beta=1.0,
        alpha=0.3,
        max_iter=20,
        normalize='min',
        tolerance=1e-8,
        flip_tolerance=1e-6,
        flip_check=10,
    ):
    """
    Min-sum loopy belief propagation on a 4-connected 2-label grid.

    All state lives in RAM: messages (4, n+2, m+2, 2) float32 = 32 B/pixel and
    unary (n, m, 2) float32 = 8 B/pixel. Each direction sweep is one parallel
    kernel over the whole image. It writes messages[d] and reads only the other
    three directions, so updating in place is exact.
    Returns (belief of label 0 as (n, m) float32, labels as (n, m) int8).
    """
    timer = helperfuncs.Timer()
    timer.start()

    n, m = prob_map_np.shape

    pairwise = np.array([[0.0, beta], [beta, 0.0]], dtype=np.float32)

    # Normalize mode for Numba (avoid strings in kernels)
    if normalize == "min":
        normalize_mode = 0
    elif normalize == "total":
        normalize_mode = 1
    else:
        raise SystemExit("[ERROR] Normalization not supported")

    eps = np.float32(1e-8)
    eps_norm = np.float32(1e-8)  # small guard to avoid 0-division in kernels

    unary = np.empty((n, m, 2), dtype=np.float32)
    compute_unary_numba(prob_map_np, eps, unary)

    # Padded messages: rows 0, n+1 and columns 0, m+1 stay zero (the border).
    messages = np.zeros((4, n + 2, m + 2, 2), dtype=np.float32)
    up    = messages[0, 0:n,   1:m+1, :]
    down  = messages[1, 2:n+2, 1:m+1, :]
    left_ = messages[2, 1:n+1, 0:m,   :]
    right_= messages[3, 1:n+1, 2:m+2, :]

    # -----------------------
    # LBP iterations (Gauss-Seidel over the directions up, down, left, right)
    # -----------------------
    early_flipping_stop = 0.0

    for it in range(max_iter):
        start = time.time()
        print(it)
        change = 0.0

        for direction_idx in range(4):
            cur = messages[direction_idx, 1:n+1, 1:m+1, :]
            delta = update_messages_numba(
                unary, up, down, left_, right_, pairwise, direction_idx,
                np.float32(alpha), normalize_mode, eps_norm,
                cur
            )
            if float(delta) > change:
                change = float(delta)

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
            # Rolling sum for the flipping heuristic
            early_flipping_stop += change
            if it % flip_check == 0:
                early_flipping_stop = 0.0

        end = time.time()
        print(f"[Time] {end - start:.3f} seconds")

    # -----------------------
    # Beliefs & labels
    # -----------------------
    beliefs = np.empty((n, m), dtype=np.float32)
    labels = np.empty((n, m), dtype=np.int8)
    beliefs_and_labels_numba(unary, up, down, left_, right_, eps_norm, beliefs, labels)

    timer.stop()
    # beliefs is prob for good quality
    return beliefs, labels

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
    figures.imshow(plt.gca(), average_cell_probability_image, cmap='viridis')
    plt.colorbar(fraction=0.046, pad=0.04)
    if ( flip ):
        plt.gca().invert_yaxis()

    plt.subplot(1, 3, 2)
    t = 0.6
    plt.title(f"Predicted Probabilities (binary > {t})")
    figures.imshow(plt.gca(), (average_cell_probability_image > t).astype(np.uint8), cmap='gray')
    helperfuncs.add_manual_legend(legend_dict={"high Q": "#FFFFFF", "low Q": "#000000"})
    if ( flip ):
        plt.gca().invert_yaxis()

    plt.subplot(1, 3, 3)
    plt.title("Inferred Labels (LBP + Early Stop)")
    figures.imshow(plt.gca(), labels, cmap='gray')
    helperfuncs.add_manual_legend(legend_dict={"mask": "#FFFFFF", "low Q": "#000000"})
    if ( flip ):
        plt.gca().invert_yaxis()
    save_figure(plt.gcf(), f'{figure_path}/markov_random_field_calculations.png', f'{figure_path}/markov_random_field_calculations.pdf', bbox_inches='tight')
    plt.close()

