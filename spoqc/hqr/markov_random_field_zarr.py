
# In[]
import numpy as np
import time
import matplotlib.pyplot as plt
import zarr

from zarr.codecs import BloscCodec

from .. import helperfuncs

def first_version_loopy_belief_propagation(
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
    
    # Read-heavy data, mostly sequential access, should be bigger in size.
    # Optimized for ~32 MiB tiles.
    chunk_read = 2048

    # Update heavy, mostly iterations, should be smaller in size.
    # Keep the spatial chunk close to the tile size to avoid I/O waste.
    # Optimized for ~8 MiB tiles for unary and beliefs and ~4 MiB for labels.
    chunk_update = 1024

    # Tile size for block-wise compute over the core (n x m) region.
    # Tune this to your RAM (e.g., (1024, 1024), (2048, 2048), etc.)
    tile = (chunk_update, chunk_update)

    n, m = prob_map_np.shape
    shape = (n, m, 2)
    n_pad, m_pad = n + 2, m + 2

    pairwise = np.array([[0, beta], [beta, 0]])

    # -----------------------
    # Zarr store setup
    # -----------------------
    compressor = BloscCodec(cname="zstd", clevel=5, shuffle="shuffle")
    store = zarr.storage.LocalStore(f"{spoqc_tmp_folder}/lbp_store_{modality}_zarr")
    root = zarr.group(store=store, overwrite=True)

    # Prob map (input) in Zarr
    prob_map = root.create_array(
        "prob_map",
        shape=(n, m),
        chunks=(min(n, chunk_read), min(m, chunk_read)),
        compressors=[compressor],
        dtype="f4",
    )
    prob_map[:] = prob_map_np  # one-time write; remove if you already have this on disk

    # Unary (n, m, 2) — chunk size matches tile size to avoid partial chunk writes
    unary = root.create_array(
        "unary",
        shape=shape,
        chunks=(min(n, chunk_update), min(m, chunk_update), 2),
        compressors=[compressor],
        dtype="f4",
    )

    # Messages (4, n+2, m+2, 2) with padding; and a copy for deltas
    messages = root.create_array(
        "messages",
        shape=(4, n_pad, m_pad, 2),
        chunks=(1, min(n_pad, chunk_update), min(m_pad, chunk_update), 2),  # chunk per direction
        compressors=[compressor],
        dtype="f4",
    )
    old_messages = root.create_array(
        "old_messages",
        shape=(4, n_pad, m_pad, 2),
        chunks=messages.chunks,
        compressors=[compressor],
        dtype="f4",
    )
    # Beliefs and labels — chunk size matches tile size to avoid partial chunk writes
    beliefs = root.create_array(
        "beliefs",
        shape=shape,
        chunks=(min(n, chunk_update), min(m, chunk_update), 2),
        compressors=[compressor],
        dtype="f4",
    )
    labels = root.create_array(
        "labels",
        shape=(n, m),
        chunks=(min(n, chunk_update), min(m, chunk_update)),
        dtype="int8",
        compressors=[compressor],
    )

    # -----------------------
    # Initialize unary in tiles (avoid building large temporaries)
    # -----------------------
    eps = 1e-8

    for i0 in range(0, n, tile[0]):
        i1 = min(n, i0 + tile[0])
        for j0 in range(0, m, tile[1]):
            j1 = min(m, j0 + tile[1])

            p = prob_map[i0:i1, j0:j1][:]  # NumPy view
            u = np.empty((i1 - i0, j1 - j0, 2))

            # log-space unaries
            u[:,:,0] = -np.log(1 - p + eps)  # -log(1 - p + eps)
            u[:,:,1] = -np.log(p + eps)

            unary[i0:i1, j0:j1, :] = u

    # Zero-init messages
    messages[:] = 0
    old_messages[:] = 0

    # -----------------------
    # Helpers to iterate tiles over the core (n x m)
    # -----------------------
    def core_tiles():
        for i0 in range(0, n, tile[0]):
            i1 = min(n, i0 + tile[0])
            for j0 in range(0, m, tile[1]):
                j1 = min(m, j0 + tile[1])
                yield i0, i1, j0, j1

    # -----------------------
    # LBP iterations (tile-wise, using Zarr slices)
    # -----------------------
    early_flipping_stop = 0.0

    for it in range(max_iter):
        start = time.time()
        print(it)

        change = 0.0

        for direction_idx in range(4):
            for i0, i1, j0, j1 in core_tiles():
                # Load incoming messages for this tile (as NumPy arrays)
                up    = messages[0, i0: i1,     j0+1: j1+1, :][:]
                down  = messages[1, i0+2: i1+2, j0+1: j1+1, :][:]
                left  = messages[2, i0+1: i1+1, j0:   j1,   :][:]
                right = messages[3, i0+1: i1+1, j0+2: j1+2, :][:]

                u = unary[i0:i1, j0:j1, :][:]  # (Ti, Tj, 2)

                # Compose message accumulator depending on direction
                if direction_idx == 0:      # sending UP, don't use up
                    acc = u + down + left + right
                elif direction_idx == 1:    # sending DOWN, don't use down
                    acc = u + up + left + right
                elif direction_idx == 2:    # sending LEFT, don't use left
                    acc = u + up + down + right
                else:                        # sending RIGHT, don't use right
                    acc = u + up + down + left

                # Apply pairwise (min-sum): for each s, msg[..., s] = min_{s'} (acc[..., s'] + pairwise[s, s'])
                # Do it without big temporaries
                msg = np.empty_like(acc)
                # s = 0
                tmp0 = np.stack((acc[:,:,0] + pairwise[0, 0],
                                acc[:,:,1] + pairwise[0, 1]), axis=-1)
                msg[:,:,0] = np.min(tmp0, axis=-1)
                # s = 1
                tmp1 = np.stack((acc[:,:,0] + pairwise[1, 0],
                                acc[:,:,1] + pairwise[1, 1]), axis=-1)
                msg[:,:,1] = np.min(tmp1, axis=-1)

                # Normalize
                if normalize == "min":
                    msg -= np.min(msg, axis=2, keepdims=True)
                    denom = (msg[:,:,0] + msg[:,:,1])
                    msg /= denom[:,:,None]
                elif normalize == "total":
                    denom = (msg[:,:,0] + msg[:,:,1])
                    msg /= denom[:,:,None]
                else:
                    raise SystemExit("[ERROR] Normalization not supported")

                # Damped update into messages on disk
                tgt_slice = (slice(direction_idx, direction_idx + 1),
                            slice(i0 + 1, i1 + 1),
                            slice(j0 + 1, j1 + 1),
                            slice(None))

                # Read old (this iter) for damping, and previous-iter for delta
                cur = messages[tgt_slice][0]       # (Ti, Tj, 2)
                prev = old_messages[tgt_slice][0]  # (Ti, Tj, 2)

                updated = (1.0 - alpha) * cur + alpha * msg
                messages[tgt_slice] = updated[None,:,:,:]

                delta_tile = np.max(np.abs(prev - updated))
                if delta_tile > change:
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
            # Snapshot for next-iter delta computation
            old_messages[:] = messages
            early_flipping_stop += change
            if it % flip_check == 0:
                early_flipping_stop = 0.0

        end = time.time()
        print(f"[Time] {end - start} seconds")

    # -----------------------
    # Beliefs & labels (tile-wise)
    # -----------------------
    for i0, i1, j0, j1 in core_tiles():
        up    = messages[0, i0: i1,     j0+1: j1+1, :][:]
        down  = messages[1, i0+2: i1+2, j0+1: j1+1, :][:]
        left = messages[2, i0+1: i1+1, j0:   j1,   :][:]
        right= messages[3, i0+1: i1+1, j0+2: j1+2, :][:]

        u = unary[i0:i1, j0:j1, :][:]

        b = u + up + down + left + right
        denom = (b[:,:,0] + b[:,:,1])
        b /= denom[:,:,None]

        beliefs[i0:i1, j0:j1, :] = b
        labels[i0:i1, j0:j1] = np.argmin(b, axis=2).astype("int8")

    print(f"Done. Zarr store at: {spoqc_tmp_folder}/lbp_store_{modality}.zarr")
    timer.stop()
    return labels


def visualize_markov_calculation(average_cell_probability_image, labels, figure_path):
    plt.figure(figsize=(30, 5))
    plt.subplot(1, 3, 1)
    plt.title("Predicted Probabilities")
    plt.imshow(average_cell_probability_image, cmap='viridis')
    plt.colorbar()

    plt.subplot(1, 3, 2)
    t = 0.6
    plt.title(f"Predicted Probabilities (binary > {t})")
    plt.imshow((average_cell_probability_image > t).astype(np.uint8), cmap='gray')
    helperfuncs.add_manual_legend(legend_dict={"high Q": "#FFFFFF", "low Q": "#000000"})

    plt.subplot(1, 3, 3)
    plt.title("Inferred Labels (LBP + Early Stop)")
    plt.imshow(labels, cmap='gray')
    helperfuncs.add_manual_legend(legend_dict={"mask": "#FFFFFF", "low Q": "#000000"})

    plt.savefig(f'{figure_path}/markov_random_field_calculations.png', bbox_inches='tight')
    plt.savefig(f'{figure_path}/markov_random_field_calculations.pdf', bbox_inches='tight')
    plt.close()

