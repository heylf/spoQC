import numpy as np
import time
import matplotlib.pyplot as plt
import sys

from .. import helperfuncs

def first_version_loopy_belief_propagation(prob_map, beta=1.0, alpha=0.3, max_iter=20,
                                           normalize='min', tolerance=1e-8, flip_tolerance=1e-6, flip_check=10):
    
    '''
    Perform Loopy Belief Propagation for binary MRF on a grid.

    Parameters:
        prob_map (ndarray): nxm array of predicted probabilities (prior knowledge).
        beta (float): strength of the pairwise potential (higher = smoother labels).
        max_iter (int): number of iterations for message passing.
        normalize (str): 'min' or 'mean' to control message normalization.
        alpha (float): damping factor to fine tune learning. Balances old and new message.
        tolerance (float): convergence condition for message learning (early stopping).
        flip_tolerance (float): condition to check for message direction flipping (early stopping).
        flip_check (int): interval to check for message flipping.
    
    Returns:
        labels (ndarray): inferred binary labels (0 = bad, 1 = good).
    '''

    n, m = prob_map.shape
    shape = (n, m, 2)
    unary = np.zeros(shape)

    # Use log space so you can use sum instead of multiplciation.
    unary[:, :, 0] = -np.log(1 - prob_map + 1e-8)
    unary[:, :, 1] = -np.log(prob_map + 1e-8)

    pairwise = np.array([[0, beta], [beta, 0]])
    
    # +2 for padding to do quicker and memory efficient matrix summation for messages.
    n_pad = n+2
    m_pad = m+2

    # first index = 0: up, 1: down, 2: left, 3: right
    # last index = 0: bad, 1: good
    messages = np.zeros((4, n_pad, m_pad, 2))
    old_messages = np.copy(messages)
    early_flipping_stop = 0

    for it in range(max_iter):
        print(it)

        up = messages[0, 0:(n_pad-2), 1:(m_pad-1)]
        down = messages[1, 2:n_pad, 1:(m_pad-1)]
        left = messages[2, 1:(n_pad-1), 0:(m_pad-2)]
        right = messages[3, 1:(n_pad-1), 2:m_pad]

        change = 0
        for direction_idx in range(4):

            if ( direction_idx == 0 ):
                msg = unary + down + left + right
            if ( direction_idx == 1 ):
                msg = unary + up + left + right
            if ( direction_idx == 2 ):
                msg = unary + up + down + right
            if ( direction_idx == 3 ):
                msg = unary + up + down + left

            # Compute outgoing message using pairwise potential.
            # I add a penalty beta (pairwise potential) for disagreement.
            # Take min to find best path. Optimization problem.
            for s in range(2):
                msg[:, :, s] = np.min(msg + pairwise[s], axis=2)

            if ( normalize == 'min' ):
                msg -= np.min(msg, axis=2)[:, :, None]
                msg = msg / (msg[:, :, 0] + msg[:, :, 1])[:, :, None]
            elif ( normalize == 'total' ):
                msg = msg / (msg[:, :, 0] + msg[:, :, 1])[:, :, None]
            else:
                sys.exit('[ERROR] Normalization not supported')

            messages[direction_idx, 1:(n_pad-1), 1:(m_pad-1),:] = \
                (1 - alpha) * messages[direction_idx, 1:(n_pad-1), 1:(m_pad-1),:] + alpha * msg
            
            delta = np.max(np.abs(old_messages[direction_idx] - messages[direction_idx]))
            change = max(change, delta)

        flipping_change = np.abs((early_flipping_stop / flip_check) - change)

        if it % flip_check == 0:
            print(f'flipping {flipping_change}')
        if change < tolerance:
            print(f'[NOTE] LBP converged after {it} iterations with {change} change')
            break
        elif flipping_change < flip_tolerance and it % flip_check == 0:
            print(f'[NOTE] LBP converged after {it} iterations with {flipping_change} flipping change')
            break
        else:
            old_messages[:] = messages
            early_flipping_stop += change
            if it % flip_check == 0:
                early_flipping_stop = 0

    up = messages[0, 0:(n_pad-2), 1:(m_pad-1)]
    down = messages[1, 2:n_pad, 1:(m_pad-1)]
    left = messages[2, 1:(n_pad-1), 0:(m_pad-2)]
    right = messages[3, 1:(n_pad-1), 2:m_pad]
    beliefs = unary + up + down + left + right
    beliefs = beliefs / (beliefs[:, :, 0] + beliefs[:, :, 1])[:, :, None]

    labels = np.argmin(beliefs, axis=2)

    return labels

def example_markov_field():

    np.random.seed(0)
    prob_map = np.random.rand(5_000, 5_000)
    prob_map[1000:1500, 1000:1500] += 0.5
    prob_map = np.clip(prob_map, 0, 1)

    start = time.time()
    labels = first_version_loopy_belief_propagation(prob_map, beta=1.5, max_iter=10, alpha=.3, normalize='total')
    end = time.time()
    print(f"[Time] {end - start} seconds")

    # Visualization
    plt.figure(figsize=(20, 5))
    plt.subplot(1, 3, 1)
    plt.title("Predicted Probabilities")
    plt.imshow(prob_map, cmap='viridis')
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.title("Predicted Probabilities binary")
    plt.imshow((prob_map > 0.6).astype(np.uint8), cmap='gray')
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.title("Inferred Labels (LBP + Early Stop)")
    plt.imshow(labels, cmap='gray')
    plt.colorbar()

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
    plt.close()


