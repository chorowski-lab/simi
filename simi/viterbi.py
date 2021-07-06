import numpy as np
from numba import jit


@jit(nopython=True)
def sentpiece_viterbi(piece_logp):
    """Find the best tokenization given a list of pieces and their probability in time.
    
    Probabilities of subsequent pieces are independent. For every time `t` we need only
    the most probable piece for every possible sentencepiece length.
    
    Args:
        piece_logp (max_piece_len x time): piece_logp[L, T] is a probability of using
            piece of len L+1 at timestep T; first row (corresponding to len=0) is ignored

    Output:
        best_lens: subsequent lengths of pieces on the most probable path
    """
    max_piece_len = piece_logp.shape[0] - 1
    T = piece_logp.shape[1]

    # Can start with a unit piece
    logp = np.ones((T,), dtype=np.float32) * -np.inf
    logp[:max_piece_len] = piece_logp[1:, 0]

    prev_len = np.zeros((T,), dtype=np.int64)
    prev_len[:max_piece_len] = np.arange(1, max_piece_len+1)

    for t in range(1, T):
        # For every timestep t, calculate probability of using piece k that ends in timestep t
        for len_ in range(1, max_piece_len+1):

            if len_ >= t:
                continue
            new_logp = piece_logp[len_, t-len_+1] + logp[t-len_]

            if new_logp > logp[t]:
                logp[t] = new_logp
                prev_len[t] = len_

    best_path = -np.ones((T,), dtype=np.int64)
    t = T
    for i in range(T):
        best_path[i] = prev_len[t-1]
        t -= prev_len[t-1]
        if t <= 0:
            break

    return best_path[:i+1][::-1]
