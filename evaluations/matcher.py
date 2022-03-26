
import numpy as np

def nn_matcher(desc0, desc1, nn_thresh, is_mutual_NN=False):
    
    # mat_score = pred['score_matrix_line']

    ## case 1)
    d, num0 = desc0.shape
    num1 = desc1.shape[1]
    desc0_ = desc0.T
    desc1_ = desc1.T

    # ## case 2)
    # num0, d = desc0.shape
    # num1 = desc1.shape[1]
    # desc0_ = desc0
    # desc1_ = desc1

    sum_sq0 = np.sum(np.square(desc0_), axis=1)[:,None]  # (1, 16, 128) -> (1, 128)
    sum_sq1 = np.sum(np.square(desc1_), axis=1)[:,None].T

    dmat = desc0_ @ desc1_.T
    # dmat_tmp = np.ones((128,128)) * np.inf
    dmat_tmp = (sum_sq0 + sum_sq1 - 2.0 * dmat).clip(min=0)


    # Get NN indices and scores.
    idx = np.argmin(dmat_tmp, axis=1)
    scores = dmat_tmp[np.arange(dmat_tmp.shape[0]), idx]
    # Threshold the NN matches.
    keep = scores < nn_thresh
    
    if is_mutual_NN:
        # Check if nearest neighbor goes both directions and keep those.
        idx2 = np.argmin(dmat_tmp, axis=0)
        keep_bi = np.arange(len(idx)) == idx2[idx]
        keep = np.logical_and(keep, keep_bi)
    idx = idx[keep]
    scores = scores[keep]
    # Get the surviving point indices.
    m_idx1 = np.arange(num0)[keep]
    # m_idx1 = np.arange(128)[keep]
    m_idx2 = idx

    mat_nn = np.zeros_like(dmat_tmp)
    mat_nn[m_idx1, m_idx2] = 1
    
    return mat_nn

def nn_matcher_batches(desc0, desc1, nn_thresh, is_mutual_NN=False):
    # desc0 = pred['line_desc0']      # [32, 256, 128]
    # desc1 = pred['line_desc1']      # [32, 256, 128]
    # # mat_score = pred['score_matrix_line']

    b, d, n0 = desc0.shape
    n1 = desc1.shape[2]

    # mat_nn_ = np.zeros((b,n0+1,n1+1))
    mat_nn_ = np.zeros((b,n0+1,n1+1))
    for b_idx in np.arange(b):
        num_subline0 = int(desc0.shape[2])
        num_subline1 = int(desc1.shape[2])
        desc0_ = desc0[b_idx].T
        desc1_ = desc1[b_idx].T

        sum_sq0 = np.sum(np.square(desc0_), axis=1)[:,None]  # (1, 16, 128) -> (1, 128)
        sum_sq1 = np.sum(np.square(desc1_), axis=1)[:,None].T

        dmat = desc0_ @ desc1_.T
        dmat_tmp = np.ones((n0,n1)) * np.inf
        dmat_tmp[:num_subline0,:num_subline1] = (sum_sq0 + sum_sq1 - 2.0 * dmat).clip(min=0)


        # Get NN indices and scores.
        idx = np.argmin(dmat_tmp, axis=1)
        scores = dmat_tmp[np.arange(dmat_tmp.shape[0]), idx]
        # Threshold the NN matches.
        keep = scores < nn_thresh
        if is_mutual_NN:
            # Check if nearest neighbor goes both directions and keep those.
            idx2 = np.argmin(dmat_tmp, axis=0)
            keep_bi = np.arange(len(idx)) == idx2[idx]
            keep = np.logical_and(keep, keep_bi)
        idx = idx[keep]
        scores = scores[keep]
        # Get the surviving point indices.
        # m_idx1 = np.arange(desc1.shape[2])[keep]
        m_idx1 = np.arange(n0)[keep]
        m_idx2 = idx

        mat_nn_[b_idx, m_idx1, m_idx2] = 1

        unmatch0 = np.array(range(n0+1))
        unmatch0 = np.delete(unmatch0, m_idx1)
        unmatch1 = np.array(range(n1+1))
        unmatch1 = np.delete(unmatch1, m_idx2)
        mat_nn_[b_idx,unmatch0,-1] = 1
        mat_nn_[b_idx,-1,unmatch1] = 1
        mat_nn_[b_idx,-1,-1] = 1
    
    return mat_nn_

def nn_matcher_score(dist_mat, nn_thresh, is_mutual_NN=False):

    n0 = dist_mat.shape[0]
    n1 = dist_mat.shape[1]

    mat_nn_ = np.zeros((n0,n1))

    # num_subline0 = int(desc0.shape[2])
    # num_subline1 = int(desc1.shape[2])
    # desc0_ = desc0[b_idx].T
    # desc1_ = desc1[b_idx].T

    # sum_sq0 = np.sum(np.square(desc0_), axis=1)[:,None]  # (1, 16, 128) -> (1, 128)
    # sum_sq1 = np.sum(np.square(desc1_), axis=1)[:,None].T

    # dmat = desc0_ @ desc1_.T
    # dmat_tmp = np.ones((n0,n1)) * np.inf
    # dmat_tmp[:num_subline0,:num_subline1] = (sum_sq0 + sum_sq1 - 2.0 * dmat).clip(min=0)
    dmat_tmp = dist_mat.clip(min=0)


    # Get NN indices and scores.
    idx = np.argmin(dmat_tmp, axis=1)
    scores = dmat_tmp[np.arange(dmat_tmp.shape[0]), idx]
    # Threshold the NN matches.
    keep = scores < nn_thresh
    if is_mutual_NN:
        # Check if nearest neighbor goes both directions and keep those.
        idx2 = np.argmin(dmat_tmp, axis=0)
        keep_bi = np.arange(len(idx)) == idx2[idx]
        keep = np.logical_and(keep, keep_bi)
    idx = idx[keep]
    scores = scores[keep]
    # Get the surviving point indices.
    # m_idx1 = np.arange(desc1.shape[2])[keep]
    m_idx1 = np.arange(n0)[keep]
    m_idx2 = idx

    mat_nn_[m_idx1, m_idx2] = 1

    # unmatch0 = np.array(range(n0+1))
    # unmatch0 = np.delete(unmatch0, m_idx1)
    # unmatch1 = np.array(range(n1+1))
    # unmatch1 = np.delete(unmatch1, m_idx2)
    # mat_nn_[unmatch0,-1] = 1
    # mat_nn_[-1,unmatch1] = 1
    # mat_nn_[-1,-1] = 1

    return mat_nn_