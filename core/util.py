""" General Utility Functions and Parameters """

# Standard Modules
# TODO

# External Modules
import numpy as np
from typing import *

# Internal Modules
# TODO

# Types to allow for both CPU and GPU models.
UFloatTensor = Union[FloatTensor, cuda.FloatTensor]
ULongTensor = Union[LongTensor, cuda.LongTensor]

def knn_indices_func_cpu(rep_pts : FloatTensor,  # (N, pts, dim)
                         pts : FloatTensor,      # (N, x, dim)
                         K : int, D : int
                        ) -> LongTensor:         # (N, pts, K)
    """
    EXAMPLE FUNCTION. Imports not included for this.

    CPU-based Indexing function based on K-Nearest Neighbors search.
    :param rep_pts: Representative points.
    :param pts: Point cloud to get indices from.
    :param K: Number of nearest neighbors to collect.
    :param D: "Spread" of neighboring points.
    :return: Array of indices, P_idx, into pts such that pts[n][P_idx[n],:]
    is the set k-nearest neighbors for the representative points in pts[n].
    """
    rep_pts = rep_pts.data.numpy()
    pts = pts.data.numpy()
    region_idx = []

    for n, p in enumerate(rep_pts):
        P_particular = pts[n]
        nbrs = NearestNeighbors(D*K + 1, algorithm = "ball_tree").fit(P_particular)
        indices = nbrs.kneighbors(p)[1]
        region_idx.append(indices[:,1::D])

    region_idx = torch.from_numpy(np.stack(region_idx, axis = 0))
    return region_idx
