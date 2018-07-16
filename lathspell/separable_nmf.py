"""
* Some of following code was lifted from Ankur Moitra: http://people.csail.mit.edu/moitra/software.html
* Our goal is implementing a parallelized version of this package, using Dask.
"""
import numpy as np
import dask.array as da

from scipy.sparse import csc_matrix
from math import sqrt


def calculate_Q(X: csc_matrix,
                update_in_place: bool = True) -> np.ndarray:
    """
    :param X: a CSC matrix of shape (rows=features, cols=observations).
    :param update_in_place: whether or not to return a new Numpy array or modify the input X.
    :return: the word-word correlation matrix Q as a dense Numpy ndarray.
    """

    n_features, n_observations = X.shape

    diagonal = np.zeros(n_features)

    if not update_in_place:
        X = X.copy()

    for col_idx in range(X.indptr.size - 1):
        col_start = X.indptr[col_idx]
        col_end = X.indptr[col_idx + 1]

        col_entries = X.data[col_start:col_end]
        col_sum = np.sum(col_entries)

        row_indices = X.indices[col_start:col_end]

        # TODO: figure out whether this loop and update by division can be written in more idiomatic Numpy
        diagonal[row_indices] += col_entries / (col_sum * (col_sum - 1))
        X.data[col_start:col_end] = col_entries / sqrt(col_sum * (col_sum - 1))

    Q = X * X.T / n_observations
    Q = np.array(Q.todense(), copy=False)

    diagonal = diagonal / n_observations
    Q = Q - np.diag(diagonal)

    return Q


def find_anchors(Q: np.ndarray,
                 k: int,
                 seed: int = None):
    pass
