"""Functions for dynamic post-hoc analysis

"""

import numpy as np
from scipy.spatial.distance import jensenshannon
from osl_dynamics.inference import metrics

def between_state_rv_coefs(matrix1, matrix2):
    """
    Computes between-state RV coefficients.

    Parameters
    ----------
    matrix1 : np.ndarray
        Set of matrices for each state.
        Shape must be (n_states, n_channels, n_channels).
    matrix2 : np.ndarray
        Set of matrices for each state.
        Shape must be (n_states, n_channels, n_channels).

    Returns
    -------
    rv_coefs : np.ndarray
        Array of state-wise RV coefficients. Shape is (n_states,).
        Each element of the array stores a coefficient value between
        two matrices of each state.
    """

    # Validation
    if matrix1.shape != matrix2.shape:
        raise ValueError("Shape of two input matrices must be identical.")
    n_states = matrix1.shape[0]

    # Compute between-state RV coefficients
    rv_coefs = np.zeros((n_states,))
    for n in range(n_states):
        rv_coefs[n] = metrics.pairwise_rv_coefficient(
            np.stack((matrix1[n, :, :], matrix2[n, :, :]))
        )[0, 1]
    
    return rv_coefs

def js_divergence_matrix(matrix1, matrix2):
    """
    Computes the Jensen-Shannon (JS) divergence between two matrices.
    Here, we compute the JS distance for each pair of rows from two 
    matrices and then calculate the mean of resulting distances.

    Parameters
    ----------
    matrix1 : np.ndarray
        A first matrix with a shape (N, N).
    matrix2 : np.ndarray
        A second matrix with a shape (N, N).

    Returns
    -------
    mean_js_divergence : float
        The Jensen-Shannon distance between two matrices.
    """

    # Validation
    if matrix1.shape != matrix2.shape:
        raise ValueError("Shape of two input matrices must be identical.")
    n_states = matrix1.shape[0]

    # Calculate JS divergence for each pair of rows
    js_divergences = [
        jensenshannon(matrix1[n, :], matrix2[n, :]) for n in range(n_states)
    ]
    mean_js_divergence = np.mean(js_divergences)

    return mean_js_divergence