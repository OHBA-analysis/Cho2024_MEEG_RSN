"""Functions for dynamic post-hoc analysis

"""

import numpy as np
from scipy.spatial.distance import jensenshannon
from osl_dynamics.inference import metrics, modes

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

def compute_summary_statistics(state_time_course, sampling_frequency):
    """
    Computes four metrics of summary statistics from a given HMM state 
    time course.

    Parameters
    ----------
    state_time_course : list of np.ndarray
        State time courses (strictly binary). Shape must be (n_subjects, 
        n_samples, n_states).
    sampling_frequency : int
        Sampling frequency in Hz.
    
    Returns
    -------
    fo : np.ndarray
        Fractional occupancy of each state. Shape is (n_subjects, n_states).
    lt : np.ndarray
        Mean lifetime of each state. Shape is (n_subjects, n_states).
    intv : np.ndarray
        Mean interval of each state. Shape is (n_subjects, n_states).
    sr : np.ndarray
        Switching rate of each state. Shape is (n_subjects, n_states).
    """

    # Compute fractional occupancies
    fo = modes.fractional_occupancies(state_time_course)
    print(f"Shape of fractional occupancy: {fo.shape}")

    # Compute mean lifetimes
    lt = modes.mean_lifetimes(state_time_course, sampling_frequency)
    lt *= 1e3 # convert seconds to milliseconds
    print(f"Shape of mean lifetimes: {lt.shape}")
    
    # Compute mean intervals
    intv = modes.mean_intervals(state_time_course, sampling_frequency)
    print(f"Shape of mean intervals: {intv.shape}")

    # Compute switching rates
    sr = modes.switching_rates(state_time_course, sampling_frequency)
    print(f"Shape of switching rates: {sr.shape}")

    return fo, lt, intv, sr
