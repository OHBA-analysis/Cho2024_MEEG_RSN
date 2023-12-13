"""Functions for static post-hoc analysis

"""

from osl_dynamics.analysis import static
from osl_dynamics.data import Data

def compute_aec(input_data, Fs, freq_range, tmp_dir):
    """Compute subject-level AEC matrices.

    Parameters
    ----------
    input_data : list of np.ndarray
        Input data containing raw time series.
    Fs : int
        Sampling frequency of the measured data.
    freq_range : list of int
        Upper and lower frequency bounds for filtering signals to calculate
        amplitude envelope.
    tmp_dir : str
        Path to a temporary directory for building a traning dataset.
        For further information, see data.Data() in osl-dynamics package.

    Returns
    -------
    conn_map : np.ndarray
        AEC functional connectivity matrix. Shape is (n_subjects, n_channels, n_channels).
    """

    # Compute amplitude envelope
    data = Data(input_data, store_dir=tmp_dir, sampling_frequency=Fs)
    data.prepare(
        methods = {
            "filter": {"low_freq": freq_range[0], "high_freq": freq_range[1]},
            "amplitude_envelope": {},
            "standardize": {},
        }
    )
    ts = data.time_series()

    # Calculate functional connectivity using AEC
    conn_map = static.functional_connectivity(ts, conn_type="corr")

    # Clean up
    data.delete_dir()

    return conn_map