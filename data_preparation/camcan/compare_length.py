"""Compare data length of the LEMON and CamCAN datasets

"""

# Set up dependencies
import os
import pickle
import numpy as np
from sys import argv
from osl_dynamics import data

# Define custom functions
def measure_data_length(dataset, sampling_frequency=None):
    """Get the length of each data in a given dataset.

    Parameters
    ----------
    dataset : osl_dynamics.data.base.Data
        Dataset containing data time series.
    sampling_frequency : int
        Sampling frequency of the data. Defaults to None.
        If None, 1 is used.

    Returns
    -------
    time_series_length : list of float
        List of the lengths of each data time series.
    """

    # Validation
    sampling_frequency = sampling_frequency or 1

    # Store lengths of time series
    time_series_length = []
    for ts in dataset.arrays:
        time_series_length.append(len(ts) / sampling_frequency)

    return time_series_length


if __name__ == "__main__":
    # Set hyperparameters
    if len(argv) != 2:
        print("Need to pass one argument: data space (e.g., python script.py sensor)")
        exit()
    data_space = argv[1]
    print(f"[INFO] Data Space: {data_space}")

    # Set directory paths
    PROJECT_DIR = "/well/woolrich/projects"
    eeg_data_dir = PROJECT_DIR + "/lemon/scho23"
    meg_data_dir = PROJECT_DIR + "/camcan/scho23"
    meg_meta_dir = PROJECT_DIR + "/camcan/participants.tsv"
    SAVE_DIR = PROJECT_DIR + "/camcan/scho23/data"
    TMP_DIR = os.path.join(SAVE_DIR, "tmp")
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Load group information
    with open(os.path.join(SAVE_DIR, "age_group_idx.pkl"), "rb") as input_path:
        age_group_idx = pickle.load(input_path)
    input_path.close()

    eeg_subject_ids = sorted(np.concatenate((age_group_idx["eeg"]["subject_ids_young"], age_group_idx["eeg"]["subject_ids_old"])))
    meg_subject_ids = sorted(np.concatenate((age_group_idx["meg"]["subject_ids_young"], age_group_idx["meg"]["subject_ids_old"])))

    # Load data
    print("Loading data ...")
    eeg_file_names, meg_file_names = [], []
    
    if data_space == "source":
        # Get file paths
        for id in eeg_subject_ids:
            eeg_file_names.append(os.path.join(eeg_data_dir, f"src_ec/{id}/sflip_parc-raw.npy"))
        for id in meg_subject_ids:
            meg_file_names.append(os.path.join(meg_data_dir, f"src/{id}/sflip_parc-raw.fif"))
        
        # Build training data
        eeg_data = data.Data(eeg_file_names, store_dir=TMP_DIR)
        meg_data = data.Data(meg_file_names, picks=["misc"], reject_by_annotation='omit', store_dir=TMP_DIR)

    elif data_space == "sensor":
        # Get file paths
        for id in eeg_subject_ids:
            eeg_file_names.append(os.path.join(eeg_data_dir, f"preproc_ec/{id}/{id}_preproc_raw.npy"))
        for id in meg_subject_ids:
            meg_file_names.append(os.path.join(meg_data_dir, f"preproc/mf2pt2_{id}_ses-rest_task-rest_meg/mf2pt2_{id}_ses-rest_task-rest_meg_preproc_raw.fif"))
        # NOTE: Only preprocessed data of subjects with corresponding source reconsturcted data will 
        #       be included here.

        # Build training data
        eeg_data = data.Data(eeg_file_names, store_dir=TMP_DIR)
        meg_data = data.Data(meg_file_names, picks=["meg"], reject_by_annotation='omit', store_dir=TMP_DIR)

    # Validation
    if len(eeg_file_names) != len(meg_file_names):
        raise ValueError("number of subjects in each dataset should be the same.")

    # Get data lengths
    Fs = 250 # sampling frequency    
    eeg_data_len = measure_data_length(eeg_data, sampling_frequency=Fs)
    meg_data_len = measure_data_length(meg_data, sampling_frequency=Fs)

    # Print out the summary
    print("*** EEG Data Length (Eyes Closed) ***")
    print(f"\tTotal # of subjects: {len(eeg_data_len)}")
    print("\tMean: {} (s) | Std: {} (s)".format(
        np.mean(eeg_data_len),
        np.std(eeg_data_len),
    ))

    print("*** MEG Data Length ***")
    print(f"\tTotal # of subjects: {len(meg_data_len)}")
    print("\tMean: {} (s) | Std: {} (s)".format(
        np.mean(meg_data_len),
        np.std(meg_data_len),
    ))

    print(f"Mean data length ratio of EEG to MEG: {np.mean(eeg_data_len) / np.mean(meg_data_len)}")
    # NOTE: Ideally, the sensor and source data should output identical results.

    # Clean up
    eeg_data.delete_dir()
    meg_data.delete_dir()

    print("Analysis complete.")
