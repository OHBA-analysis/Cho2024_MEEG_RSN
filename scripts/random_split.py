"""Randomly split the datasets in half

"""

# Set up dependencies
import os
import numpy as np
from utils.array_ops import split_half
from utils.data import load_data, load_meta_data, save_data
from utils.visualize import plot_age_distributions


if __name__ == "__main__":
    # Set random seed
    n_seed = 2023
    np.random.seed(n_seed)

    # Set directory paths
    BASE_DIR = "/well/woolrich/users/olt015/Cho2023_EEG_RSN"
    SAVE_DIR = BASE_DIR + "/data"

    # Load meta data and subject IDs
    eeg_meta_data = load_meta_data(modality="eeg")
    meg_meta_data = load_meta_data(modality="meg")
    age_group_idx = load_data(os.path.join(BASE_DIR, "data/age_group_idx.pkl"))

    # Get subject IDs for each age group
    get_subject_ids = lambda modality: (
        age_group_idx[modality]["subject_ids_young"], 
        age_group_idx[modality]["subject_ids_old"]
    )
    eeg_ids_y, eeg_ids_o = get_subject_ids("eeg")
    meg_ids_y, meg_ids_o = get_subject_ids("meg")

    # Split each age group in half
    eeg_ids_y1, eeg_ids_y2 = split_half(eeg_ids_y)
    eeg_ids_o1, eeg_ids_o2 = split_half(eeg_ids_o)

    meg_ids_y1, meg_ids_y2 = split_half(meg_ids_y)
    meg_ids_o1, meg_ids_o2 = split_half(meg_ids_o)

    # Save outputs
    output = {
        "eeg": {
            "split1": {"subject_ids_young": eeg_ids_y1, "subject_ids_old": eeg_ids_o1},
            "split2": {"subject_ids_young": eeg_ids_y2, "subject_ids_old": eeg_ids_o2},
        },
        "meg": {
            "split1": {"subject_ids_young": meg_ids_y1, "subject_ids_old": meg_ids_o1},
            "split2": {"subject_ids_young": meg_ids_y2, "subject_ids_old": meg_ids_o2},
        },
    }
    save_data(output, os.path.join(SAVE_DIR, "age_group_split_idx.pkl"))

    # Visualize age distributions
    verbose = False
    if verbose:
        get_eeg_age = lambda meta, ids: np.array([meta.loc[meta["ID"] == id]["Age"].values[0] for id in ids])
        get_meg_age = lambda meta, ids: np.array([meta.loc[meta["participant_id"] == id]["age"].values[0] for id in ids])

        # For EEG LEMON
        plot_age_distributions(
            get_eeg_age(eeg_meta_data, eeg_ids_y1),
            get_eeg_age(eeg_meta_data, eeg_ids_o1),
            modality="eeg",
            filename="first_half_eeg.png",
        )
        plot_age_distributions(
            get_eeg_age(eeg_meta_data, eeg_ids_y2),
            get_eeg_age(eeg_meta_data, eeg_ids_o2),
            modality="eeg",
            filename="second_half_eeg.png",
        )

        # For MEG Cam-CAN
        plot_age_distributions(
            get_meg_age(meg_meta_data, meg_ids_y1),
            get_meg_age(meg_meta_data, meg_ids_o1),
            modality="meg",
            filename="first_half_meg.png",
        )
        plot_age_distributions(
            get_meg_age(meg_meta_data, meg_ids_y2),
            get_meg_age(meg_meta_data, meg_ids_o2),
            modality="meg",
            filename="second_half_meg.png",
        )

    print("Process complete.")