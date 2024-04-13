"""Subject demographics of the EEG LEMON and MEG Cam-CAN datasets

"""

# Set up dependencies
import os
import numpy as np
from sys import argv
from utils import data as Data


if __name__ == "__main__":
    # ------- [1] ------- #
    #      Settings       #
    # ------------------- #
    print("*** STEP 1: SETTINGS ***")

    # Set hyperparameters
    if len(argv) != 2:
        print("Need to pass one argument: data modality (e.g., python script.py eeg)")
        exit()
    modality = argv[1] # data modality
    if modality not in ["eeg", "meg"]:
        raise ValueError("modality should be either 'eeg' or 'meg'.")
    print(f"[INFO] Data Modality: {modality.upper()}")

    # Set directory paths
    BASE_DIR = "/well/woolrich/users/olt015/Cho2024_MEEG_RSN"

    # Load subject information
    print("(Step 1-1) Loading subject information ...")
    age_group_idx = Data.load_data(os.path.join(BASE_DIR, "data/age_group_idx.pkl"))
    subject_ids_young = age_group_idx[modality]["subject_ids_young"]
    subject_ids_old = age_group_idx[modality]["subject_ids_old"]
    subject_ids = np.concatenate((subject_ids_young, subject_ids_old))
    print("Total # of subjects: {} (Young: {}, Old: {})".format(
        len(subject_ids),
        len(subject_ids_young),
        len(subject_ids_old),
    ))

    # --------------- [2] -------------- #
    #      Demographics Extraction       #
    # ---------------------------------- #
    print("\n*** STEP 2: DEMOGRAPHICS EXTRACTION ***")

    # Load meta data
    meta_data = Data.load_meta_data(modality)

    for name, ids in zip(["young", "old"], [subject_ids_young, subject_ids_old]):
        print(f"Demographics for {name} subjects")
        # Get sex information for each age group
        sexes = Data.load_sex_information(ids, modality)
        n_female = np.count_nonzero(sexes)
        n_male = len(ids) - n_female
        print(f"\tSex (F/M): {n_female}/{n_male}")

        # Get handedness information for each age group
        handedness = Data.load_handedness_information(ids, modality)
        n_left = len(handedness[handedness == 0])
        n_right = len(handedness[handedness == 1])
        n_other = len(ids) - (n_left + n_right)
        print(f"\tHandedness (L/R/O): {n_left}/{n_right}/{n_other}")

    print("Analysis complete.")
