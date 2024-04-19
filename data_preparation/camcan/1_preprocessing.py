"""Preparing the Cam-CAN MEG data
[STEP 1] Preprocessing maxfiltered sensor space data
"""

# Install dependencies
import os
import pickle
import numpy as np
from osl import preprocessing, utils
from dask.distributed import Client

# Set directories
BASE_DIR = "/home/scho/camcan"
DATA_DIR = "/ohba/pi/mwoolrich/datasets/camcan"
RAW_DIR = os.path.join(DATA_DIR, "public/cc700/meg/pipeline/release005/BIDSsep/derivatives_rest/aa/AA_movecomp/aamod_meg_maxfilt_00002")
PREPROC_DIR = os.path.join(BASE_DIR, "scho23/preproc")

# Configure pipeline
config = """
    preproc:
        - crop: {tmin: 30}
        - filter: {l_freq: 0.5, h_freq: 125, method: iir, iir_params: {order: 5, ftype: butter}}
        - notch_filter: {freqs: 50 88 100, notch_widths: 2}
        - resample: {sfreq: 250}
        - bad_segments: {segment_len: 500, picks: mag, significance_level: 0.1}
        - bad_segments: {segment_len: 500, picks: grad, significance_level: 0.1}
        - bad_segments: {segment_len: 500, picks: mag, mode: diff, significance_level: 0.1}
        - bad_segments: {segment_len: 500, picks: grad, mode: diff, significance_level: 0.1}
        - bad_channels: {picks: mag, significance_level: 0.1}
        - bad_channels: {picks: grad, significance_level: 0.1}
        - bad_segments: {segment_len: 2500, picks: eog}
        - ica_raw: {picks: meg, n_components: 64}
        - ica_autoreject: {picks: meg, ecgmethod: correlation, eogthreshold: auto}
        - interpolate_bads: {}
"""

# PREPROCESS DATA
if __name__ == "__main__":
    # Set logger
    utils.logger.set_up(level="INFO")

    # Get subject IDs
    with open(os.path.join(BASE_DIR, "scho23/data/age_group_idx.pkl"), "rb") as input_path:
        data = pickle.load(input_path)
    subject_ids = sorted(np.concatenate((data["meg"]["subject_ids_young"], data["meg"]["subject_ids_old"])))
    print(f"Number of available subjects (age-matched with LEMON): {len(subject_ids)}")

    # Set up parallel processing
    client = Client(n_workers=8, threads_per_worker=1)

    # Exclude subjects with bad ICA quality
    print("Removing subjects with bad ICA quality ...")
    subjs_to_exclude = []
    for subject in subject_ids:
        if subject in ["sub-CC620567", "sub-CC620685", "sub-CC721114", "sub-CC721704"]:
            subjs_to_exclude.append(subject)
    print(f"Subjects w/ bad ICA quality (n={len(subjs_to_exclude)}): {subjs_to_exclude}")
    valid_subject_ids = [id for id in subject_ids if id not in set(subjs_to_exclude)]

    # Update file paths
    inputs = []
    for subject in valid_subject_ids:
        raw_file = os.path.join(
            RAW_DIR, f"{subject}/mf2pt2_{subject}_ses-rest_task-rest_meg.fif"
        )
        inputs.append(raw_file)
    print(f"Number of available subjects (after exclusion): {len(inputs)}")

    # Initiate preprocessing
    preprocessing.run_proc_batch(
        config,
        inputs,
        outdir=PREPROC_DIR,
        overwrite=True,
        dask_client=True,
    )

    print("Preprocessing complete.")
