"""Preparing the LEMON EEG data
[STEP 1] Preprocessing sensor space data

NOTE: This script was adapted from Quinn2022_GLMSpectrum `lemon_preproc.yml`.
"""

# Install dependencies
import os
import glob
import mne
import logging
import numpy as np
from scipy import io
from pathlib import Path
from osl import preprocessing, utils
from dask.distributed import Client

# Start logger
logger = logging.getLogger("osl")

# Set directories
BASE_DIR = "/well/woolrich/projects/lemon"
RAW_DIR = os.path.join(BASE_DIR, "raw")
LOCALIZER_DIR = os.path.join(RAW_DIR, "EEG_MPILMBB_LEMON/EEG_Localizer_BIDS_ID")
PREPROC_DIR = os.path.join(BASE_DIR, "scho23/preproc")

# Set file paths to raw data
RAW_FILE = RAW_DIR + "/{0}/RSEEG/{0}.vhdr"
LOCALIZER_FILE = LOCALIZER_DIR + "/{0}/{0}.mat"

# Configure pipeline
config = """
    preproc:
        - lemon_set_channel_montage: {}
        - lemon_create_heog: {}
        - set_channel_types: {VEOG: eog, HEOG: eog}
        - crop: {tmin: 30}
        - filter: {l_freq: 0.5, h_freq: 125, method: iir, iir_params: {order: 5, ftype: butter}}
        - notch_filter: {freqs: 50 100}
        - resample: {sfreq: 250}
        - bad_segments: {segment_len: 500, picks: eeg, significance_level: 0.1}
        - bad_segments: {segment_len: 500, picks: eeg, mode: diff, significance_level: 0.1}
        - bad_channels: {picks: eeg, significance_level: 0.1}
        - detect_bad_channels_manual: {}
        - bad_segments: {segment_len: 2500, picks: eog}
        - lemon_ica: {picks: eeg, n_components: 52}
        - interpolate_bads: {}
        - set_eeg_reference: {projection: true}
"""

# Define extra functions
def lemon_set_channel_montage(dataset, userargs):
    subject = Path(dataset["raw"]._filenames[0]).stem
    loc_file = LOCALIZER_FILE.format(subject)
    X = io.loadmat(loc_file, simplify_cells=True)
    ch_pos = {}
    for i in range(len(X["Channel"]) - 1):  # final channel is reference
        key = X["Channel"][i]["Name"].split("_")[2]
        if key[:2] == "FP":
            key = "Fp" + key[2]
        value = X["Channel"][i]["Loc"]
        ch_pos[key] = value
    hp = X["HeadPoints"]["Loc"]
    nas = np.mean([hp[:, 0], hp[:, 3]], axis=0)
    lpa = np.mean([hp[:, 1], hp[:, 4]], axis=0)
    rpa = np.mean([hp[:, 2], hp[:, 5]], axis=0)
    dig = mne.channels.make_dig_montage(ch_pos=ch_pos, nasion=nas, lpa=lpa, rpa=rpa)
    dataset["raw"].set_montage(dig)
    return dataset

def lemon_create_heog(dataset, userargs):
    F7 = dataset["raw"].get_data(picks="F7")
    F8 = dataset["raw"].get_data(picks="F8")
    heog = F7 - F8
    info = mne.create_info(["HEOG"], dataset["raw"].info["sfreq"], ["eog"])
    eog_raw = mne.io.RawArray(heog, info)
    dataset["raw"].add_channels([eog_raw], force_update_info=True)
    return dataset

def lemon_ica(dataset, userargs, logfile=None):
    ica = mne.preprocessing.ICA(
        n_components=userargs["n_components"], max_iter=1000, random_state=42
    )
    fraw = dataset["raw"].copy().filter(l_freq=1.0, h_freq=None)
    ica.fit(fraw, picks=userargs["picks"])
    dataset["ica"] = ica
    logger.info("Starting EOG autoreject")
    # Find and exclude VEOG
    veog_indices, eog_scores = dataset["ica"].find_bads_eog(dataset["raw"], "VEOG")
    dataset["veog_scores"] = eog_scores
    dataset["ica"].exclude.extend(veog_indices)
    logger.info(
        "Marking {0} ICs as VEOG {1}".format(len(dataset["ica"].exclude), veog_indices)
    )
    # Find and exclude HEOG
    heog_indices, eog_scores = dataset["ica"].find_bads_eog(dataset["raw"], "HEOG")
    dataset["heog_scores"] = eog_scores
    dataset["ica"].exclude.extend(heog_indices)
    logger.info("Marking {0} ICs as HEOG {1}".format(len(heog_indices), heog_indices))
    # Save components as channels in raw object
    src = dataset["ica"].get_sources(fraw).get_data()
    veog = src[veog_indices[0], :]
    heog = src[heog_indices[0], :]
    ica.labels_["top"] = [veog_indices[0], heog_indices[0]]
    info = mne.create_info(
        ["ICA-VEOG", "ICA-HEOG"], dataset["raw"].info["sfreq"], ["misc", "misc"]
    )
    eog_raw = mne.io.RawArray(np.c_[veog, heog].T, info)
    dataset["raw"].add_channels([eog_raw], force_update_info=True)
    # Apply ICA denoising or not
    if ("apply" not in userargs) or (userargs["apply"] is True):
        logger.info("Removing selected components from raw data")
        dataset["ica"].apply(dataset["raw"])
    else:
        logger.info("Components were not removed from raw data")
    return dataset

def detect_bad_channels_manual(dataset, userargs, logfile=None):
    target = userargs.pop("target", "raw")
    logger.info("OSL Stage - {0} : {1}".format(target, "detect_bad_channels_manual"))
    logger.info("userargs: {0}".format(str(userargs)))

    # Get data filename
    filenames = dataset["raw"]._filenames

    # Define bad channels for specific subjects
    bad_ch_info = {
        "sub-010038": ["PO10"], "sub-010042": ["FC1"], "sub-010062": ["FC1"], "sub-010077": ["FC1", "FC3", "F1", "Fz", "F3"],
        "sub-010080": ["PO9"], "sub-010087": ["Cz"], "sub-010088": ["Fz", "Cz", "FT7", "FC5", "PO10"], "sub-010090": ["PO10"],
        "sub-010091": ["TP7", "TP8"], "sub-010155": ["FC2", "F7"], "sub-010157": ["Fz", "AFz"],
        "sub-010165": ["CP6"], "sub-010166": ["F7"], "sub-010168": ["F7"], "sub-010201": ["POz", "PO10"],
        "sub-010216": ["T7"], "sub-010218": ["T8", "C6"], "sub-010228": ["F6"], "sub-010231": ["FC3"],
        "sub-010240": ["F7"], "sub-010249": ["Cz"], "sub-010250": ["PO10"], "sub-010257": ["Cz"],
        "sub-010261": ["Cz"], "sub-010263": ["CP6"], "sub-010265": ["FC2", "F7"], "sub-010267": ["F7"],
        "sub-010271": ["Cz", "F5"], "sub-010274": ["CP6"], "sub-010275": ["CP6"], "sub-010276": ["AFz", "FC1", "T8"],
        "sub-010282": ["Cz", "CPz"], "sub-010284": ["Cz"], "sub-010290": ["Cz"], "sub-010302": ["Fp2"],
        "sub-010303": ["TP7"], "sub-010307": ["CP1"], "sub-010310": ["T7"], "sub-010315": ["CP1", "T7"],
        "sub-010316": ["TP8"], "sub-010318": ["FC3"], "sub-010319": ["CP2", "F2"],
    }       

    # Concatenate manually found bad channels to existing list
    s = "Manual bad channel detection - {0} channels appended as bad channels."
    bad_subject_ids = list(bad_ch_info.keys())
    subject_id = filenames[0].split("/")[-3]
    if len(filenames) > 1:
        raise ValueError(f"there is more than one filename: {filenames}")
    if subject_id in bad_subject_ids:
        for ch_name in bad_ch_info[subject_id]:
            if ch_name not in dataset["raw"].info["bads"]:
                dataset["raw"].info["bads"].extend([ch_name])
        logger.info(s.format(bad_ch_info[subject_id]))

    return dataset

# PREPROCESS DATA
if __name__ == "__main__":
    # Set logger
    utils.logger.set_up(level="INFO")

    # Match subjects to localizer data
    subject_ids = [file.split("/")[-1] for file in sorted(glob.glob(os.path.join(LOCALIZER_DIR, "sub-*")))]
    print(f"Number of available subjects: {len(subject_ids)}")

    # Exclude subjects with bad signal quality
    print("Removing subjects with bad EEG quality ...")
    subjs_to_exclude = ["sub-010285", "sub-010300"] # problem with recording time or measurement scale
    subjs_to_exclude += ["sub-010052", "sub-010061", "sub-010065", "sub-010070", "sub-010073", 
                         "sub-010074", "sub-010207", "sub-010287", "sub-010288"] # problem with digitization
    print(f"Subjects w/ bad EEG quality (n={len(subjs_to_exclude)}): {subjs_to_exclude}")
    valid_subject_ids = [id for id in subject_ids if id not in set(subjs_to_exclude)]

    # Get file paths
    inputs = []
    for subject in valid_subject_ids:
        raw_file = os.path.join(RAW_DIR, f"{subject}/RSEEG/{subject}.vhdr")
        if os.path.exists(raw_file):
            inputs.append(raw_file)
        else:
            print(f"[NOTICE] File {raw_file} does not exist.")
    print(f"Number of available subjects (after exclusion): {len(inputs)}")

    # Set up parallel processing
    client = Client(n_workers=8, threads_per_worker=1)

    # Initiate preprocessing
    preprocessing.run_proc_batch(
        config,
        inputs,
        outdir=PREPROC_DIR,
        overwrite=True,
        extra_funcs=[
            lemon_set_channel_montage,
            lemon_create_heog,
            lemon_ica,
            detect_bad_channels_manual,
        ],
        dask_client=True,
    )

    print("Preprocessing complete.")
