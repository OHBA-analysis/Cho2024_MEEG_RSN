# Data Preparation Pipeline

### 🔎 Overview

This directory contains the pre-processing and source reconstruction pipelines for the datasets we used.
As described in the paper, the source reconstruction procedure encompasses two distinct approaches:

1. Data reconstruction using individual structrual MRI (sMRI) images
2. Data reconstruction using the standard MNI tempalte (`MNI152_T1_1mm.nii.gz`)

For detailed descriptions of each script, please refer to the sections below.
Note that these scripts heavily depend on the OSL software package.

## 🥫 MEG Cam-CAN

For the MEG Cam-CAN dataset, we have four main scripts and a supplementary script<sup>‡</sup>:

| Scripts                         | Description                                                                                       |
| :------------------------------ | :------------------------------------------------------------------------------------------------ |
| `1_preprocessing.py`            | Pre-processes sensor space data (after maxfiltering).                                             |
| `2_coregister.py`               | Coregister the sensor data with individual sMRI files.                                            |
| `3_source_reconstruct.py`       | Source reconsturction with individual sMRI files (beamforming, parcellation).                     |
| `3‑1_source_reconstruct.py`     | Source reconstruction with the standard MNI template (coregistration, beamforming, parcellation). |
| `4_sign_flip.py`                | Apply dipole sign flipping on source space data.                                                  |
| `compare_length.py`<sup>‡</sup> | Compare sample lengths of the age-matched LEMON and Cam-CAN datasets.                             |

1. For the individual sMRI reconstruction approach, follow the pipeline: `1 → 2 → 3 → 4`.
2. For the MNI template reconstruction approach, follow the pipeline: `1 → 3‑1 → 4`.

**NOTE #1:** For the first approach, note that the source reconstruction process is split into two separate scripts. 
This division was necessary to implement a customised coregistration functionality, which accommodates individual sMRI image properties 
(e.g., whether a subject has a nose MRI data scanned or not).

**NOTE #2:** To run the script `compare_length.py`, you should complete the data preparation steps for the EEG LEMON data first.

## 🍋 EEG LEMON

For the EEG LEMON dataset, we have five main scripts and two supplementary scripts<sup>‡</sup>:

| Scripts                               | Description                                                                                     |
| :------------------------------------ | :---------------------------------------------------------------------------------------------- |
| `1_preprocessing.py`                  | Pre-processes sensor space data (after maxfiltering).                                           |
| `2_source_reconstruct.py`             | Source reconstruction with individual sMRI files or the standard MNI template (coregistration, bemaforming, parcellation). |
| `3_sign_flip.py`                      | Apply dipole sign flipping on source space data.                                                |
| `4_prepare_task_data.sh`              | Reorganises the file structure and selects eyes-closed segments of the EEG data.                |
| `4‑1_prepare_no_struct.sh`            | Replicates `4_prepare_task_data.sh` but for the data reconstructed with the MNI template.       |
| `5_match_age_distribution.py`         | Sub-sample M/EEG data base on the dataset age distributions.                                    |
| `inspect_bad_channels.py`<sup>‡</sup> | Allows the manual inspection of bad channels in subject-wise EEG data.                          |
| `select_events.py`<sup>‡</sup>        | Extracts eyes-closed segments from the EEG data; used in the `4*.sh` scripts.                   |

1. For the individual sMRI reconstruction approach, follow the pipeline: `1 → 2 → 3 → 4 → 5`.
2. For the MNI template reconstruction approach, follow the pipeline: `1 → 2 → 3 → 4‑1 → 5`.
