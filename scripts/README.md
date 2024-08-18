# Script Descriptions

### 🔎 Overview

This directory contains the scripts for model training and data analysis.
These scripts can be divided into three categories:

1. Data Inspection
   * For examining and organising datasets
2. Model Training
   * For training TDE-HMM models on datasets
3. Data Analysis
   * For analysing and visualising static and dynamic RSNs inferred by the trained HMM models

For detailed descriptions of each script, please refer to the sections below.
Note that these scripts heavily depend on the `osl-dynamics` software package.

## 🕵🏼 Data Inspection

We have two scripts for inspecting and organising the datasets:

| Scripts               | Description                                                                       |
| :-------------------- | :-------------------------------------------------------------------------------- |
| `get_demographics.py` | Summarises subject sex and handedness for each age group in the specific dataset. |
| `random_split.py`     | Randomly splits the dataset into two halves.                                      |

## ⚙️ Model Training

For training a TDE-HMM model, we use a separate script (`*_hmm.py`) for each dataset. 
When multiple models are trained, we can select the one that performs best by comparing their losses.

| Scripts                | Description                                                                                             |
| :--------------------- | :------------------------------------------------------------------------------------------------------ |
| `camcan_hmm.py`        | Prepare the MEG Cam-CAN dataset and train a TDE-HMM model on the data.                                  |
| `lemon_hmm.py`         | Prepare the EEG LEMON dataset and train a TDE-HMM model on the data.                                    |
| `run_model.sh`         | Example bash script for submitting a job to the Oxford BMRC server.                                     |
| `select_best_model.py` | Select the best model out of multiple model runs based on their losses (i.e., variational free energy). |

**NOTE:** When training an HMM on either Cam-CAN or LEMON datasets, you can specify whether to use the full set of subjects or split-half subsets. 
For a use case of how to randomly split a dataset in half, refer to the `random_split.py` script in [Data Inspection](#-data-inspection).

## 🧐 Data Analysis

For the data analysis, we have seven main scripts:

| Scripts                           | Description                                                                                                 |
| :-------------------------------- | :---------------------------------------------------------------------------------------------------------- |
| `1_static_rsn.py`                 | Compares static RSN features of the EEG and MEG data.                                                       |
| `2_dynamic_rsn.py`                | Compares dynamic RSN features of the EEG and MEG data.                                                      |
| `3_split_half_reproducibility.py` | Examines the reproducibility of the inferred RSNs within and across two data modalities.                    |
| `4_stataic_age_effects.py`        | Compares group-level age effects between the static RSNs of the EEG and MEG data.                           |
| `5_dynamic_age_effects.py`        | Compares group-level age effects between the dynamic RSNs of the EEG and MEG data.                          |
| `6_age_predictions.py`            | Predicts age groups with a logistic regression classifier using M/EEG RSN features.                         |
| `7_compare_predictions.py`        | Compares predictive accuracies across two modalities and between model runs with and without subject sMRIs. |

**NOTE:** Although the scripts are numbered and generally follow the order of figures in the paper, the numeric prefixes do not strictly 
correspond to figure numbers.

### 🙋‍♂️ FAQ: What about the `utils` subdirectory?
The `utils` subdirectory contains essential functions required to run the scripts summarised above. Each script in `utils` includes multiple 
functions. These functions are self-explanatory and include detailed annotations, so their descriptions are not repeated here.
