# Cho2024_MEEG_RSN

This repository contains the scripts and data for reproducing results in the "Comparison between EEG and MEG of static and dynamic resting-state networks" manuscript.

💡 Please email SungJun Cho at sungjun.cho@psych.ox.ac.uk or simply raise GitHub Issues if you have any questions or concerns.

## ⚡️ Getting Started

This repository contains all the scripts necessary to reproduce the analysis and figures presented in the manuscript. It is divided into three main directories:

1. `data`: Contains the optimal HMM models trained on the MEG Cam-CAN and EEG LEMON datasets.
2. `data_preparation`: Contains the scripts for preprocessing and source reconstructing the sensor-level M/EEG data.
3. `scripts`: Contains the scripts for training TDE-HMM models and analysing static and dynamic M/EEG resting-state networks.

For further details, please refer to the README files within each folders.

NOTE: Most of the codes within this repository were executed on the Oxford Biomedical Research Computing (BMRC) servers. While individual threads were allocated varying CPUs and GPUs, general information about the BRMC resources can be found at [_Using the BMRC Cluster with Slurm_](https://www.medsci.ox.ac.uk/for-staff/resources/bmrc/using-the-bmrc-cluster-with-slurm) and [_GPU Resources_](https://www.medsci.ox.ac.uk/for-staff/resources/bmrc/gpu-resources).

## 🎯 Requirements
To start, you first need to install the [`OSL`](https://github.com/OHBA-analysis/osl) and [`osl-dynamics`](https://github.com/OHBA-analysis/osl-dynamics) packages and set up its environment. Its installation guide can be found [here](https://osl.readthedocs.io/en/latest/install.html).

The `seaborn` and `openpyxl` packages should be additionally installed for visualisation and compatibility with excel files. Once these steps are complete, download this repository to your designated folder location, and you're ready to begin!

The analyses and visualisations in this paper had following dependencies:

```
python==3.10.10
osl==0.5.1
osl-dynamics==1.2.6
seaborn==0.13.2
openpyxl==3.1.2
```

## 🪪 License
Copyright (c) 2024 [SungJun Cho](https://github.com/scho97) and [OHBA Analysis Group](https://github.com/OHBA-analysis). `Cho2024_MEEG_RSN` is a free and open-source software licensed under the [MIT License](https://github.com/scho97/CompareModality/blob/main/LICENSE).
