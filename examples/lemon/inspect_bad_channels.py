"""Manually identify bad channels in the LEMON EEG data

NOTE: In LEMON EEG, we noticed that there can be a few bad channels that are difficult to 
detect for some subjects. For these subjects, bad channels were first identified and then 
manually dropped using the custom `detect_bad_channels_manual()` function during the 
preprocessing step.
"""

# Install dependencies
import os
import mne
import numpy as np
import matplotlib.pyplot as plt

# INSPECT BAD CHANNELS
if __name__ == "__main__":
    # Define subject IDs of interest
    subject_ids = ["038", "042", "062", "077", "080", "087", "088", "090", "091", "155", 
                   "157", "165", "166", "168", "201", "216", "218", "228", "231", "240",
                   "249", "250", "257", "261", "263", "265", "267", "271", "274", "275", "276",
                   "282", "284", "290", "302", "303", "307", "310", "315", "316", "318", "319"]
    subject_ids = [f"sub-010{id}" for id in subject_ids]

    # Define modality
    modality = "eeg"

    # Set file path
    preproc_file_path = "/well/woolrich/projects/lemon/scho23/preproc"
    
    for id in subject_ids:
        # Load raw signal
        preproc_file_name = os.path.join(preproc_file_path, f"{id}/{id}_preproc_raw.fif")
        raw = mne.io.read_raw_fif(preproc_file_name)
        raw.pick_types(eeg=True)
        ch_names = np.array(raw.info["ch_names"])
        x = raw.get_data(picks=[modality], reject_by_annotation="omit")
        n_channels = x.shape[0]
        print(f"{id} - # of channels: {n_channels}")
        print(f"{id} - Data shape: {x.shape}")
        
        # Plot and detect bad channel outliers
        fig, ax = plt.subplots(nrows=1, ncols=1)
        Pxx = []
        lines = []
        for ch_idx in np.arange(n_channels):
            pxx, freqs, line = ax.psd(x[ch_idx, :], Fs=raw.info["sfreq"], return_line=True)
            Pxx.append(pxx)
            lines.append(line[0])
        Pxx = np.array(Pxx) # dim: (n_channels, n_freqs)
        Pxx_dB = 10 * np.log10(Pxx) # convert to log scale
        Pxx_dB_mean = np.mean(Pxx_dB, axis=1) # sum over frequencies
        
        order = np.argsort(Pxx_dB_mean)
        ch_names = ch_names[order]
        Pxx_dB = Pxx_dB[order, :]
        lines = list(np.array(lines)[order])
        
        k = 0
        for ch_idx in np.arange(n_channels):
            if ch_idx < 2 or ch_idx > n_channels - 3:
                lines[ch_idx].set_color("red")
                ax.text(freqs[-1] - k, Pxx_dB[ch_idx, -1], ch_names[ch_idx], color="tab:orange", fontsize=8)
                k += 5
            else:
                lines[ch_idx].set_color("black")
        plt.savefig(f"bad_channels_{id}.png")

    print("Analysis complete.")
