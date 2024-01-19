"""Qualitative comparison of dynamic M/EEG resting-state networks

"""

# Set up dependencies
import os
import numpy as np
from sys import argv
from osl_dynamics import analysis
from osl_dynamics.inference import modes
from utils.data import load_data, save_data, load_order
from utils.visualize import DynamicVisualizer


if __name__ == "__main__":
    # ------- [1] ------- #
    #      Settings       #
    # ------------------- #
    print("*** STEP 1: SETTINGS ***")

    # Set hyperparameters
    if len(argv) != 5:
        print("Need to pass four arguments: data modality, number of states, run ID, and data type " 
              + "(e.g., python script.py eeg 6 0 full)")
        exit()
    modality = argv[1] # data modality
    n_states = int(argv[2]) # number of states
    run_id = int(argv[3])
    data_type = argv[4]
    if modality not in ["eeg", "meg"]:
        raise ValueError("modality should be either 'eeg' or 'meg'.")
    if data_type not in ["full", "split1", "split2"]:
        raise ValueError("invalid data type.")
    print(f"[INFO] Data Modality: {modality.upper()} | State #: {n_states} | Run ID: run{run_id} "
          + f"| Data Type: {data_type}")

    # Define dataset name
    if modality == "eeg":
        data_name = "lemon"
    else: data_name = "camcan"

    # Set directory paths
    BASE_DIR = "/well/woolrich/users/olt015/Cho2023_EEG_RSN"
    DATA_DIR = BASE_DIR + f"/results/dynamic/{data_name}/state{n_states}/run{run_id}"
    if data_type != "full":
        DATA_DIR = DATA_DIR.replace("dynamic", f"reprod/{data_type}")

    # Load data
    data = load_data(os.path.join(DATA_DIR, f"model/results/{data_name}_hmm.pkl"))
    alpha = data["alpha"]
    ts = data["training_time_series"]

    # Get state orders for the specified model run
    order = load_order(modality, n_states, data_type, run_id)

    # Load group information
    print("(Step 1-1) Loading subject information ...")
    if data_type == "full":
        age_group_idx = load_data(os.path.join(BASE_DIR, "data/age_group_idx.pkl"))
        subject_ids_young = age_group_idx[modality]["subject_ids_young"]
        subject_ids_old = age_group_idx[modality]["subject_ids_old"]
    else:
        age_group_idx = load_data(os.path.join(BASE_DIR, "data/age_group_split_idx.pkl"))
        subject_ids_young = age_group_idx[modality][data_type]["subject_ids_young"]
        subject_ids_old = age_group_idx[modality][data_type]["subject_ids_old"]
    subject_ids = np.concatenate((subject_ids_young, subject_ids_old))
    print("Total # of subjects: {} (Young: {}, Old: {})".format(
        len(subject_ids),
        len(subject_ids_young),
        len(subject_ids_old),
    ))

    # Validation
    if len(alpha) != len(subject_ids):
        raise ValueError(f"the length of alphas does not match the number of subjects.")
    
    # ------------------ [2] ------------------ #
    #      Preprocess inferred parameters       #
    # ----------------------------------------- #
    print("\n*** STEP 2: PREPROCESS INFERRED PARAMETERS ***")
    
    # Reorder states if necessary
    if order is not None:
        print(f"Reordering HMM states ...")
        print(f"\tOrder: {order}")
        alpha = [a[:, order] for a in alpha] # dim: (n_subjects, n_samples, n_states)

    # Get HMM state time courses
    stc = modes.argmax_time_courses(alpha)
    
    # --------------------- [3] --------------------- #
    #      Dynamic Network Feature Computations       #
    # ----------------------------------------------- #
    print("\n*** STEP 3: DYNAMIC NETWORK FEATURE COMPUTATIONS ***")

    save_path = os.path.join(
        BASE_DIR, "data",
        f"dynamic_{modality}_{n_states}states_run{run_id}_{data_type}.pkl"
    )

    if os.path.exists(save_path):
        # Load dynamic network features
        print("(Step 3-1) Loading dynamic network features ...")
        dynamic_network_features = load_data(save_path)
        freqs, psds, cohs, weights, power_maps, conn_maps, gfo = dynamic_network_features.values()
    else:
        # Compute state-specific power spectra
        print("(Step 3-1) Computing state-specific PSDs ...")
        Fs = 250 # sampling frequency
        n_jobs = 16 # number of CPUs to use for parallel processing
        freqs, psds, cohs, weights = analysis.spectral.multitaper_spectra(
            data=ts,
            alpha=stc,
            sampling_frequency=Fs,
            time_half_bandwidth=4,
            n_tapers=7,
            frequency_range=[1.5, 45],
            return_weights=True,
            standardize=True,
            n_jobs=n_jobs,
        )
        # dim (psd): (n_subjects, n_states, n_channels, n_freqs)
        # dim (coh): (n_subjects, n_states, n_channels, n_channels, n_freqs)

        # Compute state-specific power maps
        print("(Step 3-2) Computing state-specific power maps ...")
        power_maps = dict()
        freq_ranges = [[1.5, 20], [1.5, 4], [4, 8], [8, 13], [13, 20]]
        freq_bands = ["wide", "delta", "theta", "alpha", "beta"]
        for n, freq_range in enumerate(freq_ranges):
            power_maps[freq_bands[n]] = analysis.power.variance_from_spectra(
                freqs, psds, frequency_range=freq_range,
            )
            # dim: (n_subjects, n_states, n_channels)

        # Compute state-specific coherence maps
        print("(Step 3-3) Computing state-specific coherence maps ...")
        conn_maps = analysis.connectivity.mean_coherence_from_spectra(
            freqs, cohs, frequency_range=[1.5, 20],
        )
        # dim: (n_subjects, n_states, n_channels, n_channels)

        # Get fractional occupancies to be used as weights
        fo = modes.fractional_occupancies(stc) # dim: (n_subjects, n_states)
        gfo = np.mean(fo, axis=0)

        # Save computed features
        print("(Step 3-4) Saving computed features ...")
        output = {
            "freqs": freqs,
            "psds": psds,
            "coherences": cohs,
            "weights": weights,
            "power_maps": power_maps,
            "conn_maps": conn_maps,
            "gfo": gfo,
        }
        save_data(output, save_path)

    # --------- [4] ---------- #
    #      Visualization       #
    # ------------------------ #
    print("\n*** STEP 4: VISUALIZATION ***")

    # Set up visualization tools
    DV = DynamicVisualizer()

    # Plot wide-band power maps (averaged over all subjects)
    print("(Step 4-1) Plotting power maps ...")
    power_maps = power_maps["wide"] - np.average(power_maps["wide"], axis=1, weights=gfo, keepdims=True)
    # dim: (n_subjects, n_states, n_channels)
    gpower_all = np.mean(power_maps, axis=0) # dim: (n_states, n_channels)

    DV.plot_power_map(
        power_map=gpower_all,
        filename=os.path.join(DATA_DIR, "maps/power_map.png"),
        fontsize=26,
        plot_kwargs={"symmetric_cbar": True},
    )

    # Plot wide-band coherence maps (averaged over all subjects)
    print("(Step 4-2) Plotting coherence maps ...")
    conn_maps -= np.average(conn_maps, axis=1, weights=gfo, keepdims=True)
    # dim: (n_subjects, n_states, n_channels, n_channels)
    gconn_all = np.mean(conn_maps, axis=0) # dim: (n_states, n_channels, n_channels)
    gconn_all = analysis.connectivity.threshold(
        gconn_all,
        percentile=95,
        absolute_value=True,
    ) # select top 5%

    DV.plot_coh_conn_map(
        connectivity_map=gconn_all,
        filename=os.path.join(DATA_DIR, "maps/conn_map.png"),
    )

    # Plot power spectral densities (averaged over all subjects)
    print("(Step 4-3) Plotting PSDs ...")
    psds -= np.average(psds, axis=1, weights=gfo, keepdims=True)
    # dim: (n_subjects, n_states, n_channels, n_freqs)
    gpsd_all = np.mean(psds, axis=(2, 0)) # dim: (n_states, n_freqs)
    gpsd_sem = np.std(np.mean(psds, axis=2), axis=0) / np.sqrt(len(psds)) # dim: (n_states, n_freqs)

    DV.plot_psd(
        freqs=freqs,
        psd=gpsd_all,
        error=gpsd_sem,
        filename=os.path.join(DATA_DIR, "maps/psd.png"),
    )

    print("Computation completed.")