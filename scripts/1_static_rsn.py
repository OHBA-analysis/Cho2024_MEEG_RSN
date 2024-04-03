"""Qualitative comparison of static M/EEG resting-state networks

"""

# Set up dependencies
import os
import numpy as np
from sys import argv
from osl_dynamics.analysis import static, power, connectivity
from osl_dynamics.data import Data
from utils.data import load_data, save_data, get_raw_file_names
from utils.static import compute_aec
from utils.visualize import StaticVisualizer, _colormap_transparent


if __name__ == "__main__":
    # ------- [1] ------- #
    #      Settings       #
    # ------------------- #
    print("*** STEP 1: SETTINGS ***")

    # Set hyperparameters
    if len(argv) != 4:
        print("Need to pass two arguments: data modality, data type, and structural type "
         + "(e.g., python script.py eeg full subject)")
        exit()
    modality = argv[1] # data modality
    data_type = argv[2] # type of datasets to use
    structurals = argv[3] # type of structurals to use
    if modality not in ["eeg", "meg"]:
        raise ValueError("modality should be either 'eeg' or 'meg'.")
    if data_type not in ["full", "split1", "split2"]:
        raise ValueError("invalid data type.")
    print(f"[INFO] Data Modality: {modality.upper()} | Data Type: {data_type} | Structurals: {structurals}")

    # Define dataset name 
    if modality == "eeg":
        data_name = "lemon"
    else: data_name = "camcan"

    # Set directory paths
    BASE_DIR = "/well/woolrich/users/olt015/Cho2023_EEG_RSN"
    PROJECT_DIR = "/well/woolrich/projects"
    DATA_DIR = PROJECT_DIR + f"/{data_name}/scho23"
    SAVE_DIR = BASE_DIR + f"/results/static/{modality}"
    if data_type != "full":
        SAVE_DIR = SAVE_DIR.replace(
            f"static/{modality}", f"reprod/{data_type}/{data_name}/static"
        )
    if structurals == "standard":
        SAVE_DIR = SAVE_DIR.replace("static", "static_no_struct")
    TMP_DIR = SAVE_DIR + "/tmp"
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(TMP_DIR, exist_ok=True)

    # Load subject information
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

    # Get data files
    print("(Step 1-2) Getting data files ...")
    file_names = get_raw_file_names(DATA_DIR, subject_ids, modality, structurals)

    # Load subject-wise data arrays
    print("(Step 1-3) Loading data recordings ...")
    if modality == "eeg":
        training_data = Data(file_names, store_dir=TMP_DIR)
    if modality == "meg":
        training_data = Data(file_names, picks="misc", reject_by_annotation="omit", store_dir=TMP_DIR)
    
    input_data = [x for x in training_data.arrays]
    if input_data[0].shape[0] < input_data[0].shape[1]:
        input_data = [x.T for x in input_data]
        print("Data dimension reverted to (samples x channels).")
    input_n_samples = [d.shape[0] for d in input_data] # sample sizes of data arrays
    # NOTE: This can be used later to calculate weights for each group.
    
    training_data.delete_dir() # clean up

    print("Total # of parcels: ", input_data[0].shape[1])
    print("Shape of the single subject input data: ", np.shape(input_data[0]))

    # --------------------- [2] -------------------- #
    #      Static Network Feature Computations       #
    # ---------------------------------------------- #
    print("\n*** STEP 2: STATIC NETWORK FEATURE COMPUTATIONS ***")

    save_path = os.path.join(
        BASE_DIR, "data",
        f"static_network_features_{modality}_{data_type}.pkl"
    )
    if structurals == "standard":
        save_path = save_path.replace(".pkl", "_no_struct.pkl")
    
    if os.path.exists(save_path):
        # Load static network features
        print("(Step 2-1) Loading static network features ...")
        static_network_features = load_data(save_path)
        freqs = static_network_features["freqs"]
        psds = static_network_features["psds"]
        weights = static_network_features["weights"]
        power_maps = static_network_features["power_maps"]
        conn_maps = static_network_features["conn_maps"]
    else:
        # Compute subject-specific static power spectra
        print("(Step 2-1) Computing static PSDs ...")
        Fs = 250 # sampling frequency
        freqs, psds, weights = static.welch_spectra(
            data=input_data,
            sampling_frequency=Fs,
            window_length=int(Fs * 2),
            step_size=int(Fs),
            frequency_range=[1.5, 45],
            return_weights=True,
            standardize=True,
        )
        # dim: (n_subjects, n_parcels, n_freqs)

        # Compute subject-specific static power maps
        print("(Step 2-2) Computing static power maps ...")
        power_maps = dict()
        freq_ranges = [[1.5, 20], [1.5, 4], [4, 8], [8, 13], [13, 20]]
        freq_bands = ["wide", "delta", "theta", "alpha", "beta"]
        for n, freq_range in enumerate(freq_ranges):
            power_maps[freq_bands[n]] = power.variance_from_spectra(
                freqs, psds, frequency_range=freq_range
            )
            # dim: (n_subjects, n_parcels)

        # Compute subject-specific static AEC maps
        print("(Step 2-3) Computing static AEC maps ...")
        conn_maps = compute_aec(
            input_data, Fs, freq_range=[1.5, 20], tmp_dir=TMP_DIR
        )
        # dim: (n_subjects, n_parcles, n_parcels)

        # Save computed features
        print("(Step 2-4) Saving computed features ...")
        output = {
            "freqs": freqs,
            "psds": psds,
            "weights": weights,
            "power_maps": power_maps,
            "conn_maps": conn_maps,
        }
        save_data(output, save_path)

    # --------- [3] ---------- #
    #      Visualization       #
    # ------------------------ #
    print("\n*** STEP 3: VISUALIZATION ***")

    # Set up visualization tools
    SV = StaticVisualizer()
    cmap_hot_tp = _colormap_transparent("gist_heat")

    # Plot static wide-band power map (averaged over all subjects)
    gpower_all = np.mean(power_maps["wide"], axis=0) # dim: (n_parcels,)
    gpower_range = np.abs(max(gpower_all) - min(gpower_all))

    SV.plot_power_map(
        power_map=gpower_all,
        filename=os.path.join(SAVE_DIR, "power_map.png"),
        plot_kwargs={
            "vmin": 0,
            "vmax": max(gpower_all) + 0.1 * gpower_range,
            "symmetric_cbar": False,
            "cmap": cmap_hot_tp,
        },
    )
    
    # Plot static wide-band AEC map (averaged over all subjects)
    gconn_all = np.mean(conn_maps, axis=0) # dim: (n_parcels, n_parcels)
    gconn_all = connectivity.threshold(
        gconn_all,
        percentile=95
    ) # select top 5%

    SV.plot_aec_conn_map(
        connectivity_map=gconn_all,
        filename=os.path.join(SAVE_DIR, "conn_map.png"),
        colormap="Reds",
        plot_kwargs={"edge_vmin": 0, "edge_vmax": np.max(gconn_all)},
    )

    # Plot static power spectral densities (averaged over all subjects)
    gpsd_all = np.mean(psds, axis=(1, 0)) # dim: (n_freqs,)
    gpsd_sem = np.std(np.mean(psds, axis=1), axis=0) / np.sqrt(len(psds)) # dim: (n_freqs,)

    SV.plot_psd(
        freqs=freqs,
        psd=gpsd_all,
        error=gpsd_sem,
        filename=os.path.join(SAVE_DIR, "psd.png"),
    )

    print("Computation completed.")
