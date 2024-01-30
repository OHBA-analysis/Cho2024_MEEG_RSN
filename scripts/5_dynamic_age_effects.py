"""Qualitative group-level comparison of dynamic M/EEG RSN features

"""

# Set up dependencies
import os
import numpy as np
from sys import argv
from osl_dynamics.inference import modes
from utils import visualize
from utils.data import load_data, load_order
from utils.dynamic import compute_summary_statistics
from utils.statistics import fit_glm, max_stat_perm_test


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

    # Get group information
    group_assignments = np.zeros(len(subject_ids),)
    group_assignments[len(subject_ids_young):] = 1 # Group 1: old
    group_assignments[:len(subject_ids_young)] = 2 # Group 2: young

    # Load HMM model data
    data = load_data(os.path.join(DATA_DIR, f"model/results/{data_name}_hmm.pkl"))
    alpha = data["alpha"]
    if len(alpha) != len(subject_ids):
        raise ValueError(f"the length of alphas does not match the number of subjects.")
    
    # Load subject-level network features
    dynamic_network_features = load_data(os.path.join(
        BASE_DIR, "data",
        f"dynamic_{modality}_{n_states}states_run{run_id}_{data_type}.pkl"
    ))
    freqs, psds, _, _, power_maps, _, gfo = dynamic_network_features.values()
    freq_bands = ["delta", "theta", "alpha", "beta"]
    
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

    # -------------------------- [3] ------------------------- #
    #      Group-Level Comparisons of Summary Statistics       #
    # -------------------------------------------------------- #
    print("\n*** STEP 3: QUANTIFYING AGE EFFECTS IN SUMMARY STATS ***")

    # Compute summary statistics
    print("(Step 3-1) Computing summary statistics ...")
    Fs = 250 # sampling frequency (Hz)
    fo, lt, intv, sr = compute_summary_statistics(stc, Fs)

    # Set statistical test parameters
    print("(Step 3-2) Running max-t permutation tests ...")
    bonferroni_ntest = 4 # n_test = n_metrics
    metric_names = ["fo", "lt", "intv", "sr"]
    metric_full_names = ["Fractional Occupancy", "Mean Lifetimes (ms)", "Mean Intervals (s)", "Swithching Rates"]
    pvalues_all = []

    for i, stat in enumerate([fo, lt, intv, sr]):
        print(f"[{metric_names[i].upper()}] Running Max-t Permutation Test ...")

        # Fit GLM on subject-level summary statistics
        summ_model, summ_design, summ_data = fit_glm(
            stat,
            subject_ids,
            group_assignments,
            modality=modality,
            dimension_labels=["Subjects", "States"],
            plot_verbose=False,
        )

        # Perform a max-t permutation test
        pval = max_stat_perm_test(
            summ_model,
            summ_data,
            summ_design,
            pooled_dims=1,
            contrast_idx=0,
            n_perm=10000,
            metric="tstats",
        )
        print(f"\tP-values (before correction): {pval}")

        # Implement Bonferroni correction
        pval *= bonferroni_ntest
        print(f"\tP-values (after correction): {pval}")
        print("\tSignificant states: ", np.arange(1, n_states + 1)[pval < 0.05])
        pvalues_all.append(pval)

    # Visualize violing plots for each state with if group difference exists
    pvalues_all = np.array(pvalues_all)
    sig_states = np.arange(n_states)[np.sum(pvalues_all < 0.05, axis=0) != 0]
    group_lbl = ["Young" if val == 2 else "Old" for val in group_assignments]

    for i, stat in enumerate([fo, lt, intv, sr]):
        for c in sig_states:
            print(f"Plotting a single grouped violing plot for State {c + 1} ...")
            visualize.plot_single_grouped_violin(
                data=stat[:, c],
                group_label=group_lbl,
                filename=os.path.join(DATA_DIR, f"analysis/{metric_names[i]}_{c}.png"),
                ylbl=metric_full_names[i],
                pval=pvalues_all[i, c],
            )

    # ------------------- [4] ------------------ #
    #      Group-Level Comparisons of PSDs       #
    # ------------------------------------------ #
    print("\n*** STEP 4: QUANTIFYING AGE EFFECTS IN PSDS ***")

    # Subtract the mean across states
    print("(Step 4-1) Subtracting the mean across states ...")
    psds -= np.average(psds, axis=1, weights=gfo, keepdims=True)
    # dim: (n_subjects, n_states, n_channels, n_freqs)
    # NOTE: The (static) mean across states is subtracted from the PSDs subject-wise.

    # Visualize group-level age effects in state-specific PSDs
    visualize.plot_state_spectra_group_diff(
        freqs,
        psds,
        subject_ids,
        group_assignments,
        modality=modality,
        bonferroni_ntest=n_states,
        filename=os.path.join(
            DATA_DIR, "analysis/psd_cluster_dynamic.png"
        ),
    )

    # ---------------------- [5] --------------------- #
    #      Group-Level Comparisons of Power Maps       #
    # ------------------------------------------------ #
    print("\n*** STEP 5: QUANTIFYING AGE EFFECTS IN POWER MAPS ***")

    DV = visualize.DynamicVisualizer()
    bonferroni_ntest = n_states # n_test = n_states
    for band in freq_bands:
        print(f"[{band.upper()}] Measuring age effects in power maps ...")

        # Subtract the mean across states
        pm = power_maps[band] - np.average(power_maps[band], axis=1, weights=gfo, keepdims=True)
        # dim: (n_subjects, n_states, n_channels)

        for n in range(n_states):
            print(f"\tState {n + 1}")
            # Fit GLM on subject-level power maps
            power_model, power_design, power_data = fit_glm(
                pm[:, n, :],
                subject_ids,
                group_assignments,
                modality=modality,
                dimension_labels=["Subjects", "Parcels"],
                plot_verbose=False,
            )
            tstats = np.squeeze(power_model.tstats[0])

            # Perform max-t permutation tests
            pval = max_stat_perm_test(
                power_model,
                power_data,
                power_design,
                pooled_dims=1,
                contrast_idx=0,
                n_perm=10000,
                metric="tstats",
            )
            pval *= bonferroni_ntest

            # Plot dynamic power maps
            thr_idx = pval < 0.05 # indices of significant parcels
            if np.any(thr_idx):
                print("\tSignificant parcels identified under Bonferroni-corrected p=0.05.\n" +
                      "\tPlotting Results ...")
                tstats_sig = np.zeros((tstats.shape))
                tstats_sig[thr_idx] = tstats[thr_idx]
                print("\tSelected parcels: ", np.arange(len(tstats))[thr_idx])
                DV.plot_power_map(
                    tstats_sig,
                    filename=os.path.join(
                        DATA_DIR, "maps",
                        f"maxt_pow_map_{band}_{n}.png"
                    ),
                    subtract_mean=False,
                    mean_weights=None,
                    colormap="RdBu_r",
                    fontsize=26,
                    plot_kwargs={"symmetric_cbar": True},
                )
            else:
                print("\tNo significant parcels identified under Bonferroni-corrected p=0.05.")

    print("Analysis complete.")
