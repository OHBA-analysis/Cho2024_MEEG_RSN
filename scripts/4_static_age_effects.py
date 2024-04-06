"""Qualitative group-level comparison of static M/EEG RSN features

"""

# Set up dependencies
import os
import numpy as np
from sys import argv
from utils.data import load_data
from utils.statistics import fit_glm, cluster_perm_test, max_stat_perm_test
from utils.visualize import (GroupDifferencePSD,
                             StaticVisualizer,
                             plot_null_distribution,
                             _colormap_null)


if __name__ == "__main__":
    # ------- [1] ------- #
    #      Settings       #
    # ------------------- #
    print("*** STEP 1: SETTINGS ***")

    # Set hyperparameters
    if len(argv) != 3:
        print("Need to pass two arguments: data modality and structural type (e.g., python script.py eeg subject)")
        exit()
    modality = argv[1] # data modality
    structurals = argv[2] # type of structurals to use
    if modality not in ["eeg", "meg"]:
        raise ValueError("modality should be either 'eeg' or 'meg'.")
    print(f"[INFO] Data Modality: {modality.upper()} | Structurals: {structurals}")

    # Set directory paths
    BASE_DIR = "/well/woolrich/users/olt015/Cho2023_EEG_RSN"
    DATA_DIR = os.path.join(BASE_DIR, "data")
    SAVE_DIR = os.path.join(BASE_DIR, f"results/static/{modality}")
    if structurals == "standard":
        SAVE_DIR = SAVE_DIR.replace("static/", "static_no_struct/")

    # Load subject information
    print("(Step 1-1) Loading subject information ...")
    age_group_idx = load_data(os.path.join(DATA_DIR, "age_group_idx.pkl"))
    subject_ids_young = age_group_idx[modality]["subject_ids_young"]
    subject_ids_old = age_group_idx[modality]["subject_ids_old"]
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

    # Load subject-level network features
    data_path = os.path.join(DATA_DIR, f"static_network_features_{modality}.pkl")
    if structurals == "standard":
        data_path = data_path.replace(".pkl", "_no_struct.pkl")
    static_network_features = load_data(data_path)
    freqs, psds, weights, power_maps, conn_maps = static_network_features.values()

    # Average PSDs over channels/parcels
    ppsds = np.mean(psds, axis=1)
    # dim: (n_subjects, n_parcels, n_freqs) -> (n_subjects, n_freqs)

    # ------------------- [2] ------------------ #
    #      Group-Level Comparisons of PSDs       #
    # ------------------------------------------ #
    print("\n*** STEP 2: QUANTIFYING AGE EFFECTS IN PSDS ***")
    print("(Step 2-1) Fitting GLM ...")

     # Fit GLM on subject-level PSDs
    psd_model, psd_design, psd_data = fit_glm(
        psds,
        subject_ids,
        group_assignments,
        modality,
        dimension_labels=["Subjects", "Channels", "Frequency"],
        plot_verbose=False,
    )

    # Get group-averaged PSDs
    gpsd_old = psd_model.betas[0]
    gpsd_young = psd_model.betas[1]
    # dim: (n_parcels, n_freqs)

    # Fit GLM on parcel-averaged PSDs
    psd_model, psd_design, psd_data = fit_glm(
        ppsds,
        subject_ids,
        group_assignments,
        modality,
        dimension_labels=["Subjects", "Frequency"],
        plot_verbose=True,
        save_path=os.path.join(SAVE_DIR, "design_matrix_psd.png"),
    )

    # Perform a cluster permutation test on parcel-averaged PSDs
    print("(Step 2-2) Running a cluster permutation test ...")
    _, clu = cluster_perm_test(
        psd_model,
        psd_data,
        psd_design,
        pooled_dims=(1,),
        contrast_idx=0,
        n_perm=5000,
        metric="tstats",
        bonferroni_ntest=1,
    )

    # Plot group-level PSD differences
    print("(Step 2-3) Plotting group-level PSD differences ...")
    PSD_DIFF = GroupDifferencePSD(
        freqs,
        gpsd_young,
        gpsd_old,
        data_space="source",
        modality=modality,
    )
    PSD_DIFF.prepare_data()
    PSD_DIFF.plot_psd_diff(
        clusters=clu,
        group_lbls=["Young", "Old"],
        save_dir=SAVE_DIR,
    )

    # ---------------------- [3] --------------------- #
    #      Group-Level Comparisons of Power Maps       #
    # ------------------------------------------------ #
    print("\n*** STEP 3: QUANTIFYING AGE EFFECTS IN POWER MAPS ***")

    freq_bands = ["delta", "theta", "alpha", "beta"]
    verbose = False # for additional plotting
    for band in freq_bands:
        print(f"Processing {band.upper()} band power map ...")

        # Fit GLM on subject-level power maps
        power_model, power_design, power_data = fit_glm(
            power_maps[band],
            subject_ids,
            group_assignments,
            modality=modality,
            dimension_labels=["Subjects", "Parcels"],
            plot_verbose=verbose,
        )

        # Get power map group difference
        gpower_diff = power_model.copes[0] # old - young; dim: (n_parcels,)
        print(f"Shape of group-level power map difference: {gpower_diff.shape}")

        # Plot power map group difference
        SV = StaticVisualizer()
        SV.plot_power_map(
            gpower_diff,
            filename=os.path.join(SAVE_DIR, f"power_map_diff_{band}.png"),
            fontsize=26,
            plot_kwargs={"symmetric_cbar": True},
        )

        # Perform a max-t permutation test
        bonferroni_ntest = 4 # n_test = n_freq_bands
        pval, perm = max_stat_perm_test(
            power_model,
            power_data,
            power_design,
            pooled_dims=1,
            contrast_idx=0,
            metric="tstats",
            return_perm=True,
        )
        pval *= bonferroni_ntest # apply Bonferroni correction
        null_dist = perm.nulls # dim: (n_perm,)

        # Get critical threshold value
        p_alpha = 100 - (0.05 / bonferroni_ntest) * 100
        thresh = perm.get_thresh(p_alpha)
        # NOTE: We use 0.05 as our alpha threshold.
        print(f"Metric threshold: {thresh:.3f}")

        # Plot null distribution and threshold
        if verbose:
            plot_null_distribution(
                null_dist,
                thresh,
                filename=os.path.join(SAVE_DIR, f"null_dist_power_{band}.png"),
            )

        # Get t-statistics map
        tmap = power_model.tstats[0] # t-statistics
        thr_idx = pval < 0.05
        if np.sum(thr_idx) > 0:
            tmap_thr = np.multiply(tmap, thr_idx)
            cmap = "RdBu_r"
        else:
            tmap_thr = np.ones((tmap.shape))
            cmap = _colormap_null("RdBu_r")
        print(f"Maximum metric: {np.max(np.abs(tmap))}") # absolute values used for two-tailed t-test
        
        # Plot statistically significant group differences
        SV.plot_power_map(
            tmap_thr,
            filename=os.path.join(SAVE_DIR, f"t_map_{band}.png"),
            fontsize=26,
            plot_kwargs={
                "symmetric_cbar": True,
                "cmap": cmap,
            },
        )

    print("Analysis complete.")
