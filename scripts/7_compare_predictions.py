"""Compare predictive accuracies between runs with and without subject sMRIs

"""

# Set up dependencies
import os
import pandas as pd
from utils.data import load_data
from utils.statistics import stat_ind_two_samples, stat_ind_one_samples
from utils.visualize import plot_grouped_bars


if __name__ == "__main__":
    # ------- [1] ------- #
    #      Settings       #
    # ------------------- #
    print("*** STEP 1: SETTINGS ***")

    # Set directory paths
    BASE_DIR = "/well/woolrich/users/olt015/Cho2023_EEG_RSN"
    DATA_DIR = os.path.join(BASE_DIR, "data")
    SAVE_DIR = os.path.join(BASE_DIR, "results")

    # Specify number of states and run IDs
    n_states = 6
    run_ids = {
        "eeg": {"subject": 48, "standard": 7},
        "meg": {"subject": 28, "standard": 4},
    }

    # Load predictive accuracies
    data_ys = dict() # with structurals
    data_ns = dict() # without structurals
    for modality in ["eeg", "meg"]:
        mod_run_ids = run_ids[modality]
        data_path = os.path.join(DATA_DIR, f"pred_acc_{modality}_{n_states}states_run{{}}.pkl")
        data_ys[modality] = load_data(data_path.format(mod_run_ids["subject"])) # with structurals
        data_ns[modality] = load_data(data_path.format(f"{mod_run_ids['standard']}_no_struct")) # without strurcturals

    # Define input network features
    features_list = ["static", "dynamic", "combined"]
    n_features = len(features_list)

    # ------------ [2] ---------- #
    #      STATISTICAL TEST       #
    # --------------------------- #
    print("\n*** STEP 2: RUN STATISTICAL TESTS ***")

    # Perform statistical tests comparing structurals vs. no structurals
    print(f"\n(STEP 2-1) Comparing structurals vs. no structurals ...")
    for modality in ["eeg", "meg"]:
        for name in features_list:
            print(f"[INFO] Modality: {modality.upper()} | Feature: {name}")
            stat_ind_two_samples(
                data_ys[modality][name],
                data_ns[modality][name],
                bonferroni_ntest=n_features,
            )

    # Perform statistical tests comparing EEG vs. MEG (only for structurals)
    print(f"\n(STEP 2-2) Comparing EEG vs. MEG (only for structurals) ...")
    for name in features_list:
        print(f"[INFO] Feature: {name}")
        stat_ind_two_samples(
            data_ys["eeg"][name],
            data_ys["meg"][name],
            bonferroni_ntest=n_features,
        )
    
    # Perform statistical tests against the random chance (only for structurals)
    print(f"\n(STEP 2-3) Testing against the random chance ...")
    for modality in ["eeg", "meg"]:
        for name in features_list:
            print(f"[INFO] Modality: {modality.upper()} | Feature: {name}")
            stat_ind_one_samples(
                data_ys[modality][name],
                popmean=0.5,
                bonferroni_ntest=n_features,
            )

    # ----------- [3] -------- #
    #      VISUALISATION       #
    # ------------------------ #
    print("\n*** STEP 3: PREDICTIVE ACCURACY VISUALISATION ***")

    # Set visualisation parameters
    palette1 = ["#696969", "#FFAE42"] # grey vs. orange
    palette2 = ["#5F4B8B", "#E69A8D"] # purple vs. pink

    # Plot bar graphs comparing structurals vs. no structurals
    for modality in ["eeg", "meg"]:
        dataset = {"structurals": data_ys[modality], "no_structurals": data_ns[modality]}
        rows = []
        for struct_type, features in dataset.items():
            for feature_name, values in features.items():
                for value in values:
                    rows.append({"value": value, "feature": feature_name, "type": struct_type})
        df = pd.DataFrame.from_dict(data=rows)
        plot_grouped_bars(
            df, colors=palette1,
            filename=os.path.join(SAVE_DIR, f"clf_performance_{modality}.png"),
        )

    # Plot bar graphs comparing EEG vs. MEG (only for structurals)
    dataset = {"eeg": data_ys["eeg"], "meg": data_ys["meg"]}
    rows = []
    for modality, features in dataset.items():
        for feature_name, values in features.items():
            for value in values:
                rows.append({"value": value, "feature": feature_name, "type": modality})
    df = pd.DataFrame.from_dict(data=rows)
    plot_grouped_bars(
        df, colors=palette2, yline=0.5,
        filename=os.path.join(SAVE_DIR, f"clf_performance_modality.png"),
    )

    print("Analysis complete.")
