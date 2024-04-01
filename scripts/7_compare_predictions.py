"""Compare predictive accuracies between runs with and without subject sMRIs

"""

# Set up dependencies
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sys import argv
from utils.data import load_data
from utils.statistics import stat_ind_two_samples


if __name__ == "__main__":
    # ------- [1] ------- #
    #      Settings       #
    # ------------------- #
    print("*** STEP 1: SETTINGS ***")

    # Set hyperparameters
    if len(argv) != 2:
        print("Need to pass one argument: data modality (e.g., python script.py eeg)")
        exit()
    modality = argv[1] # data modality
    if modality not in ["eeg", "meg"]:
        raise ValueError("modality should be either 'eeg' or 'meg'.")
    print(f"[INFO] Data Modality: {modality.upper()}")

    # Specify number of states and run IDs
    n_states = 6
    run_ids = {
        "eeg": {"subject": 48, "standard": 7},
        "meg": {"subject": 28, "standard": 4},
    }

    # Set directory paths
    BASE_DIR = "/well/woolrich/users/olt015/Cho2023_EEG_RSN"
    DATA_DIR = os.path.join(BASE_DIR, "data")
    SAVE_DIR = os.path.join(BASE_DIR, "results")

    # Load predictive accuracies
    mod_run_ids = run_ids[modality]
    data_path = os.path.join(DATA_DIR, f"pred_acc_{modality}_{n_states}states_run{{}}.pkl")
    data_ys = load_data(data_path.format(mod_run_ids["subject"])) # with structurals
    data_ns = load_data(data_path.format(f"{mod_run_ids['standard']}_no_struct")) # without strurcturals

    # Define input network features
    features_list = ["static", "dynamic", "combined"]
    n_features = len(features_list)

    # ------------ [2] ---------- #
    #      STATISTICAL TEST       #
    # --------------------------- #
    print("\n*** STEP 2: RUN STATISTICAL TESTS ***")

    # Perform statistical test comparing two independent samples
    for name in features_list:
        stat_ind_two_samples(data_ys[name], data_ns[name], bonferroni_ntest=n_features)

    # ----------- [3] -------- #
    #      VISUALISATION       #
    # ------------------------ #
    print("\n*** STEP 3: PREDICTIVE ACCURACY VISUALISATION ***")

    # Reorganise test score data into a Pandas dataframe format
    dataset = {"structurals": data_ys, "no_structurals": data_ns}
    rows = []
    for struct_type, features in dataset.items():
        for feature_name, values in features.items():
            for value in values:
                rows.append({"value": value, "feature": feature_name, "type": struct_type})
    df = pd.DataFrame.from_dict(data=rows)

    # Set visualisation parameters
    colors = ["#CBCE91", "#d3687f"]

    # Plot grouped bar plots
    fig, ax = plt.subplots(nrows=1, ncols=1)
    sns.barplot(
        data=df,
        x="feature", y="value", hue="type",
        width=0.8, gap=0.1,
        errorbar="sd", capsize=0.2,
        err_kws={"linewidth": 2},
        palette=sns.color_palette(colors),
        ax=ax, legend=False,
    )

    # Adjust axes settings
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Static", "Dynamic", "Combined"])
    ax.set_xlabel("Features", fontsize=12)
    ax.set_ylabel(f"Predictive Accuracy (n={len(data_ys['static'])})", fontsize=12)
    ax.set_ylim([0, 1])
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_linewidth(1.5)
    ax.tick_params(width=1.5, labelsize=12)

    # Save figures
    fig.savefig(os.path.join(SAVE_DIR, f"clf_performance_{modality}.png"))

    print("Analysis complete.")
