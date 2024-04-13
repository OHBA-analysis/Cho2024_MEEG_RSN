"""Confirm reproducibility of the RSN inferences within and across modalities

"""

# Set up dependencies
import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.array_ops import reorder_matrix_by_indices
from utils.data import load_data, load_order
from utils.dynamic import between_state_rv_coefs, js_distance_matrix
from utils.statistics import stat_ind_two_samples


if __name__ == "__main__":
    # ------- [1] ------- #
    #      Settings       #
    # ------------------- #
    print("*** STEP 1: SETTINGS ***")

    # Set directory paths
    BASE_DIR = "/well/woolrich/users/olt015/Cho2024_MEEG_RSN"
    DATA_DIR = BASE_DIR + "/results/reprod"

    # Set hyperparameters for the data to unpack
    n_states = 6
    structurals = "subject"
    data_info = [
        ("eeg", "split1", 1), ("eeg", "split2", 3), ("meg", "split1", 7), ("meg", "split2", 8)
    ] # contains (modality, data_type, best_run_id)

    # Preallocate data dictionaries
    covariance = {
        "eeg": {"split1": {}, "split2": {}},
        "meg": {"split1": {}, "split2": {}},
    }
    transition_probability = {
        "eeg": {"split1": {}, "split2": {}},
        "meg": {"split1": {}, "split2": {}},
    }

    # Load covariance and transition probability matrices
    print("(Step 1-1) Loading data ...")
    for md, dt, _ in data_info:
        print(f"\t[INFO] Modality: {md.upper()} | Data Type: {dt}")
        # Set data name
        if md == "eeg": dn = "lemon"
        else: dn = "camcan"
        # Load data from all 10 runs
        covs, tps = [], []
        for id in range(0, 10):
            print(f"\tExtracting Run #{id} ...")
            data = load_data(os.path.join(DATA_DIR, f"{dt}/{dn}/state{n_states}/run{id}/model/results/{dn}_hmm.pkl"))
            cov = data["covariance"] # dim: (n_states, n_channels, n_channels)
            tp = data["transition_probability"] # dim: (n_states, n_states)
            # Reorder states if necessary
            order = load_order(
                modality=("eeg" if dn == "lemon" else "meg"),
                n_states=n_states,
                data_type=dt,
                run_id=id,
                structurals=structurals,
            )
            if order is not None:
                cov = cov[order, :]
                tp = reorder_matrix_by_indices(tp, order)
            covs.append(cov)
            tps.append(tp)
        covariance[md][dt] = np.array(covs)
        transition_probability[md][dt] = np.array(tps)

    # -------------- [2] --------------- #
    #      Measure reproducibility       #
    # ---------------------------------- #
    print("\n*** STEP 2: MEASURE REPRODUCIBILITY ***")

    # Measure reproducibility within and across modalities with state covariances
    cov_pairs = list(itertools.combinations(data_info, 2))
    rv_coefs = []
    for pair1, pair2 in cov_pairs:
        md1, dt1, id1 = pair1
        md2, dt2, id2 = pair2
        print(f"Comparing {md1.upper()} {dt1} Run #{id1} vs. {md2.upper()} {dt2} Run #{id2}")
        # Calculate RV coefficients of the inferred covariances
        rv_coefs.append(between_state_rv_coefs(
            covariance[md1][dt1][id1, :], covariance[md2][dt2][id2, :]
        ))

    # Measure reproducibility within and across modalities with state covariances
    tp_pairs = list(itertools.combinations([tpl[:-1] for tpl in data_info], 2))
    js_dist = []
    for pair1, pair2 in tp_pairs:
        md1, dt1 = pair1
        md2, dt2 = pair2
        # Calculate Jensen-Shannon (JS) distance between transition probability matrices
        js_pairs = [(x, y) for x in np.arange(0, 10) for y in np.arange(0, 10)]
        # NOTE: We create pairs with 10 runs in one split-half dataset and 10 runs in the 
        #       other split-half dataset.
        js_vals = []
        for i, j in js_pairs:
            js_vals.append(js_distance_matrix(
                transition_probability[md1][dt1][i], transition_probability[md2][dt2][j]
            ))
        js_dist.append(js_vals)

    # Reset pair orders for visualisation
    pair_orders = [0, 5, 1, 2, 3, 4]
    rv_coefs = np.array(rv_coefs)[pair_orders, :]
    js_dist = [js_dist[idx] for idx in pair_orders]

    # --------- [3] ---------- #
    #      Visualization       #
    # ------------------------ #
    print("\n*** STEP 3: VISUALIZATION ***")

    # Set visualisation parameters
    n_pairs = len(pair_orders)
    names = [
        "EEG 1st\nEEG 2nd",
        "MEG 1st\nMEG 2nd",
        "EEG 1st\nMEG 1st",
        "EEG 1st\nMEG 2nd",
        "EEG 2nd\nMEG 1st",
        "EEG 2nd\nMEG 2nd",
    ] # names of comparisons
    colors = ["#955196", "#dd5182", "#003f5c", "#444e86", "#ff6e54", "#ffa600"] # colors for each pair

    # Plot RV coefficients between covariance matrices for each state
    print("(Step 3-1) Summarizing state covariance RV coefficients ...")
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
    for p in range(n_pairs):
        ax.plot(np.arange(n_states) + 1, rv_coefs[p, :], color=colors[p], lw=1.5, alpha=0.5)
    for p in range(n_pairs):
        ax.scatter(
            np.arange(n_states) + 1,
            rv_coefs[p, :],
            s=70,
            c=colors[p],
            marker=("o" if p < 2 else "^"),
            label=names[p].replace("\n", " vs. "),
        )
    ax.spines[["right", "top"]].set_visible(False)
    ax.set(
        xlabel="States",
        ylabel="RV Coefficients",
        yticks=np.arange(0.55, 1, 0.1),
        ylim=[0.55, 1.00],
    )
    ax.legend(loc="upper center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, 1.18))
    fig.savefig(
        os.path.join(DATA_DIR, f"cov_rv_coefs_states{n_states}.png"),
        transparent=True, bbox_inches="tight"
    )

    # Plot JS distance between transition probability matrices
    print("(Step 3-2) Summarizing transition probability JS distance ...")
    df = pd.DataFrame(
        [(name, dist) for name, sublist in zip(names, js_dist) for dist in sublist],
        columns=["Comparisons", "JS Distance"],
    )

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 4))
    vp = sns.violinplot(
        df, x="JS Distance", y="Comparisons",
        hue="Comparisons", palette=colors,
        orient="h", ax=ax,
    )
    sns.despine(fig=fig, ax=ax)
    ax.axhline(y=1.5, color="k", linewidth=1.5, linestyle="--", alpha=0.4)
    ax.set_xlabel("JS Distances")
    ax.set_ylabel("")
    plt.tight_layout()
    fig.savefig(os.path.join(DATA_DIR, f"tp_js_dist_states{n_states}.png"), transparent=True)

    # Perform a statistical test comparing JS distances within modalities
    stat_ind_two_samples(js_dist[0], js_dist[1])

    print("Analysis completed.")
