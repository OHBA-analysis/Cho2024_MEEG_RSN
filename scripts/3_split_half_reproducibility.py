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


if __name__ == "__main__":
    # ------- [1] ------- #
    #      Settings       #
    # ------------------- #
    print("*** STEP 1: SETTINGS ***")

    # Set directory paths
    BASE_DIR = "/well/woolrich/users/olt015/Cho2023_EEG_RSN"
    DATA_DIR = BASE_DIR + "/results/reprod"

    # Set hyperparameters for the data to unpack
    n_states = 6
    data_types = ["split1", "split2"] * 2
    run_ids = [1, 3, 7, 8]
    data_names = ["lemon", "lemon", "camcan", "camcan"]

    # Load covariance and transition probability matrices
    print("(Step 1-1) Loading data ...")
    covs, tps = [], []
    for dt, id, dn in zip(data_types, run_ids, data_names):
        # Load data
        data = load_data(os.path.join(DATA_DIR, f"{dt}/{dn}/state{n_states}/run{id}/model/results/{dn}_hmm.pkl"))
        cov = data["covariance"] # dim: (n_states, n_channels, n_channels)
        tp = data["transition_probability"] # dim: (n_states, n_states)
        # Reorder states if necessary
        order = load_order(
            modality=("eeg" if dn == "lemon" else "meg"),
            n_states=n_states,
            data_type=dt,
            run_id=id,
        )
        if order is not None:
            cov = cov[order, :]
            tp = reorder_matrix_by_indices(tp, order)
        # Store outputs
        covs.append(cov)
        tps.append(tp)

    # -------------- [2] --------------- #
    #      Measure reproducibility       #
    # ---------------------------------- #
    print("\n*** STEP 2: MEASURE REPRODUCIBILITY ***")

    # Define dataset pairs to compare
    pairs = list(itertools.combinations(np.arange(len(covs)), 2))
    n_pairs = len(pairs)
    names = [
        "EEG 1st\nEEG 2nd",
        "EEG 1st\nMEG 1st",
        "EEG 1st\nMEG 2nd",
        "EEG 2nd\nMEG 1st",
        "EEG 2nd\nMEG 2nd",
        "MEG 1st\nMEG 2nd"
    ] # names of comparisons
    pair_orders = [0, 5, 1, 2, 3, 4]
    colors = ["#955196", "#dd5182", "#003f5c", "#444e86", "#ff6e54", "#ffa600"] # colors for each pair

    # Measure reproducibility within and across modalities
    rv_coefs, js_divs = [], []
    for i, j in pairs:
        print(f"Comparing {data_names[i]} {data_types[i]} vs. {data_names[j]} {data_types[j]} ...")
        # Calculate RV coefficients of the inferred covariances
        rv_coefs.append(between_state_rv_coefs(covs[i], covs[j]))
        print(f"\tRV coefficients: {rv_coefs[-1]}")
        # Calculate Jensen-Shannon (JS) distance between transition probability matrices
        js_divs.append(js_distance_matrix(tps[i], tps[j]))
        print(f"\tJS distance: {js_divs[-1]}")

    # --------- [3] ---------- #
    #      Visualization       #
    # ------------------------ #
    print("\n*** STEP 3: VISUALIZATION ***")

    # Plot RV coefficients between covariance matrices for each state
    print("(Step 3-1) Summarizing state covariance RV coefficients ...")
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
    rv_names = np.array(names)[pair_orders]
    rv_coefs = np.array(rv_coefs)[pair_orders, :]
    for p in range(n_pairs):
        ax.plot(np.arange(n_states) + 1, rv_coefs[p, :], color=colors[p], lw=1.5, alpha=0.5)
    for p in range(n_pairs):
        ax.scatter(
            np.arange(n_states) + 1,
            rv_coefs[p, :],
            s=70,
            c=colors[p],
            marker=("o" if p < 2 else "^"),
            label=rv_names[p].replace("\n", " vs. "),
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
    df = pd.DataFrame({"Comparisons": names, "JS Distance": js_divs})
    df = df.reindex(index=pair_orders)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 4))
    bp = sns.barplot(
        df, x="JS Distance", y="Comparisons",
        width=0.4,
        linewidth=2,
        orient="h",
        edgecolor=(0, 0, 0, 0),
        fill=True,
    )
    sns.despine(fig=fig, ax=ax)
    for i, patch in enumerate(bp.patches):
        patch.set_facecolor(colors[i])
    ax.axhline(y=1.5, color="k", linewidth=1.5, linestyle="--", alpha=0.4)
    ax.set_xticks([0.00, 0.02, 0.04, 0.06])
    ax.set_ylabel("")
    plt.tight_layout()
    fig.savefig(os.path.join(DATA_DIR, f"tp_js_divs_states{n_states}.png"), transparent=True)

    print("Analysis completed.")