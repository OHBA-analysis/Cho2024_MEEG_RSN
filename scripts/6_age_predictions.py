"""Age group prediction using M/EEG RSN features

"""

# Set up dependencies
import os
import copy
import numpy as np
from sys import argv
from sklearn.linear_model import LogisticRegression
from osl_dynamics.analysis import static
from osl_dynamics.data import Data
from osl_dynamics.inference import modes
from osl_dynamics.models.hmm import Model
from utils.data import (load_data,
                        save_data,
                        load_order,
                        load_age_information,
                        get_raw_file_names)
from utils.dynamic import compute_summary_statistics
from utils.statistics import repeated_multi_class_prediction


if __name__ == "__main__":
    # ------- [1] ------- #
    #      Settings       #
    # ------------------- #
    print("*** STEP 1: SETTINGS ***")

    # Set hyperparameters
    if len(argv) != 6:
        print("Need to pass five arguments: data modality, number of states, run ID, data type, and structural type " 
              + "(e.g., python script.py eeg 6 0 full subject)")
        exit()
    modality = argv[1] # data modality
    n_states = int(argv[2]) # number of states
    run_id = int(argv[3])
    data_type = argv[4]
    structurals = argv[5]
    if modality not in ["eeg", "meg"]:
        raise ValueError("modality should be either 'eeg' or 'meg'.")
    if data_type not in ["full"]:
        raise ValueError("this script only support a full dataset.")
    print(f"[INFO] Data Modality: {modality.upper()} | State #: {n_states} | Run ID: run{run_id} " + 
          f"| Data Type: {data_type} | Structural Type: {structurals}")

    # Define dataset name
    if modality == "eeg":
        data_name = "lemon"
    else: data_name = "camcan"

    # Set directory paths
    BASE_DIR = "/well/woolrich/users/olt015/Cho2024_MEEG_RSN"
    PROJECT_DIR = "/well/woolrich/projects"
    if modality == "eeg":
        DATA_DIR = os.path.join(PROJECT_DIR, "lemon/scho23")
    if modality == "meg":
        DATA_DIR = os.path.join(PROJECT_DIR, "camcan/scho23")
    MODEL_DIR = BASE_DIR + f"/results/dynamic/{data_name}/state{n_states}/run{run_id}"
    if structurals == "standard":
        MODEL_DIR = MODEL_DIR.replace("dynamic", "dynamic_no_struct")
    SAVE_DIR = BASE_DIR + f"/results/static/{modality}"
    TMP_DIR = SAVE_DIR + "/tmp"
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(TMP_DIR, exist_ok=True)

    # Load subject information
    print("(Step 1-1) Loading subject information ...")
    age_group_idx = load_data(os.path.join(BASE_DIR, "data/age_group_idx.pkl"))
    subject_ids_young = age_group_idx[modality]["subject_ids_young"]
    subject_ids_old = age_group_idx[modality]["subject_ids_old"]
    subject_ids = np.concatenate((subject_ids_young, subject_ids_old))
    print("Total # of subjects: {} (Young: {}, Old: {})".format(
        len(subject_ids),
        len(subject_ids_young),
        len(subject_ids_old),
    ))

    # Load raw data
    print("(Step 1-2) Loading subject data ...")
    file_names = get_raw_file_names(DATA_DIR, subject_ids, modality, structurals)
    if modality == "eeg":
        training_data = Data(file_names, store_dir=TMP_DIR)
    if modality == "meg":
        training_data = Data(file_names, picks="misc", reject_by_annotation="omit", store_dir=TMP_DIR)

    # Get subject ages
    ages = load_age_information(subject_ids, modality, data_type="binary")
    # dim: (n_subjects,)

    # Load model data
    print("(Step 1-3) Loading model data ...")
    model_data = load_data(os.path.join(MODEL_DIR, f"model/results/{data_name}_hmm.pkl"))
    alpha = model_data["alpha"]
    if len(alpha) != len(subject_ids):
        raise ValueError("the length of alphas does not match the number of subjects.")
    
    # Get state orders for the specified model run
    order = load_order(modality, n_states, data_type, run_id, structurals)

    # Reorder states if necessary
    if order is not None:
        print(f"Reordering HMM states ...")
        print(f"\tOrder: {order}")
        alpha = [a[:, order] for a in alpha] # dim: (n_subjects, n_samples, n_states)

    # --------------------- [2] -------------------- #
    #      Static Network Feature Computations       #
    # ---------------------------------------------- #
    print("\n*** STEP 2: STATIC NETWORK FEATURE COMPUTATIONS ***")

    save_path = os.path.join(
        BASE_DIR, "data",
        f"static_tde_cov_{modality}_{n_states}states_run{run_id}.pkl",
    )
    if structurals == "standard":
        save_path = save_path.replace(".pkl", "_no_struct.pkl")

    if os.path.exists(save_path):
        # Load static TDE covariances
        print("(Step 2-1) Loading static TDE covariances ...")
        static_tde_cov = load_data(save_path)["cov"]
    else:
        # Prepare the training dataset
        print("(Step 2-1) Preparing training dataset ...")
        static_data = copy.deepcopy(training_data)
        prepare_config = {
            "tde_pca": {"n_embeddings": 15, "n_pca_components": 80, "use_raw": True},
            "standardize": {},
        }
        static_data.prepare(methods=prepare_config)

        # Calculate static TDE covariances
        print("(Step 2-2) Calculating static TDE covariances ...")
        static_tde_cov = static.functional_connectivity(
            static_data.time_series(),
            conn_type="cov",
        ) # dim: (n_subjects, n_components, n_components)
        output = {"cov": static_tde_cov}
        save_data(output, save_path)

    # --------------------- [3] --------------------- #
    #      Dynamic Network Feature Computations       #
    # ----------------------------------------------- #
    print("\n*** STEP 3: DYNAMIC NETWORK FEATURE COMPUTATIONS ***")

    save_path = os.path.join(
        BASE_DIR, "data",
        f"dual_est_{modality}_{n_states}states_run{run_id}.pkl",
    )
    if structurals == "standard":
        save_path = save_path.replace(".pkl", "_no_struct.pkl")

    if os.path.exists(save_path):
        # Load state-specific TDE covariances
        print("(Step 3-1) Loading dynamic TDE covariances ...")
        dual_est_results = load_data(save_path)
        _, dynamic_tde_cov = dual_est_results.values()
    else:
        # Prepare the training dataset
        print("(Step 3-1) Preparing training dataset ...")
        dynamic_data = copy.deepcopy(training_data)
        prepare_config = {
            "tde_pca": {"n_embeddings": 15, "n_pca_components": 80, "use_raw": True},
            "standardize": {},
        }
        dynamic_data.prepare(methods=prepare_config)

        # Load HMM model
        print("(Step 3-2) Loading HMM model ...")
        model = Model.load(os.path.join(MODEL_DIR, "model/trained_model"))
        model.summary()

        # Calculate state-specific TDE covariances (by dual estimation)
        print("(Step 3-3) Calculating dynamic TDE covariances ...")
        dynamic_mean, dynamic_tde_cov = model.dual_estimation(dynamic_data, alpha)
        # dim (mean): (n_subjects, n_states, n_components)
        # dim (cov): (n_subjects, n_states, n_components, n_components)
        output = {
            "mean": dynamic_mean,
            "cov": dynamic_tde_cov,
        }
        save_data(output, save_path)

    # Calculate summary statistics
    print("(Step 3-4) Calculating summary statistics ...")
    Fs = 250 # sampling frequency (Hz)
    stc = modes.argmax_time_courses(alpha) # get HMM state time courses
    fo, lt, intv, sr = compute_summary_statistics(stc, Fs)
    # dim: (n_subjects, n_states)

    training_data.delete_dir() # clean up

    # ----------------- [4] ---------------- #
    #      Network Feature Combination       #
    # -------------------------------------- #
    print("\n*** STEP 4: NETWORK FEATURE COMBINATION ***")

    # Get static network features
    m, n = np.triu_indices(static_tde_cov.shape[-1])
    X_static = static_tde_cov[..., m, n]

    # Get dynamic network features
    expand_dims = lambda x: np.expand_dims(x, axis=-1)
    m, n = np.triu_indices(dynamic_tde_cov.shape[-1])
    X_dynamic = np.concatenate([
        expand_dims(fo),
        expand_dims(lt),
        expand_dims(intv),
        expand_dims(sr),
        dynamic_tde_cov[..., m, n]
    ], axis=-1)
    X_dynamic = X_dynamic.reshape(X_dynamic.shape[0], -1)

    # Combine static and dynamic network features
    X_all = np.concatenate([X_static, X_dynamic], axis=-1)

    # Define a target vector
    y_target = ages.copy()

    # Summarize input and output feature dimensions
    for nm, x in zip(["static", "dynamic", "all"], [X_static, X_dynamic, X_all]):
        print(f"Shape of {nm} input features: {x.shape}")
    print(f"Shape of target vector: {y_target.shape}")

    # -------------------- [5] ------------------ #
    #      MULTI-CLASS AGE GROUP PREDICTION       #
    # ------------------------------------------- #
    print("\n*** STEP 5: MULTI-CLASS AGE GROUP PREDICTION ***")

    print("(Step 5-1) Predicting with static network features ...")
    static_test_scores = repeated_multi_class_prediction(
        X_static,
        y_target,
        classifier=LogisticRegression(),
        n_splits=5,
        repeats=10,
    )

    print("(Step 5-2) Predicting with dynamic network features ...")
    dynamic_test_scores = repeated_multi_class_prediction(
        X_dynamic,
        y_target,
        classifier=LogisticRegression(),
        n_splits=5,
        repeats=10,
    )

    print("(Step 5-3) Predicting with combined network features")
    combined_test_scores = repeated_multi_class_prediction(
        X_all,
        y_target,
        classifier=LogisticRegression(),
        n_splits=5,
        repeats=10,
    )

    # Save outputs
    save_path = os.path.join(
        BASE_DIR, "data",
        f"pred_acc_{modality}_{n_states}states_run{run_id}.pkl",
    )
    if structurals == "standard":
        save_path = save_path.replace(".pkl", "_no_struct.pkl")
    
    output = {
        "static": static_test_scores,
        "dynamic": dynamic_test_scores,
        "combined": combined_test_scores,
    }
    save_data(output, save_path)

    print("Analysis complete.")
