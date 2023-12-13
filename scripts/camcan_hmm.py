"""Run HMM on the Cam-CAN MEG dataset (age-matched)

"""

# Set up dependencies
import os
import pickle
import numpy as np
from sys import argv
from osl_dynamics import data
from osl_dynamics.inference import tf_ops
from osl_dynamics.models.hmm import Config, Model


if __name__ == "__main__":
    # ------- [1] ------- #
    #      Settings       #
    # ------------------- #
    print("Step 1 - Setting up ...")

    # Set run name
    if len(argv) != 3:
        raise ValueError("Need to pass one argument: run ID & number of states (e.g., python script.py 1 6)")
    run = argv[1] # run ID
    n_states = int(argv[2]) # number of states

    # Set up GPU
    tf_ops.gpu_growth()

    # Set output direcotry path
    BASE_DIR = "/well/woolrich/users/olt015/Cho2023_EEG_RSN"
    output_dir = f"{BASE_DIR}/results/dynamic/camcan/state{n_states}/run{run}"
    os.makedirs(output_dir, exist_ok=True)

    # Set output sub-directory paths
    analysis_dir = f"{output_dir}/analysis"
    model_dir = f"{output_dir}/model"
    maps_dir = f"{output_dir}/maps"
    tmp_dir = f"{output_dir}/tmp"
    save_dir = f"{model_dir}/results"
    os.makedirs(analysis_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(maps_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    # Define training hyperparameters
    config = Config(
        n_states=n_states,
        n_channels=80,
        sequence_length=800,
        learn_means=False,
        learn_covariances=True,
        learn_trans_prob=True,
        batch_size=64,
        learning_rate=1e-3,
        n_epochs=20,
    )

    # --------------- [2] --------------- #
    #      Prepare training dataset       #
    # ----------------------------------- #
    print("Step 2 - Preparing training dataset ...")

    # Load data
    dataset_dir = "/well/woolrich/projects/camcan/scho23/src"
    with open(os.path.join(BASE_DIR, "data/age_group_idx.pkl"), "rb") as input_path:
        age_group_idx = pickle.load(input_path)
    input_path.close()
    subject_ids = np.concatenate((
        age_group_idx["meg"]["subject_ids_young"],
        age_group_idx["meg"]["subject_ids_old"],
    )) # subjects age-matched with EEG LEMON
    file_names = [os.path.join(dataset_dir, f"{id}/sflip_parc-raw.fif") for id in subject_ids]
    print(f"Total number of subjects available: {len(file_names)}")

    # Prepare the data for training
    training_data = data.Data(file_names, picks=["misc"], reject_by_annotation="omit", store_dir=tmp_dir)
    prepare_config = {
        "tde_pca": {"n_embeddings": 15, "n_pca_components": config.n_channels},
        "standardize": {},
    }
    training_data.prepare(methods=prepare_config)

    # ------------ [3] ------------- #
    #      Build the HMM model       #
    # ------------------------------ #
    print("Step 3 - Building model ...")
    model = Model(config)
    model.summary()

    # ------------ [4] ------------- #
    #      Train the HMM model       #
    # ------------------------------ #
    print("Step 4 - Training the model ...")

    # Initialization
    print("Initializing means/covariances based on a random state time course...")
    model.random_state_time_course_initialization(training_data, n_epochs=2, n_init=5)

    # Train the model on a full dataset
    history = model.fit(training_data)

    # Save the trained model
    model.save(f"{model_dir}/trained_model")

    # Save training history
    with open(f"{model_dir}/history.pkl", "wb") as file:
        pickle.dump(history, file)

    # -------- [5] ---------- #
    #      Save results       #
    # ----------------------- #
    print("Step 5 - Saving results ...")

    # Get results
    loss = history["loss"] # training loss
    free_energy = model.free_energy(training_data) # free energy
    alpha = model.get_alpha(training_data) # inferred state probabilities (equivalent to HMM gamma)
    tp = model.get_trans_prob() # inferred transition probability matrices
    cov = model.get_covariances() # inferred covariances
    ts = model.get_training_time_series(training_data, prepared=False) # subject-specific training data
    
    print("Final loss: ", loss[-1])
    print("Free energy: ", free_energy)
    
    # Save results
    outputs = {
        "loss": loss,
        "free_energy": free_energy,
        "alpha": alpha,
        "transition_probability": tp,
        "covariance": cov,
        "training_time_series": ts,
    }

    with open(save_dir + "/camcan_hmm.pkl", "wb") as output_path:
        pickle.dump(outputs, output_path)
    output_path.close()

    np.save(save_dir + "/free_energy.npy", free_energy)

    # ------- [6] ------- #
    #      Clean up       #
    # ------------------- #
    training_data.delete_dir()

    print("Model training complete.")