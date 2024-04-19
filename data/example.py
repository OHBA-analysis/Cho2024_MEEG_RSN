"""Example for loading the optimal HMM runs and calculating model inferences

"""

# Set up dependencies
import os
import pickle
import numpy as np
from sys import argv
from osl_dynamics import data
from osl_dynamics.inference import tf_ops
from osl_dynamics.models.hmm import Model


if __name__ == "__main__":
    # ------- [1] ------- #
    #      Settings       #
    # ------------------- #
    print("Step 1 - Setting up ...")

    # Set run name
    if len(argv) != 2:
        raise ValueError("Need to pass one argument: modality (e.g., python script.py eeg)")
    modality = argv[1] # modality type
    print(f"[INFO] Modality: {modality.upper()}")

    # Set up GPU
    tf_ops.gpu_growth()

    # Set directory and model paths
    DATA_DIR = "/well/woolrich/users/olt015/Cho2024_MEEG_RSN/data"
    model_path = os.path.join(DATA_DIR, f"trained_model_{modality}")

    # --------------- [2] --------------- #
    #      Prepare training dataset       #
    # ----------------------------------- #
    print("Step 2 - Preparing training dataset ...")

    # Load subject IDs
    with open(os.path.join(DATA_DIR, "age_group_idx.pkl"), "rb") as input_path:
        age_group_idx = pickle.load(input_path)
    subject_ids  = np.concatenate((
        age_group_idx[modality]["subject_ids_young"],
        age_group_idx[modality]["subject_ids_old"],
    ))

    # Load data
    if modality == "eeg":
        dataset_dir = "/well/woolrich/projects/lemon/scho23/src_ec"
        file_names = [os.path.join(dataset_dir,  f"{id}/sflip_parc-raw.npy") for id in subject_ids]
        training_data = data.Data(file_names)
    elif modality == "meg":
        dataset_dir = "/well/woolrich/projects/camcan/scho23/src"
        file_names = [os.path.join(dataset_dir, f"{id}/sflip_parc-raw.fif") for id in subject_ids]
        training_data = data.Data(file_names, picks=["misc"], reject_by_annotation="omit")

    # Prepare the data for training
    prepare_config = {
        "tde_pca": {"n_embeddings": 15, "n_pca_components": 80},
        "standardize": {},
    }
    training_data.prepare(methods=prepare_config)

    # ------------ [3] ------------- #
    #      Build the HMM model       #
    # ------------------------------ #
    print("Step 3 - Building model ...")

    # Load pre-trained model weights
    print("Loading pre-trained model weights ...")
    model = Model.load(model_path)
    model.summary()

    # --------------- [4] ----------------- #
    #      Calculate post-hoc metrics       #
    # ------------------------------------- #
    print("Step 4 - Calculating model inferences ...")

    # NOTE: Users may use the loaded model as a pre-trained HMM and apply their own training data to infer
    #       post-hoc metrics without training the model again. See below for an example.

    # Infer model parameters
    free_energy = model.free_energy(training_data) # free energy
    alpha = model.get_alpha(training_data) # inferred state probabilities (equivalent to HMM gamma)
    tp = model.get_trans_prob() # inferred transition probability matrices
    cov = model.get_covariances() # inferred covariances
    ts = model.get_training_time_series(training_data, prepared=False) # subject-specific training data
    # NOTE: Transition probabilities and state covariances should remain identical as no training is done 
    #       for this model.

    # ------- [5] ------- #
    #      Clean up       #
    # ------------------- #
    training_data.delete_dir()

    print("Analysis complete.")
