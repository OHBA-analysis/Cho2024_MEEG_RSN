"""Select the best model run based on free energy

"""

# Set up dependencies
import os
import pickle
import numpy as np
from sys import argv
from utils.visualize import plot_loss_curve


if __name__ == "__main__":
    # Set hyperparameters
    if len(argv) != 6:
        print("Need to pass five arguments: data type, dataset name, number of states, "
               + "run IDs, and structural type (e.g., python script.py full lemon 6 0-9 subject)")
    data_type = argv[1] # type of dataset
    data_name = argv[2] # name of dataset
    n_states = int(argv[3]) # number of states
    run_ids = list(map(int, argv[4].split("-"))) # range of runs to compare
    structurals = argv[5] # type of structurals to use
    print(f"[INFO] Dataset Name: {data_name.upper()} | State #: {n_states} |"
          + f" Run: run{run_ids[0]} - run{run_ids[1]} | Structurals: {structurals}")
    
    # Set directories
    BASE_DIR = "/well/woolrich/users/olt015/Cho2024_MEEG_RSN/results/dynamic"
    if data_type != "full":
        BASE_DIR = BASE_DIR.replace("dynamic", f"reprod/{data_type}")
    if structurals == "standard":
        for subdir in ["dynamic", "reprod"]:
            BASE_DIR = BASE_DIR.replace(subdir, f"{subdir}_no_struct")
    DATA_DIR = BASE_DIR + f"/{data_name}/state{n_states}"
    data_path = DATA_DIR + f"/run{{0}}/model/results/free_energy.npy"

    # Group 10 model runs as one set
    if np.diff(run_ids) + 1 > 10:
        intervals = [[i, min(i + 9, run_ids[1])] for i in range(run_ids[0], run_ids[1] + 1, 10)]
    else: intervals = [run_ids]

    # Get the best model from each 10 model runs
    best_fes, best_runs = [], []
    for i, (start, end) in enumerate(intervals):
        print(f"Loading free energy (run{start}-run{end}) ...")
        free_energy = []
        run_id_list = np.arange(start, end + 1)
        for id in run_id_list:
            file_name = data_path.replace("{0}", str(id))
            free_energy.append(float(np.load(file_name)))
        best_fes.append(np.min(free_energy))
        best_runs.append(run_id_list[free_energy.index(best_fes[i])])
        print(f"\tFree energy (n={len(run_id_list)}): {free_energy}")
        print(f"\tBest run: run{best_runs[i]}")
        print(f"\tBest free energy: {best_fes[i]}")

    # Identify the optimal run from all the best runs
    opt_fe = np.min(best_fes)
    opt_run = best_runs[np.argmin(best_fes)]
    print(f"The lowest free energy is {opt_fe} from run{opt_run}.")

    # Load training loss values of the optimal run
    with open(os.path.join(DATA_DIR, f"run{opt_run}/model/results/{data_name}_hmm.pkl"), "rb") as input_path:
        data = pickle.load(input_path)
    loss = data["loss"]
    epochs = np.arange(1, len(loss) + 1)

    # Plot training loss curve
    plot_loss_curve(loss, save_dir=os.path.join(DATA_DIR, f"run{opt_run}/analysis"))
    
    print("Analysis complete.")
