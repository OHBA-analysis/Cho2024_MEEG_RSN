"""Preparing the LEMON EEG Data
[STEP 5] Match age distributions of LEMON and CamCAN datasets and visualize their summary
"""

# Set up dependencies
import os
import glob
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Define custom codes
def _list_to_array(data):
    if type(data) == list:
        array = np.array(data)
    return array

def plot_age_distributions(ages_young, ages_old, modality, nbins="auto", save_dir=""):
    """Plots an age distribution of each group as a histogram.

    Parameters
    ----------
    ages_young : list or np.ndarray
        Ages of young participants. Shape is (n_subjects,)
    ages_old : list or np.ndarray
        Ages of old participants. Shape is (n_subjects,)
    modality : str
        Type of imaging modality/dataset. Can be either "eeg" or "meg".
    nbins : str, int, or list
        Number of bins to use for each histograms. Different nbins can be given
        for each age group in a list form. Defaults to "auto". Can take options
        described in `numpy.histogram_bin_edges()`.
    save_dir : str
        Path to a directory in which the plot should be saved. By default, the 
        plot will be saved to a user's current directory.
    """

    # Validation
    if modality not in ["eeg", "meg"]:
        raise ValueError("modality should be either 'eeg' or 'meg'.")
    if not isinstance(nbins, list):
        nbins = [nbins, nbins]

    # Set visualization parameters
    cmap = sns.color_palette("deep")
    sns.set_style("white")
    if modality == "eeg":
        data_name = "LEMON"
    else: data_name = "CamCAN"

    # Sort ages for ordered x-tick labels
    ages_young, ages_old = sorted(ages_young), sorted(ages_old)

    # Plot histograms
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 4))
    sns.histplot(x=ages_young, ax=ax[0], color=cmap[0], bins=nbins[0])
    sns.histplot(x=ages_old, ax=ax[1], color=cmap[3], bins=nbins[1])
    ax[0].set_title(f"Young (n={len(ages_young)})")
    ax[1].set_title(f"Old (n={len(ages_old)})")
    for i in range(2):
        ax[i].set_xlabel("Age")
        ax[i].yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    plt.suptitle(f"{data_name} Age Distribution ({modality.upper()})")
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, f"age_dist_{modality}.png"))
    plt.close(fig)

    return None

def plot_sex_distributions(ages_young, ages_old, sexes_young, sexes_old, modality, save_dir=""):
    """Plots a sex distribution of each group as a histogram.

    Parameters
    ----------
    ages_young : list or np.ndarray
        Ages of young participants. Shape is (n_subjects,)
    ages_old : list or np.ndarray
        Ages of old participants. Shape is (n_subjects,)
    sexes_young : list or np.ndarray
        Sexes of young participants. Shape is (n_subjects,)
    sexes_old : list or np.ndarray
        Sexes of old participants. Shape is (n_subjects,)
    modality : str
        Type of imaging modality/dataset. Can be either "eeg" or "meg".
    save_dir : str
        Path to a directory in which the plot should be saved. By default, the 
        plot will be saved to a user's current directory.
    """

    # Validation
    if modality not in ["eeg", "meg"]:
        raise ValueError("modality should be either 'eeg' or 'meg'.")
    ages_young = _list_to_array(ages_young)
    ages_old = _list_to_array(ages_old)
    sexes_young = _list_to_array(sexes_young)
    sexes_old = _list_to_array(sexes_old)

    # Set visualization parameters
    sns.set_style("white")
    if modality == "eeg":
        data_name = "LEMON"
    else: data_name = "CamCAN"

    # Organize sex by age intervals for each group
    age_intervals_y = [[20, 25], [25, 30], [30, 35]]
    age_intervals_o = [[55, 60], [60, 65], [65, 70], [70, 75], [75, 80]]
    sex_by_age_y = np.zeros((2, len(age_intervals_y)))
    sex_by_age_o = np.zeros((2, len(age_intervals_o)))

    for n, (start, end) in enumerate(age_intervals_y):
        for i in [0, 1]:
            if modality == "eeg":
                sex_by_age_y[i, n] = list(sexes_young[ages_young == f"{start}-{end}"]).count(i + 1)
            else:
                if n == 0:
                    sex_by_age_y[i, n] = list(sexes_young[np.logical_and(ages_young >= start, ages_young <= end)]).count(i + 1)
                else: sex_by_age_y[i, n] = list(sexes_young[np.logical_and(ages_young > start, ages_young <= end)]).count(i + 1)
    for n, (start, end) in enumerate(age_intervals_o):
        for i in [0, 1]:
            if modality == "eeg":
                sex_by_age_o[i, n] = list(sexes_old[ages_old == f"{start}-{end}"]).count(i + 1)
            else:
                if n == 0:
                    sex_by_age_o[i, n] = list(sexes_old[np.logical_and(ages_old >= start, ages_old <= end)]).count(i + 1)
                else: sex_by_age_o[i, n] = list(sexes_old[np.logical_and(ages_old > start, ages_old <= end)]).count(i + 1)

    # Reformat to dictionary
    sex_by_age_y = {"Female": sex_by_age_y[0, :], "Male": sex_by_age_y[1, :]}
    sex_by_age_o = {"Female": sex_by_age_o[0, :], "Male": sex_by_age_o[1, :]}

    # Plot histogram
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 4))
    colors = ["tab:green", "tab:orange"]
    bottom = np.zeros(len(age_intervals_y))
    for n, (lbl, count) in enumerate(sex_by_age_y.items()):
        ax[0].bar([f"{intv[0]}-{intv[1]}" for intv in age_intervals_y], count, 0.7, label=lbl, bottom=bottom, color=colors[n], alpha=0.7)
        bottom += count
    bottom = np.zeros(len(age_intervals_o))
    for n, (lbl, count) in enumerate(sex_by_age_o.items()):
        ax[1].bar([f"{intv[0]}-{intv[1]}" for intv in age_intervals_o], count, 0.7, label=lbl, bottom=bottom, color=colors[n], alpha=0.7)
        bottom += count
    for i in range(2):
        ax[i].legend(loc="upper right")
        ax[i].set_xlabel("Age")
        ax[i].yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    plt.suptitle(f"{data_name} Sex Distribution ({modality.upper()})")
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, f"sex_dist_{modality}.png"))
    plt.close(fig)

    return None

# MATCH AGE DISTRIBUTIONS
if __name__ == "__main__":
    # Set directory paths
    PROJECT_DIR = "/well/woolrich/projects"
    eeg_data_dir = PROJECT_DIR + "/lemon/scho23"
    eeg_meta_dir = PROJECT_DIR + "/lemon/raw/Behavioural_Data_MPILMBB_LEMON/META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv"
    meg_data_dir = PROJECT_DIR + "/camcan/scho23"
    meg_meta_dir = PROJECT_DIR + "/camcan/participants.tsv"
    SAVE_DIR = os.path.join(meg_data_dir, "data")
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Set random seed number
    n_seed = 2023

    # Load meta data
    eeg_meta_data = pd.read_csv(eeg_meta_dir, sep=",")
    meg_meta_data = pd.read_csv(meg_meta_dir, sep="\t")

    # Get subject IDs
    eeg_file_names = sorted(glob.glob(eeg_data_dir + "/src_ec/*/sflip_parc-raw.npy"))
    eeg_subject_ids = np.array([file.split("/")[-2] for file in eeg_file_names])
    
    meg_exclude_ids = ["sub-CC220901", "sub-CC321087", "sub-CC510259"] # missing data
    meg_exclude_ids += [
        "sub-CC120137", "sub-CC120184", "sub-CC120212", "sub-CC122016", "sub-CC210174", "sub-CC220232", "sub-CC220511", "sub-CC220901", 
        "sub-CC221585", "sub-CC221737", "sub-CC312149", "sub-CC320361", "sub-CC320428", "sub-CC320445", "sub-CC320861", "sub-CC321087", 
        "sub-CC410182", "sub-CC410286", "sub-CC420244", "sub-CC420364", "sub-CC510256", "sub-CC510259", "sub-CC520197", "sub-CC520436", 
        "sub-CC610051", "sub-CC620527", "sub-CC621118", "sub-CC710223", "sub-CC710858", "sub-CC720330", "sub-CC720723", "sub-CC720774", 
        "sub-CC721449", "sub-CC723197"
    ] # missing structural files
    meg_exclude_ids += ["sub-CC510395"] # bad data quality
    meg_subject_ids = np.array(sorted([id for id in sorted(meg_meta_data["participant_id"]) if id not in set(meg_exclude_ids)]))

    print(f"[EEG] # Subjects: {len(eeg_subject_ids)}")
    print(f"[MEG] # Subjects: {len(meg_subject_ids)}")

    # Get subject age
    eeg_subject_ages = np.array([eeg_meta_data.loc[eeg_meta_data["ID"] == id]["Age"].values[0] for id in eeg_subject_ids])
    meg_subject_ages = np.array([meg_meta_data.loc[meg_meta_data["participant_id"] == id]["age"].values[0] for id in meg_subject_ids])

    # Get subject sex (female: 1, male: 2)
    eeg_subject_sexes = np.array([eeg_meta_data.loc[eeg_meta_data["ID"] == id]["Gender_ 1=female_2=male"].values[0] for id in eeg_subject_ids])
    meg_subject_sexes = np.array([
        1 if sex == "FEMALE" else 2
        for sex in [meg_meta_data.loc[meg_meta_data["participant_id"] == id]["sex"].values[0] for id in meg_subject_ids]
    ])

    # Organize subject data into dataframes
    eeg_subject_df = pd.DataFrame({
        "ID": eeg_subject_ids,
        "Age": eeg_subject_ages,
        "Sex": eeg_subject_sexes,
    })
    meg_subject_df = pd.DataFrame({
        "ID": meg_subject_ids,
        "Age": meg_subject_ages,
        "Sex": meg_subject_sexes,
    })

    # Match age distributions of young participants
    #   NOTE: For 20-25, MEG (n=14) < EEG (n=42).
    #         For 25-30, MEG (n=45) > EEG (n=39); for MEG, the lower bound was 25<.
    #         For 30-35, MEG (n=50) > EEG (n=7);  for MEG, the lower bound was 30<.
    #         We select 14 subjects from EEG, 39 subjects from MEG, and 7 subjects from MEG, respectively.

    # [1] Match EEG
    meg_bel_25_num = np.sum(np.logical_and(meg_subject_ages >= 20, meg_subject_ages <= 25))
    eeg_bel_25_df = eeg_subject_df.loc[eeg_subject_df["Age"] == "20-25"]
    eeg_bel_25_selected_ids = eeg_bel_25_df.sample(meg_bel_25_num, random_state=n_seed)["ID"].to_numpy()
    eeg_sample_ids_y = np.concatenate((
        eeg_bel_25_selected_ids,
        eeg_subject_ids[eeg_subject_ages == "25-30"],
        eeg_subject_ids[eeg_subject_ages == "30-35"],
    ))

    # [2] Match MEG
    meg_sample_ids_y = meg_subject_ids[np.logical_and(meg_subject_ages >= 20, meg_subject_ages <= 25)]
    age_intervals = [[25, 30], [30, 35]]
    for n, (start, end) in enumerate(age_intervals):
        mask = np.logical_and(meg_subject_ages > start, meg_subject_ages <= end)
        eeg_subsample_num = np.sum(eeg_subject_ages == f"{start}-{end}")
        meg_subsample_df = meg_subject_df.loc[mask]
        meg_sample_ids_y = np.concatenate((
            meg_sample_ids_y,
            meg_subsample_df.sample(eeg_subsample_num, random_state=n_seed)["ID"].to_numpy(),
        ))

    # [3] Validation
    if len(eeg_sample_ids_y) != len(meg_sample_ids_y):
        raise ValueError("Number of young participants in M/EEG subsamples should be identical.")

    # Match age distributions of old participants
    #   NOTE: Since MEG had much more old subjects than EEG, we will randomly subsample from the Cam-CAN subjects 
    #         based on the LEMON dataset at each age interval.
    #         For 55-60, MEG (n=48) < EEG (n=4).
    #         For 60-65, MEG (n=47) < EEG (n=10); for MEG, the lower bound was 60<.
    #         For 65-70, MEG (n=55) < EEG (n=8);  for MEG, the lower bound was 65<.
    #         For 70-75, MEG (n=45) < EEG (n=12); for MEG, the lower bound was 70<.
    #         For 75-80, MEG (n=58) < EEG (n=2);  for MEG, the lower bound was 75<.

    # [1] Match M/EEG
    eeg_sample_ids_o = []
    meg_sample_ids_o = []
    age_intervals = [[55, 60], [60, 65], [65, 70], [70, 75], [75, 80]]    
    for n, (start, end) in enumerate(age_intervals):
        if n == 0:
            mask = np.logical_and(meg_subject_ages >= start, meg_subject_ages <= end)
        else:
            mask = np.logical_and(meg_subject_ages > start, meg_subject_ages <= end)
        eeg_subsample_num = np.sum(eeg_subject_ages == f"{start}-{end}")
        eeg_sample_ids_o.append(list(eeg_subject_ids[eeg_subject_ages == f"{start}-{end}"]))
        meg_subsample_df = meg_subject_df.loc[mask]
        meg_sample_ids_o.append(meg_subsample_df.sample(eeg_subsample_num, random_state=n_seed)["ID"].to_list())
    eeg_sample_ids_o = np.concatenate(eeg_sample_ids_o)
    meg_sample_ids_o = np.concatenate(meg_sample_ids_o)

    # [2] Validation
    if len(eeg_sample_ids_o) != len(meg_sample_ids_o):
        raise ValueError("Number of old participants in M/EEG subsamples should be identical.")

    # Compute subject ages in subsamples
    eeg_sample_age_y = [eeg_subject_ages[list(eeg_subject_ids).index(id)] for id in eeg_sample_ids_y]
    eeg_sample_age_o = [eeg_subject_ages[list(eeg_subject_ids).index(id)] for id in eeg_sample_ids_o]
    meg_sample_age_y = [meg_subject_ages[list(meg_subject_ids).index(id)] for id in meg_sample_ids_y]
    meg_sample_age_o = [meg_subject_ages[list(meg_subject_ids).index(id)] for id in meg_sample_ids_o]

    # Compute subject sexes in subsamples
    eeg_sample_sex_y = [eeg_subject_sexes[list(eeg_subject_ids).index(id)] for id in eeg_sample_ids_y]
    eeg_sample_sex_o = [eeg_subject_sexes[list(eeg_subject_ids).index(id)] for id in eeg_sample_ids_o]
    meg_sample_sex_y = [meg_subject_sexes[list(meg_subject_ids).index(id)] for id in meg_sample_ids_y]
    meg_sample_sex_o = [meg_subject_sexes[list(meg_subject_ids).index(id)] for id in meg_sample_ids_o]

    # Visualize age distributions
    plot_age_distributions(
        eeg_sample_age_y,
        eeg_sample_age_o,
        modality="eeg",
        save_dir=SAVE_DIR,
    )
    plot_age_distributions(
        meg_sample_age_y,
        meg_sample_age_o,
        modality="meg",
        nbins=[[20, 25, 30, 35], [55, 60, 65, 70, 75, 80]],
        save_dir=SAVE_DIR,
    )

    # Visualize sex distributions
    plot_sex_distributions(
        eeg_sample_age_y,
        eeg_sample_age_o,
        eeg_sample_sex_y,
        eeg_sample_sex_o,
        modality="eeg",
        save_dir=SAVE_DIR,
    )
    plot_sex_distributions(
        meg_sample_age_y,
        meg_sample_age_o,
        meg_sample_sex_y,
        meg_sample_sex_o,
        modality="meg",
        save_dir=SAVE_DIR,
    )

    # Save outputs
    output = {
        "eeg": {
            "subject_ids_young": eeg_sample_ids_y,
            "subject_ids_old": eeg_sample_ids_o,
        },
        "meg": {
            "subject_ids_young": meg_sample_ids_y,
            "subject_ids_old": meg_sample_ids_o,
        },
    }
    with open(os.path.join(SAVE_DIR, "age_group_idx.pkl"), "wb") as save_path:
        pickle.dump(output, save_path)
    save_path.close()

    print("Visualization Complete.")
