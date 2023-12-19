"""Functions for handling and inspecting data

"""

import os
import pickle
import mne
import numpy as np
import pandas as pd
from scipy import spatial

def _get_device_fids(raw, include_origin=False):
    """Get fiducials in the M/EEG device space.
    
    Parameters
    ----------
    raw : mne.io.Raw
        An MNE Raw object.
    include_origin : bool
        Whether to include the origin in the transform. The 
        transform of [0, 0, 0] is a good representation of 
        the head center. Can be used as a head position variable.

    Returns
    -------
    device_fids : np.ndarray
        Fiducial points in the device space. Shape is (3, 3) if 
        `include_origin` is False; otherwise, shape is (4, 3).
    """

    # Put fiducials in the device space
    head_fids = mne.viz._3d._fiducial_coords(raw.info["dig"])
    if include_origin:
        head_fids = np.vstack(([0, 0, 0], head_fids))
    fid_space = raw.info["dig"][0]["coord_frame"]
    assert(fid_space == 4) # ensure we have FIFFV_COORD_HEAD coordinates

    # Get device to head transform and inverse
    dev2head = raw.info["dev_head_t"]
    head2dev = mne.transforms.invert_transform(dev2head)
    assert(np.logical_and(head2dev["from"] == 4, head2dev["to"] == 1))

    # Apply transformation to get fiducials in the device space
    device_fids = mne.transforms.apply_trans(head2dev, head_fids)

    return device_fids

def _get_headsize_from_fids(fids):
    """Get rough estimates of the subject head size.

    Parameters
    ----------
    fids : np.ndarray
        Fiducial points for the subject. Shape must 
        be (3, 3).

    Returns
    -------
    area : float
        Estimate of the subject head size.
    
    Reference: https://en.wikipedia.org/wiki/Heron%27s_formula
    """

    # Validation
    if fids.shape != (3, 3):
        raise ValueError("fids should be in a shape (3, 3).")

    # Compute head size with Heron's formula
    dists = spatial.distance.pdist(fids)
    semi_perimeter = np.sum(dists) / 2
    area = np.sqrt(semi_perimeter * np.prod(semi_perimeter - dists))

    return area

def load_data(file_path):
    """Load data using pickle.

    Parameters
    ----------
    file_path : str
        File path containing the data to be loaded.

    Returns
    -------
    data : Any
        Loaded data.
    """

    # Load input data
    with open(file_path, "rb") as input_path:
        data = pickle.load(input_path)
    input_path.close()
    
    return data

def save_data(data, file_path):
    """Load data using pickle.

    Parameters
    ----------
    data : Any
        Data object to be saved.
    file_path : str
        File path where the data will be saved.
    """

    # Save input data
    with open(file_path, "wb") as save_path:
        pickle.dump(data, save_path)
    save_path.close()

    return None

def load_meta_data(modality):
    """Load meta data containing subject demographics.

    Parameters
    ----------
    modality : str
        Type of data modality. Should be either "eeg" or "meg".

    Returns
    -------
    meta_data : pd.DataFrame
        Pandas dataframe that contains subject demographic information.
    """

    # Validation
    if modality not in ["eeg", "meg"]:
        raise ValueError("modality should be either 'eeg' or 'meg'.")

    # Load meta data
    PROJECT_DIR = "/well/woolrich/projects"
    if modality == "eeg":
        fmt = ","
        meta_dir = PROJECT_DIR + "/lemon/raw/Behavioural_Data_MPILMBB_LEMON/META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv"
    if modality == "meg":
        fmt = "\t"
        meta_dir = PROJECT_DIR + "/camcan/participants.tsv"
    meta_data = pd.read_csv(meta_dir, sep=fmt)

    return meta_data

def load_sex_information(subject_ids, modality):
    """Get sex of each subject. The sex should be either Female or Male.

    Parameters
    ----------
    subject_ids : list of str
        List of subject IDs.
    modality : str
        Type of data modality. Should be either "eeg" or "meg".

    Returns
    -------
    sexes : np.ndarray
        1D array marking biological sexes for each subject.
        Female is marked by 0, and male is marked by 1.
    """

    # Load meta data
    meta_data = load_meta_data(modality)

    # Get subject sex (female: 1, male: 2)
    if modality == "eeg":
        sexes = np.array([meta_data.loc[meta_data["ID"] == id]["Gender_ 1=female_2=male"].values[0] for id in subject_ids])
    if modality == "meg":
        sexes = np.array([
            1 if sex == "FEMALE" else 2
            for sex in [meta_data.loc[meta_data["participant_id"] == id]["sex"].values[0] for id in subject_ids]
        ])
    sexes -= 1 # change marks of 1 and 2 to marks of 0 and 1

    return sexes

def load_headsize_information(subject_ids, modality):
    """Get headsize of each subject.

    Parameters
    ----------
    subject_ids : list of str
        List of subject IDs.
    modality : str
        Type of data modality. Should be either "eeg" or "meg".

    Returns
    -------
    headsizes : np.ndarray
        1D array containing the values of headsize for each subject.
    """

    if modality == "eeg":
        data_name = "lemon"
        file_fmt = "{0}/{0}_preproc_raw.fif"
    elif modality == "meg":
        data_name = "camcan"
        file_fmt = "mf2pt2_{0}_ses-rest_task-rest_meg/mf2pt2_{0}_ses-rest_task-rest_meg_preproc_raw.fif"
    PREPROC_DIR = f"/well/woolrich/projects/{data_name}/scho23/preproc"

    headsizes = []
    for id in subject_ids:
        preproc_file = os.path.join(PREPROC_DIR, file_fmt.replace("{0}", id))
        raw = mne.io.read_raw_fif(preproc_file, verbose=False)
        # NOTE: The verbose level of False is still an alias for the argument "WARNING".
        headsizes.append(_get_headsize_from_fids(_get_device_fids(raw)))
        raw.close()

    return np.array(headsizes)

def load_order(modality, n_states, data_type, run_id):
    """Extract a state/mode order of a given run written on the
       excel sheet. This order can be used to match the states/
       modes of a run to those of the reference run.

    Parameters
    ----------
    modality : str
        Type of the modality. Should be either "eeg" or "meg".
    n_states : int
        Number of the states. Should be 6, 8, or 10.
    data_type : str
        Type of the dataset. Should be "full", "split1", or "split2".
    run_id : int
        Number of the model run.

    Returns
    -------
    order : list of int
        Order of the states/modes matched to the reference run.
        Shape is (n_states,). If there is no change in order, None is
        returned.
    """

    # Validation
    if modality not in ["eeg", "meg"]:
        raise ValueError("modality should be either 'egg' or 'meg'.")
    if n_states not in [6, 8, 10]:
        raise ValueError("available # states are 6, 8, and 10.")
    if data_type not in ["full", "split1", "split2"]:
        raise ValueError("input data type not unavailable.")
    
    # Get list of orders
    BASE_DIR = "/well/woolrich/users/olt015/Cho2023_EEG_RSN"
    df = pd.read_excel(os.path.join(BASE_DIR, "data/run_orders.xlsx"))

    # Extract the order of a given run
    index = np.logical_and.reduce((
        df.Modality == modality.upper(),
        df.N_States == n_states,
        df.Data_Type == data_type,
        df.Run_ID == run_id,
    ))
    order = df.Order[index].values[0]
    convert_to_list = lambda x: [int(n) for n in x[1:-1].split(',')]
    order = convert_to_list(order)
    if order == list(np.arange(n_states)):
        order = None
    
    return order