"""Preparing the LEMON EEG data
[STEP 2] Source reconstruct preprocessed sensor space data

NOTE: This script was adapted from `lemon/chet22`.
"""

# Install dependencies
import os
import glob
import numpy as np
from sys import argv
from osl import source_recon, utils
from dask.distributed import Client

# Set hyperparameters
if len(argv) != 2:
    print("Need to pass one argument: use of structurals (e.g., python script.py subject)")
    exit()
structurals = argv[1] # type of structurals to use
print(f"[INFO] Structurals: {structurals}")

# Set directories
BASE_DIR = "/well/woolrich/projects/lemon"
RAW_DIR = os.path.join(BASE_DIR, "raw")
PREPROC_DIR = os.path.join(BASE_DIR, "scho23/preproc")
FSL_DIR = "/well/woolrich/projects/software/fsl"
if structurals == "subject": # use individual sMRI files
    SRC_DIR = os.path.join(BASE_DIR, "scho23/src")
elif structurals == "standard": # use standard MNI brain
    SRC_DIR = os.path.join(BASE_DIR, "scho23/src_no_struct")

# Configure pipeline
config = f"""
    source_recon:
        - extract_fiducials_from_fif:
            include_eeg_as_headshape: true
        - fix_headshape_fiducials: {{structurals: {structurals}}}
        - compute_surfaces_coregister_and_forward_model:
            include_nose: false
            use_nose: false
            use_headshape: false
            model: Triple Layer
            eeg: true
            allow_smri_scaling: true
        - beamform_and_parcellate:
            freq_range: [1, 45]
            chantypes: eeg
            rank: {{eeg: 50}}
            parcellation_file: fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz
            method: spatial_basis
            orthogonalisation: symmetric
"""

# Define extra functions
def fix_headshape_fiducials(src_dir, subject, preproc_file, smri_file, epoch_file, structurals):
    # Get coreg filenames
    filenames = source_recon.rhino.get_coreg_filenames(src_dir, subject)

    # Load saved headshape and nasion files
    hs = np.loadtxt(filenames["polhemus_headshape_file"])
    nas = np.loadtxt(filenames["polhemus_nasion_file"])
    lpa = np.loadtxt(filenames["polhemus_lpa_file"])
    rpa = np.loadtxt(filenames["polhemus_rpa_file"])

    # Shrink headshape points by 5%
    if structurals == "subject":
        hs *= 0.95
    else:
        utils.logger.log_or_print(f"skipping headshape point adjustment.")

    # Move fiducials down 1 cm
    nas[2] -= 10
    lpa[2] -= 10
    rpa[2] -= 10

    # Move fiducials back 1 cm
    nas[1] -= 10
    lpa[1] -= 10
    rpa[1] -= 10

    # Overwrite files
    utils.logger.log_or_print(f"overwritting {filenames['polhemus_nasion_file']}")
    np.savetxt(filenames["polhemus_nasion_file"], nas)
    utils.logger.log_or_print(f"overwritting {filenames['polhemus_lpa_file']}")
    np.savetxt(filenames["polhemus_lpa_file"], lpa)
    utils.logger.log_or_print(f"overwritting {filenames['polhemus_rpa_file']}")
    np.savetxt(filenames["polhemus_rpa_file"], rpa)
    utils.logger.log_or_print(f"overwritting {filenames['polhemus_headshape_file']}")
    np.savetxt(filenames["polhemus_headshape_file"], hs)

# SOURCE RECONSTRUCT DATA
if __name__ == "__main__":
    # Set logger
    utils.logger.set_up(level="INFO")

    # Set up FSL
    source_recon.setup_fsl(FSL_DIR)

    # Get file paths
    if structurals == "subject":
        preproc_files = sorted(glob.glob(os.path.join(PREPROC_DIR, "*/*_preproc_raw.fif")))
        subject_ids = [file.split("/")[-1].split("_")[0] for file in preproc_files]
        smri_files = [
            os.path.join(RAW_DIR, f"{subject}/ses-01/anat/{subject}_ses-01_inv-2_mp2rage.nii.gz") 
            for subject in subject_ids
        ]
        print("NOTE: Subject-specific sMRI files will be used.")
    else:
        subject_ids = [file.split("/")[-2] for file in sorted(glob.glob(os.path.join(BASE_DIR, "scho23/src_ec/*/sflip_parc-raw.npy")))]
        preproc_files = [BASE_DIR + f"/scho23/preproc/{id}/{id}_preproc_raw.fif" for id in subject_ids]
        smri_files = ["/well/woolrich/projects/software/fsl/data/standard/MNI152_T1_1mm.nii.gz"] * len(subject_ids)
        # NOTE: For source reconstruction without structurals (i.e., using the standard brain), we only use subject data that had 
        #       successful source reconstruction with structurals.
        print(f"NOTE: The standard MNI file will be used. ({smri_files[0]})")

    print(f"Number of available subjects: {len(subject_ids)}")

    # Set up parallel processing
    client = Client(n_workers=8, threads_per_worker=1)

    # Initiate source reconstruction
    source_recon.run_src_batch(
        config,
        src_dir=SRC_DIR,
        subjects=subject_ids,
        preproc_files=preproc_files,
        smri_files=smri_files,
        extra_funcs=[fix_headshape_fiducials],
        dask_client=True,
    )

    print("Source reconstruction complete.")
