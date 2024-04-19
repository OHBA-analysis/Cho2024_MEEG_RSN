"""Preparing the Cam-CAN MEG data
[STEP 3] Source reconstruction: forward modelling, beamforming, and parcellation

NOTE: Before running this script, the /coreg directory created by coregister.py
      must be copied and renamed to /src.
"""

# Install dependencies
import os
import glob
from osl import source_recon, utils
from dask.distributed import Client

# Set directories
BASE_DIR = "/home/scho/camcan"
PREPROC_DIR = os.path.join(BASE_DIR, "scho23/preproc")
SRC_DIR = os.path.join(BASE_DIR, "scho23/src")
FSL_DIR = "/opt/ohba/fsl/6.0.5"

# Configure pipeline
config = """
    source_recon:
        - forward_model:
            model: Single Layer
        - beamform_and_parcellate:
            freq_range: [1, 45]
            chantypes: [mag, grad]
            rank: {meg: 50}
            parcellation_file: fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz
            method: spatial_basis
            orthogonalisation: symmetric
"""

# SOURCE RECONSTRUCT DATA
if __name__ == "__main__":
    # Set logger
    utils.logger.set_up(level="INFO")

    # Set up FSL
    source_recon.setup_fsl(FSL_DIR)

    # Get file paths
    subjects = [file.split("/")[-1] for file in sorted(glob.glob(os.path.join(SRC_DIR, "sub-*")))]    
    preproc_files = [os.path.join(
        PREPROC_DIR,
        f"mf2pt2_{subject}_ses-rest_task-rest_meg",
        f"mf2pt2_{subject}_ses-rest_task-rest_meg_preproc_raw.fif"
    ) for subject in subjects]
    print(f"Number of available subjects: {len(subjects)}")

    # Set up parallel processing
    client = Client(n_workers=8, threads_per_worker=1)

    # Initiate forward modelling, beamforming, and parcellation
    source_recon.run_src_batch(
        config,
        src_dir=SRC_DIR,
        subjects=subjects,
        preproc_files=preproc_files,
        dask_client=True,
    )

    print("Source reconstruction complete.")
