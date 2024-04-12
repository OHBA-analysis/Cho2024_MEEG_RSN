"""Preparing the Cam-CAN MEG Data
[STEP 4] Apply dipole sign flipping on the source space data
"""

# Install dependencies
import os
import glob
from sys import argv
from osl import source_recon, utils
from dask.distributed import Client

# Set hyperparameters
if len(argv) != 2:
    print("Need to pass one argument: use of structurals (e.g., python script.py subject)")
    exit()
structurals = argv[1] # type of structurals used
print(f"[INFO] Structurals: {structurals}")

# Set directories
BASE_DIR = "/home/scho/camcan"
FSL_DIR = "/opt/ohba/fsl/6.0.5"
if structurals == "subject": # individual sMRI files
    SRC_DIR = os.path.join(BASE_DIR, "scho23/src")
elif structurals == "standard": # standard MNI brain
    SRC_DIR = os.path.join(BASE_DIR, "scho23/src_no_struct")

# SIGN FLIP DATA
if __name__ == "__main__":
    # Set logger
    utils.logger.set_up(level="INFO")

    # Set up FSL
    source_recon.setup_fsl(FSL_DIR)

    # Get subject IDs
    subjects = [
        file.split("/")[-3] for file in sorted(glob.glob(os.path.join(SRC_DIR, "*/parc/parc-raw.fif")))
    ]
    print(f"Number of available subjects: {len(subjects)}")

    # Find a good template subject to align other subjects to
    template = source_recon.find_template_subject(
        SRC_DIR, subjects, n_embeddings=15, standardize=True
    )

    # Configure pipeline
    config = f"""
        source_recon:
            - fix_sign_ambiguity:
                template: {template}
                n_embeddings: 15
                standardize: true
                n_init: 3
                n_iter: 5000
                max_flips: 20
    """

    # Set up parallel processing
    client = Client(n_workers=4, threads_per_worker=1)

    # Initiate sign flipping
    source_recon.run_src_batch(
        config,
        src_dir=SRC_DIR,
        subjects=subjects,
        dask_client=True,
    )

    print("Sign flipping complete.")
