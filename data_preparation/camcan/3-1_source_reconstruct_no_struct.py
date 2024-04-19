"""Preparing the Cam-CAN MEG data
[STEP 3-1] Source reconstruct preprocessed sensor space data (without sMRI files)
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
structurals = argv[1]
print(f"[INFO] Structurals: {structurals}")

# Set directories
BASE_DIR = "/home/scho/camcan"
PREPROC_DIR = os.path.join(BASE_DIR, "scho23/preproc")
SRC_DIR = os.path.join(BASE_DIR, "scho23/src_no_struct")
FSL_DIR = "/opt/ohba/fsl/6.0.5"

# Configure pipeline
config = """
    source_recon:
        - extract_fiducials_from_fif: {}
        - fix_headshape_points: {}
        - compute_surfaces_coregister_and_forward_model:
            include_nose: false
            use_nose: false
            use_headshape: true
            model: Single Layer
            allow_smri_scaling: true
        - beamform_and_parcellate:
            freq_range: [1, 45]
            chantypes: [mag, grad]
            rank: {meg: 50}
            parcellation_file: fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz
            method: spatial_basis
            orthogonalisation: symmetric
"""

# Define external functions
def fix_headshape_points(src_dir, subject, preproc_file, smri_file, epoch_file):
    filenames = source_recon.rhino.get_coreg_filenames(src_dir, subject)

    # Load saved headshape and nasion files
    hs = np.loadtxt(filenames["polhemus_headshape_file"])
    nas = np.loadtxt(filenames["polhemus_nasion_file"])
    lpa = np.loadtxt(filenames["polhemus_lpa_file"])
    rpa = np.loadtxt(filenames["polhemus_rpa_file"])

    # Drop nasion by 4cm
    nas[2] -= 40
    distances = np.sqrt(
        (nas[0] - hs[0]) ** 2 + (nas[1] - hs[1]) ** 2 + (nas[2] - hs[2]) ** 2
    )

    # Keep headshape points more than 7 cm away
    keep = distances > 70
    hs = hs[:, keep]

    # Remove anything outside of rpa
    keep = hs[0] < rpa[0]
    hs = hs[:, keep]

    # Remove anything outside of lpa
    keep = hs[0] > lpa[0]
    hs = hs[:, keep]

    if subject in [
        "sub-CC110606", "sub-CC120061", "sub-CC120727", "sub-CC221648", "sub-CC221775",
        "sub-CC121397", "sub-CC121795", "sub-CC210172", "sub-CC210519", "sub-CC210657",
        "sub-CC220419", "sub-CC220974", "sub-CC221107", "sub-CC222956", "sub-CC221828",
        "sub-CC222185", "sub-CC222264", "sub-CC222496", "sub-CC310224", "sub-CC310361",
        "sub-CC310414", "sub-CC312222", "sub-CC320022", "sub-CC320088", "sub-CC320321",
        "sub-CC320336", "sub-CC320342", "sub-CC320448", "sub-CC322186", "sub-CC410243",
        "sub-CC420091", "sub-CC420094", "sub-CC420143", "sub-CC420167", "sub-CC420241",
        "sub-CC420261", "sub-CC420383", "sub-CC420396", "sub-CC420435", "sub-CC420493",
        "sub-CC420566", "sub-CC420582", "sub-CC420720", "sub-CC510255", "sub-CC510321",
        "sub-CC510323", "sub-CC510415", "sub-CC510480", "sub-CC520002", "sub-CC520011",
        "sub-CC520042", "sub-CC520055", "sub-CC520078", "sub-CC520127", "sub-CC520239",
        "sub-CC520254", "sub-CC520279", "sub-CC520377", "sub-CC520391", "sub-CC520477",
        "sub-CC520480", "sub-CC520552", "sub-CC520775", "sub-CC610050", "sub-CC610576",
        "sub-CC620354", "sub-CC620406", "sub-CC620479", "sub-CC620518", "sub-CC620557",
        "sub-CC620572", "sub-CC620610", "sub-CC620659", "sub-CC621011", "sub-CC621128",
        "sub-CC621642", "sub-CC710131", "sub-CC710350", "sub-CC710551", "sub-CC710591",
        "sub-CC710982", "sub-CC711128", "sub-CC720119", "sub-CC720188", "sub-CC720238",
        "sub-CC720304", "sub-CC720358", "sub-CC720511", "sub-CC720622", "sub-CC720685",
        "sub-CC721292", "sub-CC721504", "sub-CC721519", "sub-CC721891", "sub-CC721374",
        "sub-CC722542", "sub-CC722891", "sub-CC121111", "sub-CC121144", "sub-CC210250",
        "sub-CC210422", "sub-CC220519", "sub-CC221209", "sub-CC221487", "sub-CC221595",
        "sub-CC221886", "sub-CC310331", "sub-CC410121", "sub-CC410179", "sub-CC420157",
        "sub-CC510395", "sub-CC610653",
    ]:
        # Remove headshape points 1cm below lpa
        keep = hs[2] > (lpa[2] - 10)
        hs = hs[:, keep]
    elif subject in [
        "sub-CC210023", "sub-CC210124", "sub-CC220132", "sub-CC220203", "sub-CC220223",
        "sub-CC221054", "sub-CC310410", "sub-CC320089", "sub-CC320651", "sub-CC721114",
        "sub-CC320680", "sub-CC320759", "sub-CC320776", "sub-CC320850", "sub-CC320888",
        "sub-CC320904", "sub-CC321073", "sub-CC321154", "sub-CC321174", "sub-CC410084",
        "sub-CC410101", "sub-CC410173", "sub-CC410226", "sub-CC410287", "sub-CC410325",
        "sub-CC410432", "sub-CC420089", "sub-CC420149", "sub-CC420197", "sub-CC420198",
        "sub-CC420217", "sub-CC420222", "sub-CC420260", "sub-CC420324", "sub-CC420356",
        "sub-CC420454", "sub-CC420589", "sub-CC420888", "sub-CC510039", "sub-CC510115",
        "sub-CC510161", "sub-CC510237", "sub-CC510258", "sub-CC510355", "sub-CC510438",
        "sub-CC510551", "sub-CC510609", "sub-CC510629", "sub-CC510648", "sub-CC512003",
        "sub-CC520013", "sub-CC520065", "sub-CC520083", "sub-CC520097", "sub-CC520147",
        "sub-CC520168", "sub-CC520209", "sub-CC520211", "sub-CC520215", "sub-CC520247",
        "sub-CC520395", "sub-CC520398", "sub-CC520503", "sub-CC520584", "sub-CC610052",
        "sub-CC610178", "sub-CC610210", "sub-CC610212", "sub-CC610288", "sub-CC610575",
        "sub-CC610631", "sub-CC620085", "sub-CC620090", "sub-CC620121", "sub-CC620164",
        "sub-CC620444", "sub-CC620496", "sub-CC620526", "sub-CC620592", "sub-CC620793",
        "sub-CC620919", "sub-CC710313", "sub-CC710486", "sub-CC710566", "sub-CC720023",
        "sub-CC720497", "sub-CC720516", "sub-CC720646", "sub-CC721224", "sub-CC721729",
        "sub-CC723395", "sub-CC222326", "sub-CC310160", "sub-CC121479", "sub-CC121685",
        "sub-CC221755", "sub-CC320687", "sub-CC620152", "sub-CC711244",
    ]:
        # Remove headshape points below rpa
        keep = hs[2] > rpa[2]
        hs = hs[:, keep]
    elif subject in [
        "sub-CC210617", "sub-CC220107", "sub-CC220198", "sub-CC220234", "sub-CC220323",
        "sub-CC220335", "sub-CC222125", "sub-CC222258", "sub-CC310008", "sub-CC610046",
        "sub-CC610508",
    ]:
        # Remove headshape points on the face
        keep = np.logical_or(hs[2] > lpa[2], hs[1] < lpa[1])
        hs = hs[:, keep]
    elif subject in [
        "sub-CC410129", "sub-CC410222", "sub-CC410323", "sub-CC410354", "sub-CC420004",
        "sub-CC410390", "sub-CC420348", "sub-CC420623", "sub-CC420729", "sub-CC510043",
        "sub-CC510086", "sub-CC510304", "sub-CC510474", "sub-CC520122", "sub-CC521040",
        "sub-CC610101", "sub-CC610146", "sub-CC610292", "sub-CC620005", "sub-CC620284",
        "sub-CC620413", "sub-CC620490", "sub-CC620515", "sub-CC621199", "sub-CC710037",
        "sub-CC710214", "sub-CC720103", "sub-CC721392", "sub-CC721648", "sub-CC721888",
        "sub-CC722421", "sub-CC722536", "sub-CC720329",
    ]:
        # Remove headshape points 1cm above lpa
        keep = hs[2] > (lpa[2] + 10)
        hs = hs[:, keep]
    elif subject in [
        "sub-CC321428", "sub-CC410097", "sub-CC510076", "sub-CC510220", "sub-CC520560",
        "sub-CC520597", "sub-CC520673", "sub-CC610285", "sub-CC610469", "sub-CC620429",
        "sub-CC620451", "sub-CC620821", "sub-CC710494", "sub-CC722651", "sub-CC110101",
        "sub-CC122172",
    ]:
        # Remove headshape points 2cm above lpa
        keep = hs[2] > (lpa[2] + 20)
        hs = hs[:, keep]
    elif subject in ["sub-CC412004", "sub-CC721704"]:
        # Remove headshape points 3cm above rpa
        keep = hs[2] > (rpa[2] + 30)
        hs = hs[:, keep]
    elif subject in [
        "sub-CC110033", "sub-CC510163", "sub-CC520287", "sub-CC520607", "sub-CC620567",
    ]:
        # Remove headshape points 2cm below lpa
        keep = hs[2] > (lpa[2] - 20)
        hs = hs[:, keep]

    # Overwrite headshape file
    utils.logger.log_or_print(f"overwritting {filenames['polhemus_headshape_file']}")
    np.savetxt(filenames["polhemus_headshape_file"], hs)

# SOURCE RECONSTRUCT DATA
if __name__ == "__main__":
    # Set logger
    utils.logger.set_up(level="INFO")

    # Set up FSL
    source_recon.setup_fsl(FSL_DIR)

    # Get file paths
    preproc_files = sorted(glob.glob(os.path.join(
        PREPROC_DIR,
        "mf2pt2_*_ses-rest_task-rest_meg",
        "mf2pt2_sub-*_ses-rest_task-rest_meg_preproc_raw.fif"
    )))
    subjects = [file.split('/')[-1].split('_')[1] for file in preproc_files]
    smri_files = ["/opt/ohba/fsl/6.0.5/data/standard/MNI152_T1_1mm.nii.gz"] * len(subjects)
    # NOTE: For source reconstruction without structurals (i.e., using the standard brain), we only use subject data that had 
    #       successful source reconstruction with structurals.
    print(f"NOTE: The standard MNI file will be used. ({smri_files[0]})")
    print(f"Number of available subjects: {len(subjects)}")

    # Set up parallel processing
    client = Client(n_workers=8, threads_per_worker=1)

    # Initiate forward modelling, beamforming, and parcellation
    source_recon.run_src_batch(
        config,
        src_dir=SRC_DIR,
        subjects=subjects,
        preproc_files=preproc_files,
        smri_files=smri_files,
        extra_funcs=[fix_headshape_points],
        dask_client=True,
    )

    print("Source reconstruction (w/o structurals) complete.")
