"""Functions for statistical tests and validations

"""

import glmtools as glm
from utils.data import load_sex_information, load_headsize_information

def fit_glm(
        input_data,
        subject_ids,
        group_assignments,
        modality,
        dimension_labels=None,
        plot_verbose=False,
        save_path=""
    ):
    """Fit a General Linear Model (GLM) to an input data given a design matrix.

    Parameters
    ----------
    input_data : np.ndarray
        Data to fit. Shape must be (n_subjects, n_features1, n_features2, ...).
    subject_ids : list of str
        List of subject IDs. Should match the order of subjects in `input_data` 
        and `group_assignments`.
    group_assignments : np.ndarray
        1D numpy array containing group assignments. A value of 1 indicates Group 1 
        and a value of 2 indicates Group 2.
    modality : str
        Type of data modality. Should be either "eeg" or "meg".
    dimension_labels : list of str
        Labels for the dimensions of an input data. Defaults to None, in which 
        case the labels will set as ["Subjects", "Features1", "Features2", ...].
    plot_verbose : bool
        Whether to plot the deisign matrix. Defaults to False.
    save_path : str
        File path to save the design matrix plot. Relevant only when plot_verbose 
        is set to True.
    
    Returns
    -------
    model : glmtools.fit.OLSModel
        A fiited GLM OLS model.
    design : glmtools.design.DesignConfig
        Design matrix object for GLM modelling.
    glm_data : glmtools.data.TrialGLMData
        Data object for GLM modelling.
    """

    # Validation
    ndim = input_data.ndim
    if ndim == 1:
        raise ValueError("data must be 2D or greater.")
    
    if input_data.shape[0] != len(group_assignments):
        raise ValueError("input_data and group_assignments should have the same number of subjects.")
    
    if dimension_labels is None:
        dimension_labels = ["Subjects"] + [f"Features {i}" for i in range(1, ndim)]

    # Define covariates (to regress confounds out)
    sex_assignments = load_sex_information(subject_ids, modality)
    hs_assignments = load_headsize_information(subject_ids, modality)
    covariates = {
        "Sex (Male)": sex_assignments,
        "Sex (Female)": 1 - sex_assignments,
        "Headsize": hs_assignments,
    }

    # Create GLM dataset
    glm_data = glm.data.TrialGLMData(
        data=input_data,
        **covariates,
        category_list=group_assignments,
        dim_labels=dimension_labels,
    )

    # Create design matrix
    DC = glm.design.DesignConfig()
    DC.add_regressor(name="Group1", rtype="Categorical", codes=1)
    DC.add_regressor(name="Group2", rtype="Categorical", codes=2)
    for name in covariates:
        DC.add_regressor(
            name=name,
            rtype="Parametric",
            datainfo=name,
            preproc="demean",
        )
    DC.add_contrast(
        name="GroupDiff",
        values=[1, -1] + [0] * len(covariates),
    ) # contrast: Group 1 - Group 2
    DC.add_contrast(
        name="OverallMean",
        values=[0.5, 0.5] + [0] * len(covariates),
    ) # contrast: (Group 1 + Group 2) / 2
    design = DC.design_from_datainfo(glm_data.info)
    if plot_verbose:
        design.plot_summary(show=False, savepath=save_path)

    # Fit GLM model
    model = glm.fit.OLSModel(design, glm_data)

    return model, design, glm_data