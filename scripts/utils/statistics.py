"""Functions for statistical tests and validations

"""

import numpy as np
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

def cluster_perm_test(
        glm_model,
        glm_data,
        design_matrix,
        pooled_dims,
        contrast_idx,
        n_perm=1500,
        metric="tstats",
        bonferroni_ntest=1,
        n_jobs=1,
        return_perm=False,
    ):
    """Perform a cluster permutation test to evaluate statistical significance 
       for the given contrast.

    Parameters
    ----------
    glm_model : glmtools.fit.OLSModel
        A fitted GLM OLS model.
    glm_data : glmtools.data.TrialGLMData
        Data object for GLM modelling.
    design_matrix : glmtools.design.DesignConfig
        Design matrix object for GLM modelling.
    pooled_dims : int or tuples
        Dimension(s) to pool over.
    contrast_idx : int
        Index indicating which contrast to use. Dependent on glm_model.
    n_perm : int
        Number of iterations to permute. Defaults to 1,500.
    metric : str, optional
        Metric to use to build the null distribution. Can be 'tstats' or 'copes'.
    bonferroni_ntest : int
        Number of tests to use for Bonferroni correction. Defaults to 1 (i.e., no
        Bonferroni correction applied).
    n_jobs : int, optional
        Number of processes to run in parallel.
    return_perm : bool, optional
        Whether to return a glmtools permutation object. Defaults to False.
    
    Returns
    -------
    obs : np.ndarray
        Statistic observed for all variables. Values can be 'tstats' or 'copes'
        depending on the `metric`. Shape is (n_freqs,).
    clusters : list of np.ndarray
        List of ndarray, each of which contains the indices that form the given 
        cluster along the tested dimension. If bonferroni_ntest was given, clusters 
        after Bonferroni correction are returned.
    perm : glm.permutations.ClusterPermutation
        Permutation object in the `glmtools` package.
    """

    # Get metric values and define cluster forming threshold
    if metric == "tstats":
        obs = np.squeeze(glm_model.tstats)
        cft = 3
    if metric == "copes":
        obs = np.squeeze(glm_model.copes)
        cft = 0.001

    # Run permutations and get null distributions
    perm = glm.permutations.ClusterPermutation(
        design=design_matrix,
        data=glm_data,
        contrast_idx=contrast_idx,
        nperms=n_perm,
        metric=metric,
        tail=0, # two-sided test
        cluster_forming_threshold=cft,
        pooled_dims=pooled_dims,
        nprocesses=n_jobs,
    )

    # Extract significant clusters
    percentile = (1 - (0.05 / bonferroni_ntest)) * 100
    # NOTE: We use alpha threshold of 0.05.
    clu_masks, clu_stats = perm.get_sig_clusters(glm_data, percentile)
    if clu_stats is not None:
        n_clusters = len(clu_stats)
    else: n_clusters = 0
    print(f"After Boneferroni correction: Found {n_clusters} clusters")

    # Get indices of significant channels and frequencies
    clusters = [
        np.arange(len(clu_masks))[clu_masks == n]
        for n in range(1, n_clusters + 1)
    ]

    if return_perm:
        return obs, clusters, perm
    return obs, clusters