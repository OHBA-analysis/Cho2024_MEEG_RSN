"""Functions for statistical tests and validations

"""

import warnings
import numpy as np
import glmtools as glm
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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

def max_stat_perm_test(
        glm_model,
        glm_data,
        design_matrix,
        pooled_dims,
        contrast_idx,
        n_perm=10000,
        metric="tstats",
        n_jobs=1,
        return_perm=False,
    ):
    """Perform a max-t permutation test to evaluate statistical significance 
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
    n_perm : int, optional
        Number of iterations to permute. Defaults to 10,000.
    metric : str, optional
        Metric to use to build the null distribution. Can be 'tstats' or 'copes'.
    n_jobs : int, optional
        Number of processes to run in parallel.
    return_perm : bool, optional
        Whether to return a glmtools permutation object. Defaults to False.
    
    Returns
    -------
    pvalues : np.ndarray
        P-values for the features. Shape is (n_features1, n_features2, ...).
    perm : glm.permutations.MaxStatPermutation
        Permutation object in the `glmtools` package.
    """

    # Run permutations and get null distributions
    perm = glm.permutations.MaxStatPermutation(
        design_matrix,
        glm_data,
        contrast_idx=contrast_idx,
        nperms=n_perm,
        metric=metric,
        tail=0, # two-sided test
        pooled_dims=pooled_dims,
        nprocesses=n_jobs,
    )
    null_dist = perm.nulls

    # Get p-values
    if metric == "tstats":
        print("Using tstats as metric")
        tstats = abs(glm_model.tstats[0])
        percentiles = stats.percentileofscore(null_dist, tstats)
    elif metric == "copes":
        print("Using copes as metric")
        copes = abs(glm_model.copes[0])
        percentiles = stats.percentileofscore(null_dist, copes)
    pvalues = 1 - percentiles / 100

    if return_perm:
        return pvalues, perm
    return pvalues

def multi_class_prediction(X, y, classifier, n_splits, seed=0):
    """Implements multi-class classifier to predict multiple class labels.
    
    - This function adopts the nested cross-validation method.
    - In the inner loop, a grid search over pre-specified hyperparameter 
    grids is conducted to choose the best hyperparameters. 
    - In the outer loop, the mean test accuracy over all folds is computed 
    to estimate the generalisation error.

    Parameters
    ----------
    X : np.ndarray
        Input features to model. Shape must be (n_samples, n_features).
    y : np.ndarray
        Multi-class target vector (i.e., output features) to model.
        Shape must be (n_samples,).
    classifier : sklearn estimator object
        A classifier to use for a prediction task.
    n_splits : int
        Number of splits/folds for the k-fold cross validation step. Note that  
        n_splits should be less than the smallest cardinality of classes.
    seed : int
        A seed for random number generator(s). Default to 0.
    
    Returns
    -------
    test_score : float
        Mean test accuracy from the nested cross-validation.
    """

    # Initiate k-fold cross validation
    outer_cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed) # divide data into train and test sets
    inner_cv = KFold(n_splits=5, shuffle=True, random_state=seed) # divide train set into train and validation sets

    # Construct prediction pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(whiten=True, random_state=seed)),
        ("clf", classifier),
    ])

    # Set hyperparameter grids
    param_grid = {"pca__n_components": [10, 20, 30]}

    # Perform classifier-based prediction
    clf = GridSearchCV(pipeline, param_grid, cv=inner_cv, n_jobs=5) # inner cross-validation for hyperparameter search
    test_scores = cross_val_score(clf, X, y, cv=outer_cv, n_jobs=5) # outer cross-validation for test evaluation
    test_score = test_scores.mean()
    print(
        "The mean test accuracy using nested cross-validation: "
        f"{test_scores.mean():.3f} +/- {test_scores.std():.3f}"
    )

    return test_score

def repeated_multi_class_prediction(X, y, classifier, n_splits, repeats):
    """Wrapper for `multi_class_prediction`. This function runs a multi-class 
    prediction task multiple times and outputs a test score for each run.

    Parameters
    ----------
    X : np.ndarray
        Input features to model. Shape must be (n_samples, n_features).
    y : np.ndarray
        Multi-class target vector (i.e., output features) to model.
        Shape must be (n_samples,).
    classifier : sklearn estimator object
        A classifier to use for a prediction task.
    n_splits : int
        Number of splits/folds for the k-fold cross validation step. Note that  
        n_splits should be less than the smallest cardinality of classes.
    repeats : int or list of int
        If int, a number of repeats to implement should be provided. The random seeds 
        will be set from 0 to a designated integer. If list, random seed integers  
        should be pre-specified.
    
    Returns
    -------
    test_scores : list of float
        List containing test accuracy of each classification task.
    """

    # Validation
    if isinstance(repeats, int):
        repeats = np.arange(repeats)
    n_repeats = len(repeats)
    print(f"Total number of classification runs: {n_repeats}")

    # Run multi-class prediction
    test_scores = []
    for i, r in enumerate(repeats):
        print(f"[INFO] Repeat #{i + 1}")
        test_scores.append(
            multi_class_prediction(X, y, classifier, n_splits, seed=r)
        )

    # Report descriptive statistics
    mean_test_score = np.mean(test_scores)
    std_test_score = np.std(test_scores)
    print(f"Mean test accuracy (w/ standard dev.): {mean_test_score} +/- {std_test_score}")

    return test_scores

def _check_stat_assumption(samples1, samples2, ks_alpha=0.05, ev_alpha=0.05):
    """Checks normality of each sample and whether samples have an equal variance.

    Parameters
    ----------
    samples1 : np.ndarray
        Array of sample data (group 1). Shape must be (n_samples,).
    samples2 : np.ndarray
        Array of sample data (group 2). Shape must be (n_samples,).
    ks_alpha : float
        Threshold to use for null hypothesis rejection in the Kolmogorov-Smirnov test.
        Defaults to 0.05.
    ev_alpha : float
        Threshold to use for null hypothesis rejection in the equal variance test.
        This test can be the Levene's test or Bartlett's test, depending on the 
        normality of sample distributions. Defaults to 0.05.

    Returns
    -------
    nm_flag : bool
        If True, both samples follow a normal distribution.
    ev_flag : bool
        If True, two sample groups have an equal variance.
    """

    # Set flags for normality and equal variance
    nm_flag, ev_flag = True, True
    print("*** Checking Normality & Equal Variance Assumptions ***")

    # Check normality assumption
    ks_pvals = []
    for s, samples in enumerate([samples1, samples2]):
        stand_samples = stats.zscore(samples)
        res = stats.ks_1samp(stand_samples, cdf=stats.norm.cdf)
        ks_pvals.append(res.pvalue)
        print(f"\t[KS Test] p-value (Sample #{s}): {res.pvalue}")
        if res.pvalue < ks_alpha:
             print(f"\t[KS Test] Sample #{s}: Null hypothesis rejected. The data are not distributed " + 
                   "according to the standard normal distribution.")
    
    # Check equal variance assumption
    if np.sum([pval < ks_alpha for pval in ks_pvals]) != 0:
        nm_flag = False
        # Levene's test
        _, ev_pval = stats.levene(samples1, samples2)
        ev_test_name = "Levene's"
    else:
        # Bartlett's test
        _, ev_pval = stats.bartlett(samples1, samples2)
        ev_test_name = "Bartlett's"
    print(f"\t[{ev_test_name} Test] p-value: ", ev_pval)
    if ev_pval < ev_alpha:
        print(f"\t[{ev_test_name} Test] Null hypothesis rejected. The populations do not have equal variances.")
        ev_flag = False

    return nm_flag, ev_flag

def stat_ind_two_samples(samples1, samples2, alpha=0.05, bonferroni_ntest=None, test=None):
    """Performs a statistical test comparing two independent samples.

    Parameters
    ----------
    samples1 : np.ndarray
        Array of sample data (group 1). Shape must be (n_samples,).
    samples2 : np.ndarray
        Array of sample data (group 2). Shape must be (n_samples,).
    alpha : float
        Threshold to use for null hypothesis rejection. Defaults to 0.05.
    bonferroni_ntest : int
        Number of tests to be used for Bonferroni correction. Default to None.
    test : str
        Statistical test to use. Defaults to None, which automatically selects
        the test after checking the assumptions.

    Returns
    -------
    stat : float
        The test statistic. The test can be the Student's t-test, Welch's t-test, 
        or Wilcoxon Rank Sum test depending on the test assumptions.
    pval : float
        The p-value of the test.
    sig_indicator : bool
        Whether the p-value is significant or not. If bonferroni_ntest is given, 
        the p-value will be evaluated against the corrected threshold.
    """

    # Check normality and equal variance assumption
    if test is None:
        nm_flag, ev_flag = _check_stat_assumption(samples1, samples2)
    else:
        if test == "ttest":
            nm_flag, ev_flag = True, True
        elif test == "welch":
            nm_flag, ev_flag = True, False
        elif test == "wilcoxon":
            nm_flag, ev_flag = False, True

    # Compare two independent groups
    print("*** Comparing Two Independent Groups ***")
    if nm_flag and ev_flag:
        print("\tConducting the two-samples independent T-Test ...")
        stat, pval = stats.ttest_ind(samples1, samples2, equal_var=True)
    if nm_flag and not ev_flag:
        print("\tConducting the Welch's t-test ...")
        stat, pval = stats.ttest_ind(samples1, samples2, equal_var=False)
    if not nm_flag:
        print("\tConducting the Wilcoxon Rank Sum test ...")
        if not ev_flag:
            warnings.warn("Caution: Distributions have unequal variances.", UserWarning)
        stat, pval = stats.ranksums(samples1, samples2)
    print(f"\tResult: statistic={stat} | p-value={pval}")

    # Apply Bonferroni correction
    if bonferroni_ntest is not None:
        alpha /= bonferroni_ntest
    sig_indicator = pval < alpha
    print(f"[Bonferroni Correction] Threshold: {alpha}, Significance: {sig_indicator}")

    return stat, pval, sig_indicator

def stat_ind_one_samples(samples, popmean, alpha=0.05, bonferroni_ntest=None):
    """Performs a one-sample independent T-Test.
    
    - Null hypothesis: the expected value (mean) of a sample of independent observations 
    is equal to the given population mean.
    - Alternative hypothesis: the expected value (mean) of a sample is greater than the 
    given population mean.

    Parameters
    ----------
    samples : np.ndarray
        Array of sample data. Shape must be (n_samples,).
    popmean : float
        Expected value in null hypothesis.
    alpha : float
        Threshold to use for null hypothesis rejection. Defaults to 0.05.
    bonferroni_ntest : int
        Number of tests to be used for Bonferroni correction. Default to None.

    Returns
    -------
    stat : float
        The test statistic (i.e., t-statistics).
    pval : float
        The p-value of the test.
    sig_indicator : bool
        Whether the p-value is significant or not. If bonferroni_ntest is given, 
        the p-value will be evaluated against the corrected threshold.
    """

    # Check normality assumption
    print("*** Checking Normality Assumption ***")
    nm_flag = True
    stand_samples = stats.zscore(samples)
    ks_res = stats.ks_1samp(stand_samples, cdf=stats.norm.cdf)
    print(f"\t[KS Test] p-value: {ks_res.pvalue}")
    if ks_res.pvalue < 0.05:
        nm_flag = False
        print(f"\t[KS Test] Null hypothesis rejected. The data are not distributed " + 
            "according to the standard normal distribution.")
    
    # Compare one independent group
    print("*** Comparing One Independent Group ***")
    if not nm_flag:
        warnings.warn("Caution: Sample distribution is not normal.", UserWarning)    
    print("\tConducting the one-sample independent T-Test ...")
    res = stats.ttest_1samp(samples, popmean=popmean, alternative="greater")
    print(f"\tResult: statistic={res.statistic} | p-value={res.pvalue}")

    # Apply Bonferroni correction
    if bonferroni_ntest is not None:
        alpha /= bonferroni_ntest
    sig_indicator = res.pvalue < alpha
    print(f"[Bonferroni Correction] Threshold: {alpha}, Significance: {sig_indicator}")

    return res.statistic, res.pvalue, sig_indicator
