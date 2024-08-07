"""Functions for visualizing results

"""

import os, pickle, mne
import numpy as np
import pandas as pd
import seaborn as sns
import nibabel as nib
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LinearSegmentedColormap, Normalize, CenteredNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from itertools import cycle
from nilearn.plotting import plot_glass_brain
from osl_dynamics import files
from osl_dynamics.analysis import power, connectivity
from osl_dynamics.utils import plotting
from osl_dynamics.utils.parcellation import Parcellation
from utils.array_ops import round_nonzero_decimal, round_up_half
from utils.statistics import fit_glm, cluster_perm_test

def _colormap_transparent(cmap_name, start_opacity=0.2, end_opacity=1.0):
    """Add transparency to a selected colormap registered in matplotlib.

    Parameters
    ----------
    cmap_name : str
        Name of the colormap.
    start_opacity : float
        Alpha transparency level at the start of a colormap. Defaults to 
        0.3.
    end_opacity : float
        Alpha transparency level at the end of a colormap. Defaults to 
        1.0.

    Returns
    -------
    custom_colormap : matplotlib.colors.Colormap
        A matplotlib colormap object.
    """

    # Define the number of colors
    n_colors = 256

    # Get the default plasma colormap
    cmap = plt.cm.get_cmap(cmap_name, n_colors)

    # Modify the plasma coormap
    alphas = np.linspace(start_opacity, end_opacity, n_colors)
    colors_with_alpha = [(cmap(i)[:3] + (alpha,)) for i, alpha in enumerate(alphas)]
    custom_colormap = LinearSegmentedColormap.from_list(
        "colormap_transparent", colors_with_alpha, N=n_colors
    )

    return custom_colormap

def _colormap_null(name):
    """Creates a custom colormap with a single color, located at the middle of a 
    given color palette. This function is specifically designed to plot background 
    power / connectivity maps with the nilearn package.

    Parameters
    ----------
    name : str
        Name of the colormap.

    Returns
    -------
    cmap : matplotlib.colors.LinearSegmentedColormap
        Customized Matplotlib colormap object.
    """

    # Get original colormap
    n_colors = 256
    colors = plt.get_cmap(name)(range(n_colors))

    # Fill colormap with a color at the middle
    mid_clr = colors[n_colors // 2]
    colors = np.tile(mid_clr, (n_colors, 1))

    # Create a colormap object
    cmap = LinearSegmentedColormap.from_list(name="custom_cmap", colors=colors)

    return cmap

def _format_colorbar_ticks(ax):
    """Formats x-axis ticks in the colobar such that integer values are 
    plotted, instead of decimal values.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        A colobar axis to format.
    """

    if np.any(np.abs(ax.get_xlim()) < 1):
        hmin = round_nonzero_decimal(ax.get_xlim()[0], method="ceil") # ceiling for negative values
        hmax = round_nonzero_decimal(ax.get_xlim()[1], method="floor") # floor for positive values
        ax.set_xticks(np.array([hmin, 0, hmax]))
    else:
        ax.set_xticks(
            [round_up_half(val) for val in ax.get_xticks()[1:-1]]
        )
    
    return None

def load_accessible_colors(palette_name="tol_bright", sel_ind=None):
    """Loads colour-blind-friendly colours from a designated colour palette.

    Parameters
    ----------
    palette_name : str
        Name of the colour palette to load colours from. Defaults to `tol_bright`.
    sel_ind : list of int
        Indices to select or redorder specific colours. Defaults to None.

    Returns
    -------
    palette : list of str
        A list of accessible colours in hex codes.
    """

    palette = {
        "tol_bright": ["#4477AA", "#66CCEE", "#228833", "#CCBB44", 
                       "#EE6677", "#AA3377", "#BBBBBB"],
        "tol_vibrant": ["#0077BB", "#33BBEE", "#009988", "#EE7733", 
                        "#CC3311", "#EE3377", "#BBBBBB"],
        "tol_muted": ["#332288", "#88CCEE", "#44AA99", "#117733", 
                      "#999933", "#DDCC77", "#CC6677", "#882255", 
                      "#AA4499", "#DDDDDD"],
        "okabe_ito": ["#E69F00", "#56B4E9", "#009E73", "#F0E442", 
                      "#0072B2", "#D55E00", "#CC79A7", "#000000"],
    }
    if sel_ind:
        return [palette[palette_name][ind] for ind in sel_ind]
    return palette[palette_name]
    
def plot_loss_curve(loss, x_step=5, save_dir=None):
    """Plots a training/validation/test loss curve.

    Parameters
    ----------
    loss : list
        Array of loss values.
    x_step : int
        Number of epoch steps for the x-axis ticks.
    save_dir : str
        Directory where a figure object will be saved.
    """

    # Validation
    if save_dir is None:
        save_dir = os.get_cwd()

    # Get epoch array
    epochs = np.arange(1, len(loss) + 1)

    # Plot loss curve
    fig, ax = plotting.plot_line([epochs], [loss], plot_kwargs={"lw": 2})
    ax.set_xticks(np.arange(0, len(loss) + x_step, x_step))
    ax.tick_params(axis='both', which='both', labelsize=18, width=2)
    ax.set_xlabel("Epochs", fontsize=18)
    ax.set_ylabel("Loss", fontsize=18)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss.png"))
    plt.close(fig)

    return None

def plot_age_distributions(ages_young, ages_old, modality, nbins="auto", filename=None):
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
    filename : str
        File name to be used when saving a figure object. By default, the 
        plot will be saved to a user's current directory with a name 
        `age_dist_{modality}.png`.
    """

    # Validation
    if modality not in ["eeg", "meg"]:
        raise ValueError("modality should be either 'eeg' or 'meg'.")
    if not isinstance(nbins, list):
        nbins = [nbins, nbins]
    if filename is None:
        filename = os.path.join(os.get_cwd(), f"age_dist_{modality}.png")

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
    fig.savefig(filename)
    plt.close(fig)

    return None

def plot_nnmf_components(freqs, components, filename, comp_lbls=None, fontsize=18):
    """Plots NNMF spectral component(s) in one plot.

    Parameters
    ----------
    freqs : np.ndarray
        Array of frequencies in the frequency axis. Shape is (n_freqs,).
    components : np.ndarray
        Decomposed spectral components. Shape is (n_components, n_freqs) 
        or (n_freqs,).
    filename : str
        File name to be used when saving a figure object.
    comp_lbls : list of str
        Legend labels for each component. Defaults to None.
    fontsize : int
        Fontsize for a plot. Defaults to 18.
    """
    
    # Validation
    if components.ndim < 2:
        components = components[np.newaxis, ...]
    n_components = components.shape[0]
    
    if comp_lbls is None:
        comp_lbls = [None] * n_components
    
    # Set colormap
    cmap = sns.color_palette("colorblind", n_colors=n_components)

    # Visualize spectral components
    fig, ax = plt.subplots(nrows=1, ncols=1)
    for n in range(n_components):
        ax.plot(freqs, components[n, :], color=cmap[n], lw=2, label=comp_lbls[n])
        ax.fill_between(freqs, components[n, :], color=cmap[n], alpha=0.3)
    ax.set_xlabel("Frequency (Hz)", fontsize=fontsize)
    ax.set_ylabel("Spectral Mode Magnitude", fontsize=fontsize)
    if any(lbl is not None for lbl in comp_lbls):
        ax.legend(loc="best", fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    plt.tight_layout()
    fig.savefig(filename, transparent=True)
    plt.close(fig)
    
    return None

def plot_surfaces(
        data_map,
        mask_file,
        parcellation_file,
        vmin=None,
        vmax=None,
        symmetric_cbar=True,
        figure=None,
        axis=None,
    ):
    """Wrapper of the `plot_glass_brain()` function in the nilearn package.

    Parameters
    ----------
    data_map : np.ndarray
        Data array containing values to be plotted on brain surfaces.
        Shape must be (n_parcels,).
    mask_file : str
        Path to a masking file.
    parcellation_file : str
        Path to a brain parcellation file.
    vmin : float
        Minimum value of the data. Acts as a lower bound of the colormap.
    vmax : float
        Maximum value of the data. Acts as an upper bound of the colormap.
    symmetric_cbar : bool
        Whether the colorbar should be symmetric. Defaults to True.
    figure : matplotlib.pyplot.Figure
        Matplotlib figure object.
    axis : matplotlib.axes.axes
        Axis object to plot on.
    """

    # Create a copy of the data map so we don't modify it
    data_map = np.copy(data_map)

    # Validation
    mask_file = files.check_exists(mask_file, files.mask.directory)
    parcellation_file = files.check_exists(
        parcellation_file, files.parcellation.directory
    )

    # Calculate data map grid
    data_map = power.parcel_vector_to_voxel_grid(mask_file, parcellation_file, data_map)

    # Load the mask
    mask = nib.load(mask_file)

    # Reset color ranges if symmetric
    if symmetric_cbar:
        vmax = np.max(np.abs([vmin, vmax]))
        vmin = -vmax

    # Plot the surface map
    nii = nib.Nifti1Image(data_map, mask.affine, mask.header)
    plot_glass_brain(
        nii,
        output_file=None,
        display_mode='z',
        colorbar=False,
        figure=figure,
        axes=axis,
        cmap=plt.cm.Spectral_r,
        alpha=0.9,
        vmin=vmin,
        vmax=vmax,
        plot_abs=False,
        symmetric_cbar=symmetric_cbar,
    )

    return None

def plot_null_distribution(null_dist, thresh, filename):
    """Plots a null distribution of a permutation test and its threhsold.

    Parameters
    ----------
    null dist : np.ndarray
        Null distribution of test metrics. Shape must be (n_perm,).
    thresh : float
        Threshold to test statistical significance.
    filename : str
        File name to be used when saving a figure object.
    """

    # Plot null distribution and threshold
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 4))
    ax.hist(null_dist, bins=50, histtype="step", density=True)
    ax.axvline(thresh, color="black", linestyle="--")
    ax.set_xlabel("Max t-statistics")
    ax.set_ylabel("Density")
    ax.set_title("Threshold: {:.3f}".format(thresh))
    plt.savefig(filename)
    plt.close(fig)
    
    return None

def _categorise_pvalue(pval):
    """Assigns a label indicating statistical significance that corresponds 
    to an input p-value.

    Parameters
    ----------
    pval : float
        P-value from a statistical test.

    Returns
    -------
    p_label : str
        Label representing a statistical significance.
    """ 

    thresholds = [1e-3, 0.01, 0.05]
    labels = ["***", "**", "*", "n.s."]
    ordinal_idx = np.max(np.where(np.sort(thresholds + [pval]) == pval)[0])
    # NOTE: use maximum for the case in which a p-value and threshold are identical
    p_label = labels[ordinal_idx]

    return p_label

def plot_single_grouped_violin(
        data,
        group_label,
        filename,
        xlbl=None,
        ylbl=None,
        pval=None
    ):
    """Plots a grouped violin plot for each state.

    Parameters
    ----------
    data : np.ndarray
        Input data. Shape must be (n_subjects,).
    group_label : list of str
        List containing group labels for each subject.
    filename : str
        Path for saving the figure.
    xlbl : str
        X-axis tick label. Defaults to None. If you input a string of a number,
        it will print out "State {xlbl}".
    ylbl : str
        Y-axis tick label. Defaults to None.
    pval : np.ndarray
        P-values for each violin indicating staticial differences between
        the groups. If provided, statistical significance is plotted above the
        violins. Defaults to None.
    """

    # Validation
    if not isinstance(data, np.ndarray):
        raise ValueError("Input data should be an numpy array.")

    # Build dataframe
    df = pd.DataFrame(data, columns=["Statistics"])
    df["Age"] = group_label
    df["State"] = np.ones((len(data),))

    # Plot grouped split violins
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3.7, 4))
    sns.set_theme(style="white")
    vp = sns.violinplot(data=df, x="State", y="Statistics", hue="Age",
                        split=True, inner="box", linewidth=1,
                        palette={"Young": "b", "Old": "r"}, ax=ax)
    if pval is not None:
        vmin, vmax = [], []
        for collection in vp.collections:
            if isinstance(collection, matplotlib.collections.PolyCollection):
                vmin.append(np.min(collection.get_paths()[0].vertices[:, 1]))
                vmax.append(np.max(collection.get_paths()[0].vertices[:, 1]))
        vmin = np.min(np.array(vmin))
        vmax = np.max(np.array(vmax))
        ht = (vmax - vmin) * 0.045
        p_lbl = _categorise_pvalue(pval)
        if p_lbl != "n.s.":
            ax.text(
                vp.get_xticks(),
                vmax + ht,
                p_lbl, 
                ha="center", va="center", color="k", 
                fontsize=20, fontweight="bold"
            )
    sns.despine(fig=fig, ax=ax) # get rid of top and right axes
    ax.set_ylim([ax.get_ylim()[0], ax.get_ylim()[1] + np.max(vmax - vmin) * 0.05])
    if xlbl is not None:
        ax.set_xlabel(f"State {xlbl}", fontsize=22)
    else: ax.set_xlabel("")
    ax.set_ylabel(ylbl, fontsize=22)
    ax.set_xticks([])
    ax.tick_params(labelsize=22)
    ax.get_legend().remove()
    plt.tight_layout()
    fig.savefig(filename)
    plt.close(fig)

    return None

def plot_state_spectra_group_diff(
        f,
        psd,
        subject_ids,
        group_assignments,
        modality,
        bonferroni_ntest,
        filename
    ):
    """Plots state-specific PSDs and their between-group statistical differences.
    This function tests statistical differences using a cluster permutation test 
    on the frequency axis.

    Parameters
    ----------
    f : np.ndarray
        Frequencies of the power spectra and coherences. Shape is (n_freqs,).
    psd : np.ndarray
        Power spectra for each subject and state/mode. Shape must be (n_subjects,
        n_states, n_channels, n_freqs).
    subject_ids : list of str
        Subject IDs corresponding to the input data.
    group_assignments : np.ndarray
        1D array containing group labels for input subjects. A value of 1 indicates
        Group 1 (old) and a value of 2 indicates Group 2 (young).
    modality : str
        Type of data modality. Should be either "eeg" or "meg".
    bonferroni_ntest : int
        Number of tests to use for Bonferroni correction. If None, Bonferroni
        correction will not take place.
    filename : str
        Path for saving the figure.
    test_type : str
        Type of the cluster permutation test function to use. Should be "mne"
        or "glmtools" (default).
    """

    # Number of states
    n_states = psd.shape[1]
    n_columns = n_states // 2

    # Get group-averaged PSDs
    gpsd_old = np.mean(psd[group_assignments == 1], axis=0)
    gpsd_young = np.mean(psd[group_assignments == 2], axis=0)
    # dim: (n_states, n_channels, n_freqs)

    # Build a colormap
    qcmap = plt.rcParams["axes.prop_cycle"].by_key()["color"] # qualitative

    # Plot state-specific PSDs and their statistical difference
    fig, ax = plt.subplots(nrows=2, ncols=n_columns, figsize=(6.5 * n_columns, 10))
    k, j = 0, 0 # subplot indices
    for n in range(n_states):
        print(f"Plotting State {n + 1}")
        
        # Set the row index
        if (n % n_columns == 0) and (n != 0):
            k += 1
        
        # Fit GLM on state-specific, parcel-averaged PSDs
        ppsd = np.mean(psd, axis=2)
        # dim: (n_subjects, n_states, n_freqs)
        psd_model, psd_design, psd_data = fit_glm(
            ppsd[:, n, :],
            subject_ids,
            group_assignments,
            modality=modality,
            dimension_labels=["Subjects", "Frequency"],
        )

        # Perform cluster permutation tests on state-specific PSDs
        t_obs, clu_idx = cluster_perm_test(
            psd_model,
            psd_data,
            psd_design,
            pooled_dims=(1,),
            contrast_idx=0,
            n_perm=5000,
            metric="tstats",
            bonferroni_ntest=bonferroni_ntest,
        )
        n_clusters = len(clu_idx)
        t_obs = t_obs[0] # select the first contrast

        # Average group-level PSDs over the parcels
        po = np.mean(gpsd_old[n], axis=0)
        py = np.mean(gpsd_young[n], axis=0)
        eo = np.std(gpsd_old[n], axis=0) / np.sqrt(gpsd_old.shape[0])
        ey = np.std(gpsd_young[n], axis=0) / np.sqrt(gpsd_young.shape[0])

        # Plot mode-specific group-level PSDs
        ax[k, j].plot(f, py, c=qcmap[n], label="Young")
        ax[k, j].plot(f, po, c=qcmap[n], label="Old", linestyle="--")
        ax[k, j].fill_between(f, py - ey, py + ey, color=qcmap[n], alpha=0.1)
        ax[k, j].fill_between(f, po - eo, po + eo, color=qcmap[n], alpha=0.1)
        if n_clusters > 0:
            for c in range(n_clusters):
                ax[k, j].axvspan(f[clu_idx[c]][0], f[clu_idx[c]][-1], facecolor='tab:red', alpha=0.1)

        # Set labels
        ax[k, j].set_xlabel("Frequency (Hz)", fontsize=18)
        if j == 0:
            ax[k, j].set_ylabel("PSD (a.u.)", fontsize=18)
        ax[k, j].set_title(f"State {n + 1}", fontsize=18)
        ax[k, j].ticklabel_format(style="scientific", axis="y", scilimits=(-2, 6))
        ax[k, j].tick_params(labelsize=18)
        ax[k, j].yaxis.offsetText.set_fontsize(18)

        # Plot observed statistics
        end_pt = np.mean([py[int(len(py) // 3):], py[int(len(py) // 3):]])
        criteria = np.mean([ax[k, j].get_ylim()[0], ax[k, j].get_ylim()[1]])
        if end_pt >= criteria:
            inset_bbox = (0, -0.22, 1, 1)
        if end_pt < criteria:
            inset_bbox = (0, 0.28, 1, 1)
        ax_inset = inset_axes(ax[k, j], width='40%', height='30%', 
                            loc='center right', bbox_to_anchor=inset_bbox,
                            bbox_transform=ax[k, j].transAxes)
        ax_inset.plot(f, t_obs, color='k', lw=2) # plot t-spectra
        for c in range(len(clu_idx)):
            ax_inset.axvspan(f[clu_idx[c]][0], f[clu_idx[c]][-1], facecolor='tab:red', alpha=0.1)
        ax_inset.set(
            xticks=np.arange(0, max(f), 20),
            ylabel="t-stats",
        )
        ax_inset.set_ylabel('t-stats', fontsize=16)
        ax_inset.tick_params(labelsize=16)

        # Set the column index
        j += 1
        if (j % n_columns == 0) and (j != 0):
            j = 0

    plt.subplots_adjust(hspace=0.5)
    fig.savefig(filename)
    plt.close(fig)

    return None

def plot_grouped_bars(
        dataframe,
        colors,
        filename,
        yline=None,
        add_hatch=False,
        return_objects=False
    ):
    """Plots a grouped bar graph. This function is customised specifically for 
    the predictive accuracy values.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        A dataframe containing data from classification tasks.
    colors : list of str or str
        A color palette to use as a list. If a string "transparent" is given,
        bar graphs will have transparent face colours.
    filename : str
        Path for saving the figure.
    yline : float
        Constant y-value to be plotted. Can be used to mark a random chance value.
        Defaults to None.
    add_hatch : bool
        Whether to add hatched patterns to bars. Default to False.
    return_objects : bool
        Whether to return figure and axes objects. Deafult to False.
        If set to True, filename will be ignored, and the figure won't be saved.

    Returns
    -------
    fig : matplotlib.figure
        The figure object.
    ax : matplotlib.axes
        The axes object.
    """

    # Get sample size
    sample_size = int(
        len(dataframe) / (dataframe["feature"].nunique() * dataframe["type"].nunique())
    )

    # Set colors
    if colors == "transparent":
        palette = None
        ec = "#696969"
        lw = 1.5
    else:
        palette = sns.color_palette(colors)
        ec, lw = None, None

    # Plot grouped bar plots
    fig, ax = plt.subplots(nrows=1, ncols=1)
    sns.barplot(
        data=dataframe,
        x="feature", y="value", hue="type",
        width=0.8, gap=0.1,
        errorbar="sd", capsize=0.2,
        err_kws={"linewidth": 2},
        palette=palette,
        edgecolor=ec, linewidth=lw,
        ax=ax, legend=False,
    )

    # Apply facecolor separately to each patch if colors are transparent
    if colors == "transparent":
        for patch in ax.patches:
            patch.set_facecolor((0, 0, 0, 0)) # make transparent bars
    # NOTE: Setting facecolor should be done separately or else it will 
    #       interfere with setting the colour palette.

    # Add hatch patterns
    if add_hatch:
        hatches = cycle(["", "//"])
        num_category = dataframe["feature"].nunique()
        for i, patch in enumerate(ax.patches):
            if i % num_category == 0: # in grouped bar plots, patches loop over categories
                hatch = next(hatches)
            patch.set_hatch(hatch)

    # Plot a random chance value
    if yline is not None:
        ax.axhline(y=yline, color="k", alpha=0.5, linewidth=2, linestyle="--")

    # Adjust axes settings
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Static", "Dynamic", "Combined"])
    ax.set_xlabel("Features", fontsize=12)
    ax.set_ylabel(f"Predictive Accuracy (n={sample_size})", fontsize=12)
    ax.set_ylim([0, 1])
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_linewidth(1.5)
    ax.tick_params(width=1.5, labelsize=12)

    if return_objects:
        return fig, ax
    else:
        # Save figures
        fig.savefig(filename)
        return None

class StaticVisualizer():
    """Class for visualizing static network features"""
    def __init__(self):
        self.mask_file = "MNI152_T1_8mm_brain.nii.gz"
        self.parcellation_file = "fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz"

    def plot_power_map(
            self,
            power_map,
            filename,
            fontsize=20,
            plot_kwargs=None,
        ):
        """Plots a subject-averaged power map on the brain surface.

        Parameters
        ----------
        power_map : np.ndarray
            Subject-averaged power map. Shape must be (n_channels,).
        filename : str
            File name to be used when saving a figure object.
        fontsize : int
            Fontsize for a power map colorbar. Defaults to 20.
        plot_kwargs : dict
            Keyword arguments to pass to `nilearn.plotting.plot_img_on_surf`.
            Defaults to None.
        """

        # Plot surface power map
        figures, axes = power.save(
            power_map=power_map,
            mask_file=self.mask_file,
            parcellation_file=self.parcellation_file,
            plot_kwargs=plot_kwargs,
        )
        fig = figures[0]
        fig.set_size_inches(5, 6)
        cb_ax = axes[0][-1] # colorbar axis
        pos = cb_ax.get_position()
        new_pos = [pos.x0 * 0.92, pos.y0 + 0.02, pos.width * 1.20, pos.height * 1.10]
        cb_ax.set_position(new_pos)

        # Set colorbar styles
        _format_colorbar_ticks(cb_ax)
        cb_ax.ticklabel_format(style='scientific', axis='x', scilimits=(-2, 4))
        cb_ax.tick_params(labelsize=fontsize)
        cb_ax.xaxis.offsetText.set_fontsize(fontsize)

        # Save figure
        fig.savefig(filename, bbox_inches="tight", transparent=True)
        plt.close(fig)

        return None

    def plot_aec_conn_map(
            self,
            connectivity_map,
            filename,
            colormap="red_transparent_full_alpha_range",
            plot_kwargs={},
        ):
        """Plots a subject-averaged AEC map on the brain graph network.

        Parameters
        ----------
        connectivity_map : np.ndarray
            Subject-averaged connectivity map. Shape must be (n_channels, n_channels).
        filename : str
            File name to be used when saving a figure object.
        colormap : str
            Type of a colormap to use. Defaults to "red_transparent_full_alpha_range".
            If you put any colormap registered in matplotlib appended by "_transparent",
            it will add transparency that changes from low to high opacity.
        plot_kwargs : dict
            Keyword arguments to pass to `nilearn.plotting.plot_connectome`.
        """

        # Validation
        if connectivity_map.ndim != 2:
            raise ValueError("connectivity_map should be a 2-D array.")
        if colormap.split("_")[-1] == "transparent":
            colormap = _colormap_transparent(colormap.split("_")[0])

        # Plot AEC connectivity network
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 3))
        plot_kwargs.update({"edge_cmap": colormap, "figure": fig, "axes": ax})
        connectivity.save(
            connectivity_map=connectivity_map,
            parcellation_file=self.parcellation_file,
            plot_kwargs=plot_kwargs,
        )
        cb_ax = fig.get_axes()[-1]
        pos = cb_ax.get_position()
        new_pos = [pos.x0 * 1.03, pos.y0, pos.width, pos.height]
        cb_ax.set_position(new_pos)
        cb_ax.tick_params(labelsize=18)

        # Save figure
        fig.savefig(filename, transparent=True)
        plt.close(fig)

        return None
    
    def plot_psd(
            self,
            freqs,
            psd,
            error,
            filename,
        ):
        """Plots a subject-averaged PSD.

        Parameters
        ----------
        freqs : np.ndarray
            Array of frequencies in the frequency axis. Shape is (n_freqs,).
        psd : np.ndarray
            Subject-averaged PSD. Shape must be (n_freqs,).
        error : np.ndarray
            Standard error of the mean of subject-averaged PSD. Shape must be (n_freqs,).
        filename : str
            File name to be used when saving a figure object.
        """

        # Validation
        if (psd.ndim != 1) or (error.ndim != 1):
            raise ValueError("psd and error should be a 1-D array of the same shape.")
        
        # Plot PSD and its standard error of the mean
        fig ,ax = plt.subplots(nrows=1, ncols=1, figsize=(7.5, 3.5))
        ax.plot(freqs, psd, color="k", lw=2)
        ax.fill_between(freqs, psd - error, psd + error, facecolor="k", alpha=0.25)
        ax.set_xlabel("Frequency (Hz)", fontsize=18)
        ax.set_ylabel("PSD (a.u.)", fontsize=18)

        # Configure figure settings
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["bottom", "left"]].set_linewidth(2)
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        ax.tick_params(width=2, labelsize=18)
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(-3, 6))
        ax.yaxis.offsetText.set_fontsize(18)

        # Save figure
        plt.tight_layout()
        fig.savefig(filename, transparent=True)
        plt.close(fig)

        return None
    
class DynamicVisualizer():
    """Class for visualizing dynamic network features"""
    def __init__(self):
        self.mask_file = "MNI152_T1_8mm_brain.nii.gz"
        self.parcellation_file = "fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz"

    def plot_power_map(
            self,
            power_map,
            filename,
            subtract_mean=False,
            mean_weights=None,
            colormap=None,
            fontsize=20,
            plot_kwargs={},
        ):
        """Plots state-specific power map(s). Wrapper for `osl_dynamics.analysis.power.save().

        Parameters
        ----------
        power_map : np.ndarray
            Power map to save. Can be of shape: (n_components, n_modes, n_channels),
            (n_modes, n_channels) or (n_channels,). A (..., n_channels, n_channels)
            array can also be passed. Warning: this function cannot be used if n_modes
            is equal to n_channels.
        filename : str
            File name to be used when saving a figure object.
        subtract_mean : bool
            Should we subtract the mean power across modes? Defaults to False.
        mean_weights : np.ndarray
            Numpy array with weightings for each mode to use to calculate the mean.
            Default is equal weighting.
        colormap : str
            Colors for connectivity edges. If None, a default colormap is used 
            ("cold_hot").
        fontsize : int
            Fontsize for a power map colorbar. Defaults to 20.
        plot_kwargs : dict
            Keyword arguments to pass to `nilearn.plotting.plot_img_on_surf`.
        """

        # Set visualization parameters
        if colormap is None:
            colormap = "cold_hot"

        # Plot surface power maps
        plot_kwargs.update({"cmap": colormap})
        figures, axes = power.save(
            power_map=power_map,
            mask_file=self.mask_file,
            parcellation_file=self.parcellation_file,
            subtract_mean=subtract_mean,
            mean_weights=mean_weights,
            plot_kwargs=plot_kwargs,
        )
        for i, fig in enumerate(figures):
            # Reset figure size
            fig.set_size_inches(5, 6)
            
            # Change colorbar position
            cb_ax = axes[i][-1]
            pos = cb_ax.get_position()
            new_pos = [pos.x0 * 0.92, pos.y0 + 0.02, pos.width * 1.20, pos.height * 1.10]
            cb_ax.set_position(new_pos)
            
            # Set colorbar styles
            _format_colorbar_ticks(cb_ax)
            cb_ax.ticklabel_format(style='scientific', axis='x', scilimits=(-2, 6))
            cb_ax.tick_params(labelsize=fontsize)
            cb_ax.xaxis.offsetText.set_fontsize(fontsize)
            if len(figures) > 1:
                fig.savefig(
                    filename.replace(filename.split('.')[0], filename.split('.')[0] + f"_{i}"),
                    bbox_inches="tight",
                    transparent=True,
                )
            else:
                fig.savefig(filename, bbox_inches="tight", transparent=True)
            plt.close(fig)

        return None

    def plot_coh_conn_map(
            self,
            connectivity_map,
            filename,
            colormap="bwr",
            plot_kwargs={},
        ):
        """Plots state-specific connectivity map(s). Wrapper for `osl_dynamics.analysis.connectivity.save()`.

        Parameters
        ----------
        connectivity_map : np.ndarray
            Matrices containing connectivity strengths to plot. Shape must be 
            (n_modes, n_channels, n_channels) or (n_channels, n_channels).
        filename : str
            File name to be used when saving a figure object.
        colormap : str
            Type of a colormap to use for connectivity edges. Defaults to "bwr".
        plot_kwargs : dict
            Keyword arguments to pass to `nilearn.plotting.plot_connectome`.
        """

        # Validation
        if connectivity_map.ndim == 2:
            connectivity_map = connectivity_map[np.newaxis, ...]

        # Number of states/modes
        n_states = connectivity_map.shape[0]

        # Plot connectivity maps
        for n in range(n_states):
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 3))
            temp_kwargs = {"edge_cmap": colormap, "figure": fig, "axes": ax}
            connectivity.save(
                connectivity_map=connectivity_map[n, :],
                parcellation_file=self.parcellation_file,
                plot_kwargs={**plot_kwargs, **temp_kwargs},
            )
            cb_ax = fig.get_axes()[-1]
            pos = cb_ax.get_position()
            new_pos = [pos.x0 * 1.05, pos.y0, pos.width, pos.height]
            cb_ax.set_position(new_pos)
            cb_ax.tick_params(labelsize=20)
            if n_states != 1:
                fig.savefig(
                    filename.replace(filename.split('.')[0], filename.split('.')[0] + f"_{n}"),
                    transparent=True
                )
            else:
                fig.savefig(filename, transparent=True)
            plt.close(fig)

        return None
    
    def plot_psd(
            self,
            freqs,
            psd,
            error,
            filename,
            fontsize=22,
        ):
        """Plots state-specific subject-averaged PSD(s).

        Parameters
        ----------
        freqs : np.ndarray
            Array of frequencies in the frequency axis. Shape is (n_freqs,).
        psd : np.ndarray
            State-specific subject-averaged PSD. Shape must be (n_states, n_freqs).
        error : np.ndarray
            Standard error of the mean of state-specific subject-averaged PSD.
            Shape must be (n_states, n_freqs).
        filename : str
            File name to be used when saving a figure object.
        fontsize : int
            Fontsize for axes ticks and labels. Defaults to 22.
        """

        # Validation
        if (psd.ndim != 2) or (error.ndim != 2):
            raise ValueError("psd and error should be a 2-D array of the same shape.")
        
        # Number of states
        n_states = psd.shape[0]

        # Plot PSD and its standard error of the mean for each state
        hmin, hmax = 0, np.ceil(freqs[-1])

        for n in range(n_states):
            fig, ax = plotting.plot_line(
                [freqs],
                [psd[n]],
                errors=[[psd[n, :] - error[n, :]], [psd[n, :] + error[n, :]]],
            )
            ax.set_xlabel("Frequency (Hz)", fontsize=fontsize)
            ax.set_ylabel("PSD (a.u.)", fontsize=fontsize)

            # Reset figure size
            fig.set_size_inches(6, 4)
            
            # Set axis styles
            ax.set_xticks(np.arange(hmin, hmax, 10))
            ax.ticklabel_format(style="scientific", axis="y", scilimits=(-2, 6))
            ax.tick_params(labelsize=fontsize)
            ax.yaxis.offsetText.set_fontsize(fontsize)

            # Save figure
            if n_states != 1:
                fig.savefig(
                    filename.replace(filename.split('.')[0], filename.split('.')[0] + f"_{n}"),
                    bbox_inches="tight",
                    transparent=True,
                )
            else:
                fig.savefig(filename, bbox_inches="tight", transparent=True)
            plt.close(fig)

        return None

class GroupDifferencePSD():
    """Class for visualizing group-level spectral differences"""
    def __init__(self, freqs, gpsd1, gpsd2, data_space, modality):
        # Organize input parameters
        self.freqs = freqs
        self.gpsd1 = gpsd1
        self.gpsd2 = gpsd2
        self.data_space = data_space
        self.modality = modality

        # Get file paths to parcellation data
        self.mask_file = "MNI152_T1_8mm_brain.nii.gz"
        self.parcellation_file = (
            "fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz"
        )

    def _get_minmax(self, data):
        """Get minimum and maximum values of a data array.

        Parameters
        ----------
        data : np.ndarray
            Input data array.
    
        Returns
        -------
        minimum : float
            A minimum value
        maximum : float
            A maximum value
        """

        minimum, maximum = data.min(), data.max()
        return minimum, maximum

    def prepare_data(self):
        """Compute group-level PSD differences and get sensor/parcel information
        """

        # Compute group-level PSD differences
        gpsd_diff = self.gpsd2 - self.gpsd1 # Group 2 - Group 1

        # Get ROI positions
        if self.data_space == "source":
            # Get the center of each parcel
            parcellation = Parcellation(self.parcellation_file)
            roi_centers = parcellation.roi_centers()
        if self.data_space == "sensor":
            # Get sensor positions from an example subject
            eeg_flag, meg_flag = False, False
            if self.modality == "eeg":
                raw = mne.io.read_raw_fif("/ohba/pi/mwoolrich/scho/NTAD/preproc/eeg/P1058_resting_close_bl_raw_tsss/P1058_resting_close_bl_tsss_preproc_raw.fif")
                # Select common sensor locations (to account for different EEG layouts)
                if self.data_space == "sensor":
                    eeg_ch_names = np.array(raw.info["ch_names"])[mne.pick_types(raw.info, eeg=True)]
                    with open("/home/scho/AnalyzeNTAD/results/data/common_eeg_sensor.pkl", "rb") as input_path:
                        common_eeg_idx = pickle.load(input_path)
                    input_path.close()
                    eeg_ch_names = eeg_ch_names[common_eeg_idx["EasyCap70"]]
                    raw = raw.pick_channels(eeg_ch_names, ordered=True, verbose=None)
                eeg_flag = True
            if self.modality == "meg":
                raw = mne.io.read_raw_fif("/ohba/pi/mwoolrich/scho/NTAD/preproc/meg/P1007_resting_close_bl_raw_tsss/P1007_resting_close_bl_tsss_preproc_raw.fif")
                meg_flag = True
             # Get the position of each channel
            roi_centers = raw._get_channel_positions()
            picks = mne.pick_types(raw.info, eeg=eeg_flag, meg=meg_flag)
            mag_picks = mne.pick_types(raw.info, meg='mag')
            # Re-order ROI positions to use colour to indicate anterior -> posterior location (for sensors)
            # For source space, this is done within `plotting.plot_psd_topo()`.
            order = np.argsort(roi_centers[:, 1])
            roi_centers = roi_centers[order]
            picks = picks[order]
            gpsd_diff = gpsd_diff[order, :]
            if self.modality == "meg":
                # Re-order ROI positions of magnetometers
                roi_centers_mag = raw._get_channel_positions()[mag_picks]
                gpsd_diff_mag = (self.gpsd2 - self.gpsd1)[mag_picks] # select PSDs for magnetometer channels
                # NOTE: We only use magnetometer for MEG sensor data when plotting topographical map (visualisation purpose).
                #       MEG CamCAN used only orthogonal planar gradiometers (i.e., no axial gradiometers)
                # Repeat specifically for magnetometers
                mag_order = np.argsort(roi_centers_mag[:, 1])
                roi_ceneters_mag = roi_centers_mag[mag_order]
                mag_picks = mag_picks[mag_order]
                gpsd_diff_mag = gpsd_diff_mag[mag_order, :]

        # Compute first and second moments of subject-averaged PSD differences (over parcels/channels)
        gpsd_diff_avg = np.mean(gpsd_diff, axis=0)
        gpsd_diff_sde = np.std(gpsd_diff, axis=0) / np.sqrt(gpsd_diff.shape[0])
        # dim: (n_parcels, n_freqs) -> (n_freqs,)

        # Assign results to the class object
        self.roi_centers = roi_centers
        self.gpsd_diff = gpsd_diff
        self.gpsd_diff_avg = gpsd_diff_avg
        self.gpsd_diff_sde = gpsd_diff_sde
        if self.data_space == "sensor":
            self.raw = raw.copy()
            self.picks = picks
            if self.modality == "meg":
                self.mag_picks = mag_picks
                self.gpsd_diff_mag = gpsd_diff_mag
                
        return None
    
    def plot_psd_diff(self, clusters, group_lbls, save_dir, plot_legend=False):
        """Plots group-level PSD differences.

        Parameters
        ----------
        clusters : list of np.ndarray
            List of an array indicating clustered frequencies with age effects.
        group_lbls : list of str
            Labels of groups to compare.
        save_dir : str
            Directory where a figure object will be saved.
        plot_legend : bool
            Whether to plot a legend. Defaults to False.
        """

        # Set visualization parameters
        matplotlib.rc('xtick', labelsize=14) 
        matplotlib.rc('ytick', labelsize=14)

        # Compute inputs and frequencies to draw topomaps (alpha)
        alpha_range = np.where(np.logical_and(self.freqs >= 8, self.freqs <= 13))
        gpsd_diff_alpha = self.gpsd_diff_avg[alpha_range]
        topo_freq_top = sorted([
            self.freqs[self.gpsd_diff_avg == max(gpsd_diff_alpha)],
            self.freqs[self.gpsd_diff_avg == min(gpsd_diff_alpha)],
        ])
        gpsd_data = [
            np.squeeze(np.array(self.gpsd_diff[:, self.freqs == topo_freq_top[i]]))
            for i in range(len(topo_freq_top))
        ]
        if self.data_space == "sensor" and self.modality == "meg":
            gpsd_data_mag = [
                np.squeeze(np.array(self.gpsd_diff_mag[:, self.freqs == topo_freq_top[i]]))
                for i in range(len(topo_freq_top))
            ]

        # Compute inputs and frequencies to draw topomaps (low frequency, beta)
        low_beta_range = [[1.5, 8], [13, 20]]
        topo_data = [
            power.variance_from_spectra(self.freqs, self.gpsd_diff, frequency_range=low_beta_range[0]),
            power.variance_from_spectra(self.freqs, self.gpsd_diff, frequency_range=low_beta_range[1]),
        ] # dim: (n_band, n_parcels)
        topo_freq_bottom = [
            self.freqs[np.where(np.logical_and(self.freqs >= 1.5, self.freqs <= 8))].mean(),
            self.freqs[np.where(np.logical_and(self.freqs >= 13, self.freqs <= 20))].mean(),
        ]
        if self.data_space == "sensor" and self.modality == "meg":
            topo_data_mag = [
                power.variance_from_spectra(self.freqs, self.gpsd_diff_mag, frequency_range=low_beta_range[0]),
                power.variance_from_spectra(self.freqs, self.gpsd_diff_mag, frequency_range=low_beta_range[1]),
            ]
        
        # Get maximum and minimum values for topomaps
        if self.data_space == "sensor" and self.modality == "meg":
            vmin_top, vmax_top = self._get_minmax(self.gpsd_diff_mag[:, alpha_range])
            vmin_bottom, vmax_bottom = np.min(topo_data_mag), np.max(topo_data_mag)
        else:
            vmin_top, vmax_top = self._get_minmax(self.gpsd_diff[:, alpha_range])
            vmin_bottom, vmax_bottom = np.min(topo_data), np.max(topo_data)

        # Visualize
        if self.data_space == "source":
            # Start a figure object
            fig, ax = plotting.plot_psd_topo(
                self.freqs,
                self.gpsd_diff,
                parcellation_file=self.parcellation_file,
                topomap_pos=[0.45, 0.37, 0.5, 0.55],
            )
            # Shrink axes to make space for topos
            fig.set_size_inches(7, 9)
            ax_pos = ax.get_position()
            ax.set_position([ax_pos.x0, ax_pos.y0 * 1.7, ax_pos.width, ax_pos.height * 0.65])
            # Plot parcel-averaged PSD differences
            max_zorder = max(line.get_zorder() for line in ax.lines)
            ax.plot(self.freqs, self.gpsd_diff_avg, lw=3, linestyle="--", color="tab:red", alpha=0.8, zorder=max_zorder + 2)
            ax.fill_between(self.freqs, self.gpsd_diff_avg - self.gpsd_diff_sde, self.gpsd_diff_avg + self.gpsd_diff_sde, color="tab:red", alpha=0.3, zorder=max_zorder + 1)
            # Plot topomaps for the alpha band
            topo_centers = np.linspace(0, 1, len(topo_freq_top) + 2)[1:-1]
            for i in range(len(topo_freq_top)):
                topo_pos = [topo_centers[i] - 0.2, 1.1, 0.25, 0.25]
                topo_ax = ax.inset_axes(topo_pos)
                # Plot parcel topographical map
                plot_surfaces(
                    gpsd_data[i],
                    self.mask_file,
                    self.parcellation_file,
                    vmin=-vmin_top,
                    vmax=vmax_top,
                    figure=fig,
                    axis=topo_ax,
                )
                # Connect frequencies to topographical map
                xy = (float(topo_freq_top[i]), ax.get_ylim()[1])
                con = matplotlib.patches.ConnectionPatch(xyA=xy, xyB=(np.mean(topo_ax.get_xlim()), topo_ax.get_ylim()[0]),
                                                        coordsA=ax.transData, coordsB=topo_ax.transData,
                                                        axesA=ax, axesB=topo_ax, color="tab:gray", lw=2)
                ax.add_artist(con)
                ax.axvline(x=topo_freq_top[i], color="tab:gray", lw=2)
            # Plot topomaps for the low frequency and beta band
            topo_centers = np.linspace(0, 1, len(topo_freq_bottom) + 2)[1:-1]
            for i in range(len(topo_freq_bottom)):
                topo_pos = [topo_centers[i] - 0.2, -0.4, 0.25, 0.25]
                topo_ax = ax.inset_axes(topo_pos)
                # Plot parcel topographical map
                plot_surfaces(
                    topo_data[i],
                    self.mask_file,
                    self.parcellation_file,
                    vmin=-vmin_bottom,
                    vmax=vmax_bottom,
                    figure=fig,
                    axis=topo_ax,
                )
                # Connect frequencies to topographical map
                xy = (float(topo_freq_bottom[i]), ax.get_ylim()[0])
                con = matplotlib.patches.ConnectionPatch(xyA=xy, xyB=(np.mean(topo_ax.get_xlim()), topo_ax.get_ylim()[0]),
                                                        coordsA=ax.transData, coordsB=topo_ax.transData,
                                                        axesA=ax, axesB=topo_ax, color="k", alpha=0.3, lw=2)
                ax.add_artist(con)
                ax.axvspan(low_beta_range[i][0], low_beta_range[i][1], facecolor="k", alpha=0.1, zorder=0)
        elif self.data_space == "sensor":
            # Start a figure object
            fig, ax = plt.subplots(nrows=1, ncols=1)
            cmap = plt.cm.viridis_r
            n_channels = self.gpsd_diff.shape[0]
            if self.modality == "eeg":
                for i in reversed(range(n_channels)):
                    ax.plot(self.freqs, self.gpsd_diff[i], c=cmap(i / n_channels))
            if self.modality == "meg":
                n_locations = n_channels / 3
                k = n_locations - 1
                for i in reversed(range(n_channels)):
                    ax.plot(self.freqs, self.gpsd_diff[i], c=cmap(k / n_locations))
                    if i % 3 == 0: k -= 1
            # Plot channel topomap
            inside_ax = ax.inset_axes([0.65, 0.62, 0.30, 0.35])
            chs = [self.raw.info["chs"][pick] for pick in self.picks]
            ch_names = np.array([ch["ch_name"] for ch in chs])
            bads = [idx for idx, name in enumerate(ch_names) if name in self.raw.info["bads"]]
            colors = [cmap(i / n_channels) for i in range(n_channels)]
            mne.viz.utils._plot_sensors(self.roi_centers, self.raw.info, self.picks, colors, bads, ch_names, title=None, show_names=False,
                                        ax=inside_ax, show=False, kind="topomap", block=False, to_sphere=True, sphere=None, pointsize=25, linewidth=0.5)
            # Plot parcel-averaged PSD difference
            max_zorder = max(line.get_zorder() for line in ax.lines)
            ax.plot(self.freqs, self.gpsd_diff_avg, lw=3, linestyle="--", color="tab:red", alpha=0.8, zorder=max_zorder + 2)
            ax.fill_between(self.freqs, self.gpsd_diff_avg - self.gpsd_diff_sde, self.gpsd_diff_avg + self.gpsd_diff_sde, color="tab:red", alpha=0.3, zorder=max_zorder + 1)
            # Plot topomaps for the alpha band
            topo_centers = np.linspace(0, 1, len(topo_freq_top) + 2)[1:-1]
            vtop = np.max(np.abs([vmin_top, vmax_top]))
            cnorm_top = CenteredNorm(vcenter=0, halfrange=vtop)
            if self.modality == "eeg":
                topo_raw = self.raw.copy().pick_types(eeg=True, meg=False).reorder_channels(ch_names)
                for i in range(len(topo_freq_top)):
                    topo_pos = [topo_centers[i] - 0.2, 1.1, 0.25, 0.25]
                    topo_ax = ax.inset_axes(topo_pos)
                    mne.viz.plot_topomap(gpsd_data[i], topo_raw.info, axes=topo_ax, cmap="Spectral_r", cnorm=cnorm_top, show=False)
                    # Connect frequencies to topographical map
                    xy = (float(topo_freq_top[i]), ax.get_ylim()[1])
                    con = matplotlib.patches.ConnectionPatch(xyA=xy, xyB=(np.mean(topo_ax.get_xlim()), topo_ax.get_ylim()[0]),
                                                            coordsA=ax.transData, coordsB=topo_ax.transData,
                                                            axesA=ax, axesB=topo_ax, color="tab:gray", lw=2)
                    ax.add_artist(con)
                    ax.axvline(x=topo_freq_top[i], color="tab:gray", lw=2)
            elif self.modality == "meg":
                chs = [self.raw.info["chs"][pick] for pick in self.mag_picks]
                ch_names = np.array([ch["ch_name"] for ch in chs])
                topo_raw = self.raw.copy().pick_types(eeg=False, meg="mag").reorder_channels(ch_names)
                for i in range(len(topo_freq_top)):
                    topo_pos = [topo_centers[i] - 0.2, 1.1, 0.25, 0.25]
                    topo_ax = ax.inset_axes(topo_pos)
                    mne.viz.plot_topomap(gpsd_data_mag[i], topo_raw.info, axes=topo_ax, cmap="Spectral_r", cnorm=cnorm_top, show=False)
                    # Connect frequencies to topographical map
                    xy = (float(topo_freq_top[i]), ax.get_ylim()[1])
                    con = matplotlib.patches.ConnectionPatch(xyA=xy, xyB=(np.mean(topo_ax.get_xlim()), topo_ax.get_ylim()[0]),
                                                            coordsA=ax.transData, coordsB=topo_ax.transData,
                                                            axesA=ax, axesB=topo_ax, color="tab:gray", lw=2)
                    ax.add_artist(con)
                    ax.axvline(x=topo_freq_top[i], color="tab:gray", lw=2)
            # Plot topomaps for the low frequency and beta band
            topo_centers = np.linspace(0, 1, len(topo_freq_bottom) + 2)[1:-1]
            vbottom = np.max(np.abs([vmin_bottom, vmax_bottom]))
            cnorm_bottom = CenteredNorm(vcenter=0, halfrange=vbottom)
            if self.modality == "eeg":
                topo_raw = self.raw.copy().pick_types(eeg=True, meg=False).reorder_channels(ch_names)
                for i in range(len(topo_freq_bottom)):
                    topo_pos = [topo_centers[i] - 0.2, -0.4, 0.25, 0.25]
                    topo_ax = ax.inset_axes(topo_pos)
                    mne.viz.plot_topomap(topo_data[i], topo_raw.info, axes=topo_ax, cmap="Spectral_r", cnorm=cnorm_bottom, show=False)
                    # Connect frequencies to topographical map
                    xy = (float(topo_freq_bottom[i]), ax.get_ylim()[0])
                    con = matplotlib.patches.ConnectionPatch(xyA=xy, xyB=(np.mean(topo_ax.get_xlim()), topo_ax.get_ylim()[0]),
                                                            coordsA=ax.transData, coordsB=topo_ax.transData,
                                                            axesA=ax, axesB=topo_ax, color="k", alpha=0.3, lw=2)
                    ax.add_artist(con)
                    ax.axvspan(low_beta_range[i][0], low_beta_range[i][1], facecolor="k", alpha=0.1, zorder=0)
            elif self.modality == "meg":
                chs = [self.raw.info["chs"][pick] for pick in self.mag_picks]
                ch_names = np.array([ch["ch_name"] for ch in chs])
                topo_raw = self.raw.copy().pick_types(eeg=False, meg="mag").reorder_channels(ch_names)
                for i in range(len(topo_freq_bottom)):
                    topo_pos = [topo_centers[i] - 0.2, -0.4, 0.25, 0.25]
                    topo_ax = ax.inset_axes(topo_pos)
                    mne.viz.plot_topomap(topo_data_mag[i], topo_raw.info, axes=topo_ax, cmap="Spectral_r", cnorm=cnorm_bottom, show=False)
                    # Connect frequencies to topographical map
                    xy = (float(topo_freq_bottom[i]), ax.get_ylim()[0])
                    con = matplotlib.patches.ConnectionPatch(xyA=xy, xyB=(np.mean(topo_ax.get_xlim()), topo_ax.get_ylim()[0]),
                                                            coordsA=ax.transData, coordsB=topo_ax.transData,
                                                            axesA=ax, axesB=topo_ax, color="k", alpha=0.3, lw=2)
                    ax.add_artist(con)
                    ax.axvspan(low_beta_range[i][0], low_beta_range[i][1], facecolor="k", alpha=0.1, zorder=0)
            # Shrink axes to make space for topos
            fig.set_size_inches(7, 9)
            ax_pos = ax.get_position()
            ax.set_position([ax_pos.x0, ax_pos.y0 * 2.1, ax_pos.width, ax_pos.height * 0.65])

        ylim = ax.get_ylim()

        # Mark significant frequencies
        ymax = ax.get_ylim()[1] * 0.95
        for clu in clusters:
            if len(clu) > 1:
                ax.plot(self.freqs[clu], [ymax] * len(clu), color="tab:red", lw=5, alpha=0.7)
            else:
                ax.plot(self.freqs[clu], ymax, marker="s", markersize=12,
                        markeredgecolor="tab:red", marker_edgewidth=12,
                        markerfacecolor="tab:red", alpha=0.7)

        # Add manual colorbar for topographies at the top
        cb_ax = ax.inset_axes([0.78, 1.12, 0.03, 0.22])
        cmap = plt.cm.Spectral_r
        if self.data_space == "source":
            vtop = np.max(np.abs([vmin_top, vmax_top]))
            norm = Normalize(vmin=-vtop, vmax=vtop)
            cb = plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cb_ax, orientation="vertical")
            cb.ax.set_yticks([-vtop, 0, vtop])
        elif self.data_space == "sensor":
            cb = plt.colorbar(matplotlib.cm.ScalarMappable(norm=cnorm_top, cmap=cmap), cax=cb_ax, orientation="vertical")
            cb.ax.set_yticks([-vtop, 0, vtop])
        cb.ax.set_ylabel("PSD (a.u.)", fontsize=14)
        cb.ax.ticklabel_format(style="scientific", axis="y", scilimits=(-2, 4))
       
        # Add manual colorbar for topographies at the bottom
        cb_ax = ax.inset_axes([0.78, -0.38, 0.03, 0.22])
        cmap = plt.cm.Spectral_r
        if self.data_space == "source":
            vbottom = np.max(np.abs([vmin_bottom, vmax_bottom]))
            norm = Normalize(vmin=-vbottom, vmax=vbottom)
            cb = plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cb_ax, orientation="vertical")
            cb.ax.set_yticks([-vbottom, 0, vbottom])
        elif self.data_space == "sensor":
            cb = plt.colorbar(matplotlib.cm.ScalarMappable(norm=cnorm_bottom, cmap=cmap), cax=cb_ax, orientation="vertical")
            cb.ax.set_yticks([-vbottom, 0, vbottom])
        cb.ax.set_ylabel("Power (a.u.)", fontsize=14)
        cb.ax.ticklabel_format(style='scientific', axis='y', scilimits=(-2, 4))

        # Set labels
        ax.set_xlabel("Frequency (Hz)", fontsize=14)
        ax.set_ylabel(f"PSD $\Delta$ ({group_lbls[1]} - {group_lbls[0]}) (a.u.)", fontsize=14)
        ax.set_xlim([0, 46])
        ax.set_ylim(ylim)
        ax.ticklabel_format(style="scientific", axis="y", scilimits=(-2, 4))

        # Create manual legend
        if plot_legend:
            hl = [
                matplotlib.lines.Line2D([0], [0], color="tab:red", linestyle="--", lw=3),
                matplotlib.lines.Line2D([0], [0], color="tab:red", alpha=0.3, lw=3),
            ]
            ax.legend(hl, ["Average", "Standard Error"], loc="lower right", fontsize=12)

        # Save figure
        save_path = os.path.join(save_dir, "psd_diff.png")
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        print("Saved: ", save_path)

        return None
