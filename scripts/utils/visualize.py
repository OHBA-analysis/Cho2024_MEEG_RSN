"""Functions for visualizing results

"""

import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LinearSegmentedColormap
from osl_dynamics.analysis import power, connectivity
from osl_dynamics.utils import plotting

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
    ax.legend(loc="best", fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    plt.tight_layout()
    fig.savefig(filename, transparent=True)
    plt.close(fig)
    
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
        cb_ax.ticklabel_format(style='scientific', axis='x', scilimits=(-3, 4))
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