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
            Fontsize for a power map colorbar. Defaults to 24.
        plot_kwargs : dict
            Keyword arguments to pass to `nilearn.plotting.plot_img_on_surf`.
            Defaults to None.
        """

        # Plot surface power map
        figures, axes  = power.save(
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