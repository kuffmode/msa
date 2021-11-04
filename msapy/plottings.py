import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from typeguard import typechecked
from typing import Dict, Optional, Any


@typechecked
def set_style(font_size: int = 10):
    """
    Just some basic things I do before plotting.
    """
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = 'GothamSSm'
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['font.size'] = font_size


@typechecked
def color_code(shapley_table: pd.DataFrame,
               significants: Optional[pd.DataFrame] = None,
               as_cmap:Optional[bool] = False):
    """
    Maps the custom color map onto the elements for plotting.
    Args:
        shapley_table:
            Raw Shapley table but sorted.

        significants:
            Significant df, the one that comes out of ut.bootstrap_hypothesis_testing.

        as_cmap:
            If it should be a cmap or not. Usually not necessary but some plots like plt.scatter need cmap.

    Returns:
        Either a seaborn color palette or a matplotlib colormap.

    """
    if type(significants) != pd.DataFrame:
        if any(shapley_table.mean()<0):
            colors = ['#006685', '#3FA5C4', '#FFE48D', '#E84653', '#BF003F']
        else:
            colors = ['#FFE48D', '#E84653', '#BF003F']
        return sns.blend_palette(colors, n_colors=len(shapley_table.columns),as_cmap=as_cmap)

    else:
        grays = {x: 'gray' for x in shapley_table.columns if x not in significants}

        if any(significants.columns[significants.mean() > 0]):
            warms_keys = significants.columns[significants.mean() > 0]
            warms_vals = sns.blend_palette(['#E84653', '#BF003F'], len(warms_keys))
            warms = dict(zip(warms_keys, warms_vals))
        else:
            warms = {}
        if any(significants.columns[significants.mean() < 0]):
            colds_keys = significants.columns[significants.mean() < 0]
            colds_vals = sns.blend_palette(['#006685', '#3FA5C4'], len(colds_keys))
            colds = dict(zip(colds_keys, colds_vals))
        else:
            colds = {}
        return {**colds, **grays, **warms}


@typechecked
def plot_shapley_ranks(shapley_table: pd.DataFrame, colors: Any, ax: plt.Axes, barplot_params: Optional[Dict] = None):
    """
    Plots a barplot with the given colors assigned to each bar. The warmer the larger the shapley values.
    Args:
        shapley_table:
            Raw Shapley table, sorted.
        colors:
            seaborn color palette from color_code function.
        ax:
            which axes to plot?
        barplot_params:
            extra parameters for the seaborn barplot. If nothing is given:
                {"ci": 95, "orient": "h", "errcolor": "k"} will be passed automatically.
    """
    barplot_params = barplot_params if barplot_params else {"ci": 95, "orient": "h", "errcolor": "k"}
    sns.barplot(data=shapley_table, palette=colors, ax=ax, **barplot_params)
    if barplot_params["orient"] == "h":
        ax.axvline(linewidth=1, color="k")

    elif barplot_params["orient"] == "v":
        ax.axhline(linewidth=1, color="k")

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
