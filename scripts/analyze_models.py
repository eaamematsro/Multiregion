import os
import re
import glob
import pickle
import pdb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sequential_reaching import Network
from scipy.linalg import schur
from typing import List, Union, Tuple


cwd = Path.cwd()
data_path = cwd / 'data'
save_path = Path(__file__).parent.parent.resolve() / 'figures'

def set_plt_params(
    font_size: int = 16,
    legend_size: int = 16,
    axes_label_size: int = 20,
    linewidth: float = 1.5,
    axes_title_size: int = 12,
    xtick_label_size: int = 16,
    ytick_label_size: int = 16,
    ticksize: float = 5,
    fig_title_size: int = 34,
    style: str = "fast",
    font: str = "avenir",
    file_format: str = "svg",
    fig_dpi: int = 500,
    figsize: Tuple[float, float] = (11, 8),
    auto_method: str = None,
    x_margin: float = None,
    y_margin: float = None,
    render_path: bool = False,
):
    """
    This function sets the plot parameters for a plot
    :param font_size:
    :param legend_size:
    :param axes_label_size:
    :param linewidth:
    :param axes_title_size:
    :param xtick_label_size:
    :param ytick_label_size:
    :param ticksize:
    :param fig_title_size:
    :param style:
    :param font:
    :param file_format:
    :param fig_dpi:
    :param figsize
    :return:
    """
    plt.rcParams.update(plt.rcParamsDefault)
    plt.style.use(style)
    plt.rcParams["savefig.format"] = file_format
    plt.rcParams["backend"] = file_format
    plt.rcParams["savefig.dpi"] = fig_dpi
    plt.rcParams["figure.figsize"] = figsize
    if not auto_method is None:
        plt.rcParams["axes.autolimit_mode"] = auto_method
    if not x_margin is None:
        assert (x_margin >= 0) & (x_margin <= 1)
        plt.rcParams["axes.xmargin"] = x_margin
    if not y_margin is None:
        assert (y_margin >= 0) & (y_margin <= 1)
        plt.rcParams["axes.ymargin"] = y_margin

    plt.rc("font", size=font_size)
    plt.rc("axes", titlesize=axes_title_size)
    plt.rc("axes", labelsize=axes_label_size)
    plt.rc("xtick", labelsize=xtick_label_size)
    plt.rc("ytick", labelsize=ytick_label_size)
    plt.rc("legend", fontsize=legend_size)
    plt.rc("figure", titlesize=fig_title_size)
    plt.rc("lines", linewidth=linewidth)
    plt.rcParams["font.family"] = font
    plt.rcParams["xtick.major.size"] = ticksize
    plt.rcParams["ytick.major.size"] = ticksize
    plt.rcParams["xtick.minor.size"] = ticksize / 3
    plt.rcParams["ytick.minor.size"] = ticksize / 3

    if not render_path:
        plt.rcParams["svg.fonttype"] = "none"

    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.left"] = True
    plt.rcParams["axes.spines.bottom"] = True


def make_axis_nice(
    ax: Union[plt.Axes, plt.Figure] = None,
    offset: int = 10,
    line_width: float = 0.5,
    spines: List = None,
    color: str = None,
):
    """Makes axis pretty
    This function modifies the x and y axis so that there is a vertical and horizontal gap between them
    Args:
        ax: This is the axis (axes) that need to be changed. When this argument is a fig all axes of the fig
        get modified
        offset: Size of the gap.
        line_width: Linew width of the new axes.
        spines:
        color:
    """
    if ax is None:
        ax = plt.gca()

    if spines is None:
        spines = ["left", "bottom"]
    if type(ax) == plt.Figure:
        ax_list = ax.axes
    else:
        ax_list = [ax]

    for ax in ax_list:
        for spine in spines:
            ax.spines[spine].set_linewidth(line_width)
            if color is not None:
                ax.spines[spine].set_color(color)
            ax.spines[spine].set_position(("outward", offset))



all_models = list(data_path.glob('**/model.pickle'))
results = []
for model in all_models:
    with open(model, 'rb') as handle:
        data_dict = pickle.load(handle)
    model_string = model.parent.stem
    date_string = model.parent.parent.stem
    date_save_folder = save_path / date_string
    data_save_folder = save_path / date_string / model_string
    date_save_folder.mkdir(exist_ok=True)
    data_save_folder.mkdir(exist_ok=True)
    weights = data_dict['model_state']
    network = data_dict['full_model']
    params = network.params
    # J = weights['rnn.J'].numpy()
    # Z, T = schur(J, output='real')
    network.test_sensitivity(save_path=data_save_folder)

    single_results = network.evaluate_contribution()
    results.extend(single_results)

df = pd.DataFrame(results)
r_df = df.loc[(df['Type'] == 'Single') | (df['Type'] == 'Cumulative')]
plt.close('all')

plt_params = {'data': r_df, 'x': 'Regions Removed',
                  'y': 'VAF', 'hue': 'wd'}


g = sns.catplot(**plt_params, row='Type', kind='point')
g.set(yscale='symlog', ylim=[-1e4, 1])
lw =.75 # lw of first line
for axis in g.axes.flatten():
    plt.setp(axis.lines, linewidth=lw)
file_name = save_path / 'region_removal'
plt.savefig(file_name)
fig, ax = plt.subplots(1, 4, figsize=(11, 8))
sns.pointplot(df.loc[df['Type'] == 'Single Linear'], x='Preserved Region', y='VAF', hue='wd',
              ax=ax[0], dodge=True)

plt.setp(ax[0].lines, linewidth=lw)

sns.lineplot(df.loc[df['Type'] == 'Single Linear'], x='wd', y='VAF Ratio',
             ax=ax[1])
sns.lineplot(df.loc[df['Type'] == 'Single Linear'], x='wd', y='gini',
             ax=ax[2])
sns.lineplot(df.loc[df['Type'] == 'Single Linear'], x='wd', y='Final Loss',
             ax=ax[3])
for axis in ax[1:]:
    axis.set_xscale('log')
fig.tight_layout()

file_name = save_path / 'parameter_summary'
fig.savefig(file_name)
plt.pause(.1)