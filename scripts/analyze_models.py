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


cwd = Path.cwd()
data_path = cwd / 'data'
save_path = Path(__file__).parent.parent.resolve() / 'figures'


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