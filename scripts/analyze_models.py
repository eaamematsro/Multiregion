import os
import re
import glob
import pickle
import pdb
from pathlib import Path
from sequential_reaching import Network
from scipy.linalg import schur

cwd = Path.cwd()
data_path = cwd / 'data'

all_models = list(data_path.glob('**/model.pickle'))
for model in all_models:
    with open(model, 'rb') as handle:
        data_dict = pickle.load(handle)
    weights = data_dict['model_state']
    network = data_dict['full_model']
    params = network.params
    J = weights['rnn.J'].numpy()
    Z, T = schur(J, output='real')
    network.evaluate_contribution()
    pdb.set_trace()
