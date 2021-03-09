from config import *
from settings import *
import os
import numpy as np
import pickle
import itertools
import pandas as pd
import plotly.express as px
import torch.multiprocessing as mp
import ray

from ray.experimental.sgd.pytorch.pytorch_trainer import PyTorchTrainer
from policy_factory import *
from test_envs import *
from clause_model import *


def runtime_get_graph(adj_arrays, clause_labels, lit_labels):
  # print(np.shape(list(zip(adj_arrays["cols_arr"], adj_arrays["rows_arr"]))))
  G = dgl.heterograph(
            {('literal', 'l2c', 'clause') : list(zip(adj_arrays["cols_arr"], adj_arrays["rows_arr"])),
             ('clause', 'c2l', 'literal') : list(zip(adj_arrays["rows_arr"], adj_arrays["cols_arr"]))},
            {'literal': len(lit_labels),
             'clause': len(clause_labels)})

  G.nodes['literal'].data['lit_labels'] = lit_labels
  G.nodes['clause'].data['clause_ids']  = clause_labels[:, 0]
  G.nodes['clause'].data['clause_labels']  = clause_labels[:, 1:-2]

  # Cap activity
  G.nodes['literal'].data['lit_labels'][:,2].tanh_()
  G.nodes['clause'].data['clause_labels'][:,3].tanh_()
  # Mask lbd
  G.nodes['clause'].data['clause_labels'][:,2] = 0

  return G


def get_settings_from_file(fname):
	settings = CnfSettings(cfg())
	conf = load_config_from_file(fname)
	for (k,v) in conf.items():
		settings.hyperparameters[k]=v
	return settings

def wrap_sat_model(trainer_checkpoint, exp_filename):
	settings = get_settings_from_file(exp_filename)
	all_data = torch.load(trainer_checkpoint)
	state = all_data['models'][0]
	model = ClausePredictionModel(settings,prediction=False)
	model.load_state_dict(state)
	model.eval()
	def f(adj_arrays, cl_label_arr, lit_label_arr, gss):
		exp_gss = gss.expand(len(cl_label_arr),settings['state_dim'])
		G = runtime_get_graph(adj_arrays, cl_label_arr, lit_label_arr)
		logits = model({'gss': exp_gss, 'graph': G})
		rc = torch.softmax(logits,dim=1)
		return rc[:,0].detach().numpy()

	return f



