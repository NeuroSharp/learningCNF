import os.path
import torch
# from torch.distributions import Categorical
import ipdb
import random
import time

from settings import *
from qbf_train import *
from utils import *
import torch.nn.utils as tutils

settings = CnfSettings()

qbf_dirs = ['randomQBF_8_1519436028', 'randomQBF_8_1519436021', 'randomQBF_8_1519435293', 'randomQBF_8_1519435289', 
            'randomQBF_8_1519434573', 'randomQBF_8_1519434567', 'randomQBF_8_1519433507', 
            'randomQBF_8_1519433461', 'randomQBF_8']

sat_dirs = ['train_big_10/sat', 'train_big_10/unsat']

def qbf_train_main():
  ds = QbfDataset()
  for dname in qbf_dirs:
    ds.load_dir('data/{}/'.format(dname))
  print('Loaded data, {} SAT and {} UNSAT'.format(ds.num_sat, ds.num_unsat))
  base_model = settings['base_model']
  model = QbfClassifier()
  if base_model:    
    model.load_state_dict(torch.load('{}/{}'.format(settings['model_dir'],base_model)))
  if settings['cuda']:
    model = model.cuda()
  else:
    model = model.cpu()
  optimizer = optim.SGD(model.parameters(), lr=settings['init_lr'], momentum=0.9)

  train(ds,model,optimizer=optimizer)