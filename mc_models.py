import os
import sys
import torch
import ipdb
import pickle
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init as nn_init

from settings import *
from utils import *
from common_components import *

def world_to_features(world, start):
  num_feats = int(np.max(world))
  start_feat = np.zeros_like(world).astype(int)
  start_feat[start[0]-1][start[1]-1]=1.
  return np.stack([(world==i).astype(int) for i in range(num_feats)]+[start_feat],axis=0)

class GridModelBase(nn.Module):
  def __init__(self, size=10):
    super(GridModelBase, self).__init__()
    self.settings = CnfSettings()
    self.curr_fname = None
    self.size = size
    self.decoded_dim = self.settings['sharp_decoded_emb_dim']
    self.world = None
    self.start_pos = None
    self.time_mapping = None

  def update_from_file(self):
    fname = os.path.splitext(self.curr_fname)[0]+'.annt'
    with open(fname,'rb') as f:
      self.world, self.start_pos, self.time_mapping = pickle.load(f)

class GridModel1(GridModelBase):
  def __init__(self, size=10):
    super(GridModel1, self).__init__()
    self.layer1 = nn.Sequential(
      nn.Conv2d(5, 16, kernel_size=3, stride=1, padding=1),
      nn.ReLU(), 
      nn.Conv2d(16, 16, kernel_size=5, stride=1, padding=2),
      nn.ReLU(), 
      nn.Conv2d(16, self.decoded_dim, kernel_size=3, stride=1, padding=1),
      nn.ReLU()) 
    self.layer2 = MLPModel([self.decoded_dim*self.size*self.size,256,self.decoded_dim])

  # data is numpy array
  # literal_mapping is Tensor.
  # I know.


  def decode(self, data, literal_mapping, vembs, fname):
    if fname != self.curr_fname:
      self.curr_fname = fname
      self.update_from_file()  

    grid = torch.from_numpy(world_to_features(self.world, self.start_pos)).float()
    decoded_blob = self(grid)
    decoded_vembs = decoded_blob.expand(len(literal_mapping),len(decoded_blob))
    return decoded_vembs

  def forward(self, input_tensor):
    inp = input_tensor.unsqueeze(0).detach()     # Add batch and channel dimensions    
    out = self.layer1(inp)
    out = out.squeeze(0).transpose(1,2).transpose(0,2)
    rc = self.layer2(out.reshape(-1))
    return rc

class GridModel2(GridModelBase):
  def __init__(self, size=10):
    super(GridModel2, self).__init__()
    self.MAX_TIMESTEP = 20
  # data is numpy array
  # literal_mapping is Tensor.
  # I know.


  def decode(self, data, literal_mapping, vembs, fname):
    if fname != self.curr_fname:
      self.curr_fname = fname
      self.update_from_file()  
    idx = literal_mapping.abs()
    decoded_vembs = torch.Tensor([self.time_mapping[int(x)] for x in idx]).reshape(-1,1) / self.MAX_TIMESTEP
    return decoded_vembs

  def forward(self, input_tensor):
    inp = input_tensor.unsqueeze(0).detach()     # Add batch and channel dimensions    
    out = self.layer1(inp)
    out = out.squeeze(0).transpose(1,2).transpose(0,2)
    rc = self.layer2(out.reshape(-1))
    return rc

# The attention model

class GridModel3(GridModelBase):
  def __init__(self, size=10):
    super(GridModel3, self).__init__()
    self.vemb_dim = self.settings['sharp_emb_dim']
    self.layer1 = nn.Sequential(
      nn.Conv2d(5, 16, kernel_size=3, stride=1, padding=1),
      nn.ReLU(), 
      nn.Conv2d(16, 16, kernel_size=5, stride=1, padding=2),
      nn.ReLU(), 
      nn.Conv2d(16, self.decoded_dim+self.vemb_dim, kernel_size=3, stride=1, padding=1),  # vemb_dim for the key
      nn.ReLU()) 

  # data is numpy array
  # literal_mapping is Tensor.
  # I know.


  def decode(self, data, literal_mapping, vembs, fname):
    if fname != self.curr_fname:
      self.curr_fname = fname
      self.update_from_file()  

    grid = torch.from_numpy(world_to_features(self.world, self.start_pos)).float()
    global_grid = self(grid)
    s1, s2 = vembs.shape
    ipdb.set_trace()
    K = global_grid[:,:,:self.vemb_dim]
    V = global_grid[:,:,self.vemb_dim:]
    z = vembs.reshape(s1,1,1,s2).expand(s1,self.size,self.size,s2)
    decoded_vembs = decoded_blob.expand(len(literal_mapping),len(decoded_blob))
    return decoded_vembs

  def forward(self, input_tensor):
    inp = input_tensor.unsqueeze(0).detach()     # Add batch and channel dimensions    
    out = self.layer1(inp)
    rc = out.squeeze(0).transpose(1,2).transpose(0,2)
    return rc

