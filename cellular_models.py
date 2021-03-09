import sys
import torch
import ipdb
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init as nn_init

from settings import *
from utils import *

def var2grid(size, var):
  var = var.abs() -1
  col = var % size
  row = (var - col) / size
  return row, col

class CellularModel1(nn.Module):
  def __init__(self, size=300):
    super(CellularModel1, self).__init__()
    self.settings = CnfSettings()
    self.decoded_dim = self.settings['sharp_decoded_emb_dim']
    self.size = size
    self.layer1 = nn.Sequential(
      nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
      nn.ReLU(), 
      nn.Conv2d(16, 16, kernel_size=5, stride=1, padding=2),
      nn.ReLU(), 
      nn.Conv2d(16, self.decoded_dim, kernel_size=3, stride=1, padding=1),
      nn.ReLU()) 

  # data is numpy array
  # literal_mapping is Tensor.
  # I know.


  def decode(self, data, literal_mapping, *_):    
    literal_stack = torch.from_numpy(data)
    maxvar = torch.max(literal_stack.long().abs().max(),literal_mapping.long().abs().max()).detach().numpy()
    res = maxvar % self.size
    maxrow = int((maxvar - res) / self.size)
    grid = torch.zeros(size=(maxrow, self.size), requires_grad=False)
    indices = torch.stack(var2grid(self.size,literal_stack),dim=1).long()
    comp_indices = torch.stack(var2grid(self.size,literal_mapping),dim=1).long()
    vals = literal_stack.sign()
    patch_grid(grid.numpy(),indices.numpy(),vals.numpy())
    decoded_embs = self(grid)
    decoded_vembs = torch.from_numpy(get_from_grid(decoded_embs.detach().numpy(),comp_indices.numpy()))
    return decoded_vembs

  def forward(self, input_tensor):
    inp = input_tensor.unsqueeze(0).unsqueeze(0).detach()     # Add batch and channel dimensions    
    out = self.layer1(inp)
    rc = out.squeeze(0).transpose(1,2).transpose(0,2)
    return rc