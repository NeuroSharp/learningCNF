import sys
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init as nn_init
sys.path.append('./generators')
from samplers.sudoku_sampler import var2rcn, rcn2var
from utils import *
from settings import *


class SudokuModel1(nn.Module):
  def __init__(self, size=9):
    super(SudokuModel1, self).__init__()
    self.settings = CnfSettings()
    self.decoded_dim = self.settings['sharp_decoded_emb_dim']
    self.size = size
    self.layer1 = nn.Sequential(
      nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1),
      nn.ReLU(), 
      nn.Conv3d(16, 16, kernel_size=5, stride=1, padding=2),
      nn.ReLU(), 
      nn.Conv3d(16, self.decoded_dim, kernel_size=3, stride=1, padding=1),
      nn.ReLU()) 


  def decode(self, data, literal_mapping, *_):
    grid = torch.zeros(size=(self.size,)*3, requires_grad=False)
    literal_stack = torch.from_numpy(data)
    indices = torch.stack(var2rcn(self.size,literal_stack),dim=1).long()
    comp_indices = torch.stack(var2rcn(self.size,literal_mapping),dim=1).long()
    vals = literal_stack.sign()
    patch_grid(grid.numpy(),indices.numpy(),vals.numpy())
    decoded_embs = self(grid)
    decoded_vembs = torch.from_numpy(get_from_grid(decoded_embs.detach().numpy(),comp_indices.numpy()))
    return decoded_vembs

  def forward(self, input_tensor):
    inp = input_tensor.unsqueeze(0).unsqueeze(0).detach()     # Add batch and channel dimensions    
    out = self.layer1(inp).squeeze(0).transpose(0,3)
    return out