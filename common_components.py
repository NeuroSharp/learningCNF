import torch
import ipdb
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init as nn_init

class MLPModel(nn.Module):
  def __init__(self, dims, activation=nn.ReLU, dropout=0, batchnorm=False, layernorm=False):
    super().__init__()
    layers = []
    if dropout>0:
      nn.Dropout(dropout),
    for i in range(len(dims)-1):
      layers.append(nn.Linear(dims[i], dims[i+1]))
      if i+2 < len(dims):
        layers.append(activation())
        if batchnorm:
          layers.append(nn.BatchNorm1d(dims[i+1]))
        elif layernorm:
          layers.append(nn.LayerNorm(dims[i+1]))          
        if dropout>0:
          layers.append(nn.Dropout(dropout))

    self.model = nn.Sequential(*layers)
    for m in self.model:
      if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        nn.init.constant_(m.bias, 0)

  def forward(self, input_tensor):
    return self.model(input_tensor)
