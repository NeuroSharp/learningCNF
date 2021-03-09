import os
import numpy as np
import torch
import torch.multiprocessing as mp
import time
from IPython.core.debugger import Tracer
import pickle
import itertools
import Pyro4
from collections import namedtuple
from namedlist import namedlist

from settings import *
from utils import *
from rl_utils import *

class AbstractVBN(nn.Module):
  def __init__(self, dim, **kwargs):
    super(AbstractVBN, self).__init__()
    self.settings = kwargs['settings'] if 'settings' in kwargs.keys() else CnfSettings()
    self.dim = dim
    self.vbn_init_fixed = self.settings['vbn_init_fixed']
    self.scale = nn.Parameter(self.settings.FloatTensor(dim), requires_grad=True)
    self.shift = nn.Parameter(self.settings.FloatTensor(dim), requires_grad=True)
    self.effective_mean = nn.Parameter(self.settings.FloatTensor(dim), requires_grad=False)
    self.effective_std = nn.Parameter(self.settings.FloatTensor(dim), requires_grad=False)
    if self.vbn_init_fixed:
      nn.init.constant_(self.shift,0.)
      nn_init.constant_(self.scale,1.)
    else:
      nn_init.normal_(self.scale)
      nn_init.normal_(self.shift)
    nn.init.constant_(self.effective_mean,0.)
    nn_init.constant_(self.effective_std,1.)

  def forward(self, input, **kwargs):
    n1 = input - self.effective_mean
    normed_input = n1 / (self.effective_std + float(np.finfo(np.float32).eps))
    outputs = normed_input*self.scale + self.shift
    return outputs

  def recompute_moments(self, data, **kwargs):
    pass

class MovingAverageVBN(AbstractVBN):
  def __init__(self, size, **kwargs):
    super(MovingAverageVBN,self).__init__(size[1])
    self.size = size
    self.total_length = 0
    self.moving_mean = nn.Parameter(self.settings.FloatTensor(*self.size), requires_grad=False)    
    nn.init.constant_(self.moving_mean,0.)
    self.settings['g2l_blacklist'].extend(['effective_mean', 'effective_std'])
    self.settings['l2g_whitelist'].extend(['effective_mean', 'effective_std'])        
    self.settings['g2l_blacklist'].append('moving_mean')

  def recompute_moments(self, data, **kwargs):
    dsize = len(data)
    window_size = self.size[0]
    if dsize > window_size:
      self.moving_mean.data = data[-window_size:]
    else:
      size_left = window_size - dsize
      self.moving_mean.data = torch.cat([data,self.moving_mean[:size_left]],dim=0)

    self.total_length = min(self.total_length+dsize, window_size)
    self.effective_mean.data = self.moving_mean[:self.total_length].mean(dim=0)
    self.effective_std.data = self.moving_mean[:self.total_length].std(dim=0)


# This is a weird one. It computes a moving --mean-- of a window over two lists, of means and stds.
# The idea is that every formula has its own wildly different mean/std, and 10k's of datapoints. We can't keep all
# of them around. Instead, we compute mean/std for every formula, and take a mean of a window over those, hoping
# it settles on reasonable values that don't move a lot. The window should probably depend on the size of the dataset.

class MovingAverageAndStdVBN(AbstractVBN):
  def __init__(self, size, **kwargs):
    super(MovingAverageAndStdVBN,self).__init__(size[1])
    self.size = size
    self.total_length = 0
    self.moving_mean = nn.Parameter(self.settings.FloatTensor(*self.size), requires_grad=False)
    self.moving_std = nn.Parameter(self.settings.FloatTensor(*self.size), requires_grad=False)
    nn.init.constant_(self.moving_mean,0.)
    nn.init.constant_(self.moving_std,1.)
    # This class -always- fixes the scale and shift, they will not be changed.
    nn.init.constant_(self.shift,0.)
    nn_init.constant_(self.scale,1.)
    self.settings['g2l_blacklist'].extend(['effective_mean', 'effective_std'])
    self.settings['l2g_whitelist'].extend(['effective_mean', 'effective_std'])            
    self.settings['g2l_blacklist'].append('moving_mean')
    self.settings['g2l_blacklist'].append('moving_std')



  def recompute_moments(self, data_mean, data_std):
    assert(len(data_mean) == len(data_std))
    dsize = len(data_mean)
    window_size = self.size[0]
    if dsize > window_size:
      self.moving_mean.data = data_mean[-window_size:]
      self.moving_std.data = data_std[-window_size:]
    else:
      size_left = window_size - dsize
      self.moving_mean.data = torch.cat([data_mean,self.moving_mean[:size_left]],dim=0)
      self.moving_std.data = torch.cat([data_std,self.moving_std[:size_left]],dim=0)

    self.total_length = min(self.total_length+dsize, window_size)
    self.effective_mean.data = self.moving_mean[:self.total_length].mean(dim=0)
    self.effective_std.data = self.moving_std[:self.total_length].mean(dim=0)

class NodeAverageAndStdVBN(AbstractVBN):
  def __init__(self, size, name, is_global_model=False, init_pyro=False, **kwargs):
    super(NodeAverageAndStdVBN,self).__init__(size)
    self.init_pyro = init_pyro
    if init_pyro:    
      self.main_node = self.settings['main_node'] and is_global_model
    self.name = name
    self.is_global_model = is_global_model
    self.size = size
    self.total_length = nn.Parameter(self.settings.LongTensor(1), requires_grad=False)
    nn.init.constant_(self.total_length,0.)
    # nn.init.constant_(self.shift,0.)
    # nn_init.constant_(self.scale,1.)
    self.settings['g2l_blacklist'].extend(['total_length'])
    if init_pyro:
      if self.main_node:
        main_vbn_uri = self.settings.pyrodaemon.register(self)
        self.settings.ns.register("{}.{}".format(self.settings['pyro_name'], self.name), main_vbn_uri)
      else:
        self.main_vbn = Pyro4.core.Proxy("PYRONAME:{}.{}".format(self.settings['pyro_name'], self.name))    

  @Pyro4.expose
  def recompute_moments(self, data_mean, data_std=None, weight=None):
    if data_mean.dim() == 2:
      assert(weight==None)
      assert(data_std==None)
      weight = len(data_mean)
      data_std = data_mean.std(dim=0)
      data_mean = data_mean.mean(dim=0)
    assert(len(data_mean) == len(data_std))
    if (data_std!=data_std).any().numpy():
      print('Error! nan in data_std. Dropping everything and just returning the previous values')      
      return self.effective_mean.data, self.effective_std.data      
    if not self.init_pyro or self.main_node:  
      total = (self.total_length + weight).float()
      curr_weight = self.total_length.float() / total
      new_weight = float(weight) / total      
      new_mean = curr_weight*self.effective_mean + new_weight*data_mean
      new_std = curr_weight*self.effective_std + new_weight*data_std
      self.total_length.apply_(lambda x: x+weight)
    else:
      new_mean, new_std = self.main_vbn.recompute_moments(data_mean,data_std, weight=weight)
    self.effective_mean.data = new_mean
    self.effective_std.data = new_std

    return self.effective_mean.data, self.effective_std.data


    

class FixedVBN(AbstractVBN):
  def __init__(self, size, fixed_mean=0., fixed_std=1.,  **kwargs):
    super(FixedVBN,self).__init__(size[1])
    self.size = size
    self.effective_mean.data = torch.ones(self.effective_mean.shape)*fixed_mean
    self.effective_std.data = torch.ones(self.effective_std.shape)*fixed_std
    self.settings['g2l_blacklist'].extend(['effective_mean', 'effective_std'])
