import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import init as nn_init
import torch.nn.functional as F
import torch.optim as optim
import utils
import numpy as np
import dgl
import dgl.function as fn
from collections import namedtuple
from IPython.core.debugger import Tracer
import cadet_utils
from common_components import *
from qbf_data import *
from qbf_model import *
from settings import *
from policy_base import *
from rl_utils import *
from common_components import *

class CNFLayer(nn.Module):
  def __init__(self, in_size, clause_size, out_size, activation=None, settings=None, norm_class=None, **kwargs):
    super(CNFLayer, self).__init__()
    self.settings = settings if settings else CnfSettings()
    # W_r for each relation
    self.weight = nn.ModuleDict({
      'l2c' : MLPModel(in_size, 128, clause_size),
      'c2l' : MLPModel(clause_size+self.settings['clabel_dim'], 128, out_size)
    })

    self.activation = activation if activation else eval(self.settings['non_linearity'])
    self.aggregate = fn.sum if self.settings['use_sum'] else fn.mean
    
  def forward(self, G, feat_dict):
    # the input is a dictionary of node features for each type
    Wh_l2c = self.weight['l2c'](feat_dict['literal'])
    G.nodes['literal'].data['Wh_l2c'] = Wh_l2c
    G['l2c'].update_all(fn.copy_src('Wh_l2c', 'm'), self.aggregate('m', 'h'))
    cembs = self.activation(G.nodes['clause'].data['h'])            # cembs now holds the half-round embedding
    G.nodes['clause'].data['cembs'] = cembs
    Wh_c2l = self.weight['c2l'](torch.cat([cembs,feat_dict['clause']], dim=1))
    G.nodes['clause'].data['Wh_c2l'] = Wh_c2l
    G['c2l'].update_all(fn.copy_src('Wh_c2l', 'm'), self.aggregate('m', 'h'))    
    lembs = self.activation(G.nodes['literal'].data['h'])
                    
    return lembs

# This is identical to the QbfNewEncoder (up to 1 iter)

class CNFLayer2(nn.Module):
  def __init__(self, in_size, clause_size, out_size, activation=None, settings=None, **kwargs):
    super(CNFLayer2, self).__init__()
    self.ntypes = ['literal', 'clause']
    self.etypes = ['l2c', 'c2l']
    # W_r for each relation
    self.weight = nn.ModuleDict({
      self.etypes[0] : nn.Linear(in_size, clause_size),
      self.etypes[1] : nn.Linear(clause_size+1, out_size)
    })
    self.settings = settings if settings else CnfSettings()
    self.activation = activation if activation else eval(self.settings['non_linearity'])
    
  def forward(self, G, feat_dict):
    # the input is a dictionary of node features for each type
    # mat = G.adjacency_matrix(etype='l2c')
    Wh_l2c = self.weight['l2c'](feat_dict['literal'])
    G.nodes['literal'].data['Wh_l2c'] = feat_dict['literal']
    G['l2c'].update_all(fn.copy_src('Wh_l2c', 'm'), fn.sum('m', 'h'))
    cembs = self.activation(self.weight['l2c'](G.nodes['clause'].data['h']))            # cembs now holds the half-round embedding
    G.nodes['clause'].data['Wh_c2l'] = torch.cat([cembs,feat_dict['clause']], dim=1)
    G['c2l'].update_all(fn.copy_src('Wh_c2l', 'm'), fn.sum('m', 'h'))
    lembs = self.activation(self.weight['c2l'](G.nodes['literal'].data['h']))                    
    return lembs

