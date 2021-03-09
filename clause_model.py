import os
import ipdb
import dgl
import torch
import torch.nn as nn

from dgl_layers import *
from dgl_encoders import *
from common_components import *

# class NodeApplyModule(nn.Module):
#     """Update the node feature hv with ReLU(Whv+b)."""
#     def __init__(self, in_feats, out_feats, activation):
#         super(NodeApplyModule, self).__init__()
#         self.linear = nn.Linear(in_feats, out_feats)
#         self.activation = activation

#     def forward(self, node):
#         h = self.linear(node.data['h'])
#         h = self.activation(h)
#         return {'h' : h}

class ClausePredictionModel(nn.Module):
  def __init__(self, settings=None, prediction=True, **kwargs):
    super(ClausePredictionModel, self).__init__(**kwargs)
    self.settings = settings if settings else CnfSettings()
    self.prediction = prediction
    self.gss_dim = self.settings['state_dim']
    encoder_class = eval(self.settings['cp_encoder_type'])
    self.encoder = encoder_class(settings)
    inp_size = 0
    if self.settings['cp_add_embedding']:
      inp_size += self.encoder.output_size()
    if self.settings['cp_add_labels']:
      inp_size += self.encoder.clabel_dim
    if self.settings['cp_add_gss']:
      inp_size += self.gss_dim
    self.decision_layer = MLPModel([inp_size,256,64,self.settings['cp_num_categories']])

  def forward(self, input_dict, **kwargs):
    gss = input_dict['gss']
    G = input_dict['graph'].local_var()
    if self.prediction:
      pred_idx = torch.where(G.nodes['clause'].data['predicted_clauses'])[0]
    feat_dict = {
      'literal': G.nodes['literal'].data['literal_feats'],
      'clause': G.nodes['clause'].data['clause_feats'],        
      # 'literal': G.nodes['literal'].data['lit_labels'],
      # 'clause': G.nodes['clause'].data['clause_labels'],        
    }

    out = []
    if self.settings['cp_add_embedding']:
      vembs, cembs = self.encoder(G,feat_dict)    
      out.append(cembs)
    if self.settings['cp_add_labels']:
      out.append(feat_dict['clause'])
    if self.settings['cp_add_gss']:
      out.append(gss)
    logits = self.decision_layer(torch.cat(out,dim=1))
    if self.prediction:
      return logits[pred_idx]
    else:
      return logits


