import ipdb
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import utils
import numpy as np
from torch.nn import init as nn_init
from collections import namedtuple
from IPython.core.debugger import Tracer
from settings import *
from policy_base import *
from rl_utils import *
from ray.rllib.policy.sample_batch import SampleBatch
from ray.util.sgd.utils import TimerStat

from dgl_layers import *
from dgl_encoders import *
from common_components import *
from graph_utils import graph_from_adj
from sudoku_models import *
from cellular_models import *
from mc_models import *

class SharpModel(PolicyBase):
  def __init__(self, *args, **kwargs):
    super(SharpModel, self).__init__(*args)
    encoder_class = eval(self.settings['sharp_encoder_type'])
    decode_class = eval(self.settings['sharp_decode_class'])
    self.decode = self.settings['sharp_decode']    
    self.decode_size = self.settings['sharp_decode_size']
    self.decoded_dim = self.settings['sharp_decoded_emb_dim']
    self.decode_module = decode_class(self.decode_size)
    self.encoder = encoder_class(self.settings)
    self.curr_fname = None
    inp_size = 0
    if self.settings['sharp_add_embedding']:
      inp_size += self.encoder.output_size()
    if self.settings['sharp_decode']:
      inp_size += self.decoded_dim
    if self.settings['sharp_add_labels']:
      inp_size += self.encoder.vlabel_dim
    self.decision_layer = MLPModel([inp_size,256,64,1])
    # self.pad = torch.Tensor([torch.finfo().min])
    self.timers = {k: TimerStat() for k in ["make_graph", "encoder", "decode", "decision"]}

  def from_batch(self, train_batch, is_training=True):
    """Convenience function that calls this model with a tensor batch.

    All this does is unpack the tensor batch to call this model with the
    right input dict, state, and seq len arguments.
    """
    def obs_from_input_dict(input_dict):
      z = list(input_dict.items())
      dense_obs = [undensify_obs(DenseState(*x)) for x in list(z[3][1])]
      return dense_obs

    collated_batch = collate_observations(obs_from_input_dict(train_batch),settings=self.settings)
    input_dict = {
      "collated_obs": collated_batch,
      "is_training": is_training,
    }
    if SampleBatch.PREV_ACTIONS in train_batch:
      input_dict["prev_actions"] = train_batch[SampleBatch.PREV_ACTIONS]
    if SampleBatch.PREV_REWARDS in train_batch:
      input_dict["prev_rewards"] = train_batch[SampleBatch.PREV_REWARDS]
    states = []
    i = 0
    while "state_in_{}".format(i) in train_batch:
      states.append(train_batch["state_in_{}".format(i)])
      i += 1
    return self.__call__(input_dict, states, train_batch.get("seq_lens"))
  
  # state is just a (batched) vector of fixed size state_dim which should be expanded. 
  # ground_embeddings are batch * max_vars * ground_embedding

  # cmat_net and cmat_pos are already "batched" into a single matrix
  def forward(self, input_dict, state, seq_lens, es=True, **kwargs):
    def obs_from_input_dict(input_dict):
      z = list(input_dict.items())
      z1 = list(z[0][1][0])
      return undensify_obs(DenseState(*z1))

    with self.timers['make_graph']:
      if es:
        obs = undensify_obs(input_dict)
      else:
        obs = obs_from_input_dict(input_dict)       # This is an experience rollout
      self.decode_module.eval()
      self.encoder.eval()      
      lit_features = obs.ground[:,1:]
      literal_mapping = obs.ground[:,0]
      G = graph_from_adj(lit_features, None, obs.cmat)
      self._value_out = torch.zeros(1).expand(len(lit_features))
    with self.timers['encoder']:
      vembs, cembs = self.encoder(G)    
    with self.timers['decode']:
      if self.decode and len(obs.ext_data[1]): 
        decoded_vembs = self.decode_module.decode(obs.ext_data[1], literal_mapping, vembs, self.curr_fname)
      elif self.decode:
        decoded_vembs = torch.zeros(len(lit_features),self.decoded_dim)
    out = []
    if self.settings['sharp_add_embedding']:
      out.append(vembs)
    if self.decode:
      out.append(decoded_vembs)      
    if self.settings['sharp_add_labels']:
      out.append(lit_features)
    with self.timers['decision']:
      logits = self.decision_layer(torch.cat(out,dim=1)).t()
    # allowed_actions = self.get_allowed_actions(obs).int().float()
    # inf_mask = torch.max(allowed_actions.log(),torch.Tensor([torch.finfo().min]))
    # logits = logits + inf_mask
    
    # self.outputs = torch.cat([logits,self.pad.expand((1,self.max_vars-logits.shape[1]))], dim=1)
    # return self.outputs, []
    return logits, []

  def get_allowed_actions(self, obs, **kwargs):
    def add_other_polarity(indices):
      pos = torch.where(1-indices%2)[0]
      neg = torch.where(indices%2)[0]
      add_pos = indices[pos] + 1
      add_neg = indices[neg] - 1
      return torch.cat([indices,add_pos,add_neg],axis=0).unique()

    literal_indices = torch.unique(obs.cmat.coalesce().indices()[1])
    allowed_indices = add_other_polarity(literal_indices)
    allowed_actions = torch.zeros(obs.ground.shape[0])
    allowed_actions[allowed_indices] = 1
    return allowed_actions



  def value_function(self):
    return self._value_out.view(-1)
