import torch
import shelve
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import init as nn_init
import torch.nn.functional as F
import torch.optim as optim
import utils
import numpy as np
from collections import namedtuple
from IPython.core.debugger import Tracer
from settings import *
from policy_base import *
from rl_utils import *
from tick_utils import *
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal

class FunctionPolicy(PolicyBase):
  def __init__(self, **kwargs):
    super(FunctionPolicy, self).__init__(**kwargs)
    self.sigma = self.settings.FloatTensor(np.array(float(self.settings['threshold_sigma'])))
    self.scale = self.settings.FloatTensor(np.array(float(self.settings['threshold_scale'])))
    prev = self.input_dim()
    self.policy_layers = nn.Sequential()
    n = 0
    num_layers = len([x for x in self.settings['policy_layers'] if type(x) is int])
    for (i,x) in enumerate(self.settings['policy_layers']):
      if x == 'r':
        self.policy_layers.add_module('activation_{}'.format(i), nn.ReLU())
      elif x == 'lr':
        self.policy_layers.add_module('activation_{}'.format(i), nn.LeakyReLU())
      elif x == 'h':
        self.policy_layers.add_module('activation_{}'.format(i), nn.Tanh())        
      elif x == 's':
        self.policy_layers.add_module('activation_{}'.format(i), nn.Sigmoid())        
      else:
        n += 1
        layer = nn.Linear(prev,x)
        prev = x
        if self.settings['init_threshold'] is not None:          
          if n == num_layers:
            nn.init.constant_(layer.weight,0.)
            nn.init.constant_(layer.bias,self.settings['init_threshold'])
        self.policy_layers.add_module('linear_{}'.format(i), layer)

  # state is just a (batched) vector of fixed size state_dim which should be expanded. 
  # vlabels are batch * max_vars * vlabel_dim

  # cmat is already "batched" into a single matrix

  def forward(self, obs, **kwargs):
    return self.policy_layers(obs)

  def input_dim(self):
    return 1

  def select_action(self, obs_batch, training=True, **kwargs):
    output = self.forward(obs_batch.state, **kwargs)*self.scale
    if not training:
      return output

    m = Normal(output,self.sigma)
    sampled_output = m.sample()    
    if every_tick(100):
      self.logger.info('At state: {}'.format(obs_batch.state.numpy()))
      self.logger.info('Output and sample: {}/{}'.format(output,sampled_output.numpy()))
    return sampled_output

  def translate_action(self, action, obs, **kwargs):
    return float(action.detach())

  def compute_loss(self, transition_data, **kwargs):
    collated_state = self.settings.cudaize_var(torch.cat([x.state.state for x in transition_data])).reshape(-1,1)
    collated_actions = self.settings.cudaize_var(torch.cat([x.action for x in transition_data]))
    returns = self.settings.FloatTensor([x.reward for x in transition_data])
    # Tracer()()
    outputs = self.forward(collated_state)    
    logprobs = gaussian_logprobs(outputs.view(-1,1),self.sigma,collated_actions.view(-1,1)).view(-1)    
    adv_t = returns
    value_loss = 0.
    adv_t = (adv_t - adv_t.mean())
    if self.settings['use_sum']:
      pg_loss = (-adv_t*logprobs).sum()
    else:
      pg_loss = (-adv_t*logprobs).mean()

    # Recompute moving averages

    if self.state_bn:
      self.state_vbn.recompute_moments(collated_batch.state.state.detach())
    if self.use_bn:
      z = collated_batch.state.clabels.detach().view(-1,6)
      self.clabels_vbn.recompute_moments(z.mean(dim=0).unsqueeze(0),z.std(dim=0).unsqueeze(0))

    loss = pg_loss
    return loss, outputs
