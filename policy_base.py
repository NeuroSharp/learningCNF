import logging
import torch.nn as nn
from settings import *
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import cadet_utils
import utils

class RLLibModel(nn.Module, TorchModelV2):
  def __init__(self, *args, **kwargs):  
    TorchModelV2.__init__(self, *args, **kwargs)
    nn.Module.__init__(self)

    # Make rllib happy
    self.outputs = None
    self.state_out = None
    self._validate_output_shape = lambda: None
    self.settings = kwargs['settings'] if 'settings' in kwargs.keys() else CnfSettings()        
    self.state_dim = self.settings['state_dim']
    self.embedding_dim = self.settings['embedding_dim']
    self.vemb_dim = self.settings['vemb_dim']
    self.cemb_dim = self.settings['cemb_dim']
    self.vlabel_dim = self.settings['vlabel_dim']
    self.clabel_dim = self.settings['clabel_dim']
    self.policy_dim1 = self.settings['policy_dim1']
    self.policy_dim2 = self.settings['policy_dim2']   
    self.max_iters = self.settings['max_iters']
    self.max_vars = self.settings['max_variables']
    self.state_bn = self.settings['state_bn']
    self.use_bn = self.settings['use_bn']
    self.entropy_alpha = self.settings['entropy_alpha']    
    self.lambda_value = self.settings['lambda_value']
    self.lambda_disallowed = self.settings['lambda_disallowed']
    self.lambda_aux = self.settings['lambda_aux']
    self.non_linearity = self.settings['policy_non_linearity']
    self.print_every = self.settings['print_every']
    self.logger = utils.get_logger(self.settings, 'RLLibModel')
    if self.non_linearity is not None:
      self.activation = eval(self.non_linearity)
    else:
      self.activation = lambda x: x



class PolicyBase(nn.Module):
  def __init__(self, oracletype=None, **kwargs):
    super(PolicyBase, self).__init__()
    self.settings = kwargs['settings'] if 'settings' in kwargs.keys() else CnfSettings()        
    self.state_dim = self.settings['state_dim']
    self.embedding_dim = self.settings['embedding_dim']
    self.vemb_dim = self.settings['vemb_dim']
    self.cemb_dim = self.settings['cemb_dim']
    self.vlabel_dim = self.settings['vlabel_dim']
    self.clabel_dim = self.settings['clabel_dim']
    self.policy_dim1 = self.settings['policy_dim1']
    self.policy_dim2 = self.settings['policy_dim2']   
    self.max_iters = self.settings['max_iters']   
    self.state_bn = self.settings['state_bn']
    self.use_bn = self.settings['use_bn']
    self.entropy_alpha = self.settings['entropy_alpha']    
    self.lambda_value = self.settings['lambda_value']
    self.lambda_disallowed = self.settings['lambda_disallowed']
    self.lambda_aux = self.settings['lambda_aux']
    self.non_linearity = self.settings['policy_non_linearity']
    self.print_every = self.settings['print_every']
    self.logger = utils.get_logger(self.settings, 'PolicyBase')
                                    
    self.oracletype = oracletype

    if self.non_linearity is not None:
      self.activation = eval(self.non_linearity)
    else:
      self.activation = lambda x: x

  
  def get_oracletype(self):
    return self.oracletype

  def forward(self, obs, **kwargs):
    raise NotImplementedError

  def select_action(self, obs_batch, **kwargs):
    raise NotImplementedError

  def translate_action(self, action, obs, **kwargs):
    raise NotImplementedError

  def combine_actions(self, actions, **kwargs):
    raise NotImplementedError
    
  def get_allowed_actions(self, obs, **kwargs):
    return cadet_utils.get_allowed_actions(obs,**kwargs)



