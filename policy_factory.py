from IPython.core.debugger import Tracer
import time
import logging

from utils import Singleton
from settings import *
from new_policies import *
from sat_policies import *
from function_policies import *
from policy_config import *

class PolicyFactory(metaclass=Singleton):
  def __init__(self, settings=None, **kwargs):
    if not settings:
      settings = CnfSettings()
    self.settings = settings

  def create_policy(self, is_clone=False, requires_grad=True, **kwargs):
    base_model = self.settings['base_model']
    policy_class = eval(self.settings['policy'])
    if base_model and not is_clone:
      print('Loading parameters from {}'.format(base_model))
      if self.settings['base_mode'] == BaseMode.ALL:
        policy = policy_class(settings=self.settings, **kwargs)
        fname = base_model if os.path.exists(base_model) else '{}/{}'.format(self.settings['model_dir'],base_model)
        if self.settings['policy_initializer'] is None:
          policy.load_state_dict(torch.load(fname))
        else:
          eval(self.settings['policy_initializer'])(policy,torch.load(fname))
      elif self.settings['base_mode'] == BaseMode.ITERS:
        base_iters = self.settings['base_iters']
        if base_iters != self.settings['max_iters']:
          base_settings = copy.deepcopy(self.settings)
          base_settings.hyperparameters['max_iters'] = base_iters
        else:
          base_settings = self.settings
        base_policy = policy_class(settings=base_settings)
        base_policy.load_state_dict(torch.load('{}/{}'.format(self.settings['model_dir'],base_model)))      
        policy = policy_class(settings=self.settings)
        policy.encoder.copy_from_encoder(base_policy.encoder, freeze=True)
      else:
        model = QbfClassifier()
        model.load_state_dict(torch.load('{}/{}'.format(self.settings['model_dir'],base_model)))
        encoder=model.encoder
        policy = policy_class(settings=self.settings,encoder=encoder)
    else:
      policy = policy_class(settings=self.settings, **kwargs)
    if self.settings['cuda']:
      policy = policy.cuda()

    if not requires_grad:
      for param in policy.parameters():
        param.requires_grad = False
      policy.eval()

    self.settings.policy = policy
    return policy
