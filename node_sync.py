import torch
import ipdb
import Pyro4


from shared_adam import SharedAdam
from settings import *
from utils import *
from rl_utils import *
from policy_factory import *

@Pyro4.expose
class NodeSync(object):
  def __init__(self, settings=None):
    if settings:
      self.settings = settings
    else:
      self.settings = CnfSettings()    
    # self.settings.ns = Pyro4.locateNS(host=self.settings['pyro_host'], port=self.settings['pyro_port'])    
    self.curr_lr = self.settings['init_lr']
    self.gmodel = PolicyFactory().create_policy(is_global_model=True, init_pyro=True)
    self.gmodel.share_memory()
    self.optimizer = SharedAdam(filter(lambda p: p.requires_grad, self.gmodel.parameters()), lr=self.curr_lr, weight_decay=self.settings['weight_decay'])
    self._g_steps = 0
    self._g_grad_steps = 0
    self._g_episodes = 0
    self.curr_worker = 0

    self.logger = logging.getLogger('node_sync')
    self.logger.setLevel(eval(self.settings['loglevel']))    
    fh = logging.FileHandler('logs/{}_node_sync.log'.format(log_name(self.settings)), mode='w')
    fh.setLevel(logging.DEBUG)
    self.logger.addHandler(fh)    

    self.internal_grad_steps = 0

    self.blacklisted_keys = []
    self.whitelisted_keys = []
    global_params = self.gmodel.state_dict()
    for k in global_params.keys():
      if any([x in k for x in self.settings['g2l_blacklist']]):
        self.blacklisted_keys.append(k)    
      if any([x in k for x in self.settings['l2g_whitelist']]):
        self.whitelisted_keys.append(k)    

  @property
  def g_steps(self):
    return self._g_steps

  @g_steps.setter
  def g_steps(self, val):
    self._g_steps = val

  @property
  def g_grad_steps(self):
    return self._g_grad_steps

  @property
  def g_episodes(self):
    return self._g_episodes

  @g_episodes.setter
  def g_episodes(self, val):
    self._g_episodes = val

  def mod_all(self, g_steps, g_episodes):
    self._g_steps += g_steps
    self._g_episodes += g_episodes

  def mod_g_steps(self, i):
    self._g_steps += i

  def mod_g_episodes(self, i):
    self._g_episodes += i

  def zero_grad(self):
    self.optimizer.zero_grad()

  def get_worker_num(self):
    self.curr_worker += 1
    return self.curr_worker

  # We don't bother to return the blacklisted values between nodes

  def get_state_dict(self, include_all=False):
    global_params = self.gmodel.state_dict()
    if not include_all:
      for k in self.blacklisted_keys:
        global_params.pop(k,None)
    return global_params


  # This should be called with already filtered (whitelisted) keys

  def set_state_dict(self, sdict):
    self.gmodel.load_state_dict(sdict,strict=False)

  def update_grad_and_step(self, grads):
    for lp, gp in zip(grads, self.gmodel.parameters()):
        gp._grad = lp
    self.optimizer.step()
    self._g_grad_steps += 1

  def update_lr(self, new_lr):
    if new_lr != self.curr_lr:
      utils.set_lr(self.optimizer,new_lr)
      self.curr_lr = new_lr
      self.logger.info('setting new learning rate to {}'.format(new_lr))
    return self.curr_lr
