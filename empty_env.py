from IPython.core.debugger import Tracer
import utils
import random
import numpy as np
import time

from namedlist import namedlist
from settings import *
from envbase import *
from dispatcher import *
from rl_types import *


class EmptyEnv(EnvBase):
  EnvObservation = namedlist('EnvObservation', ['state', 'clabels', 'reward', 'done'], default=None)
  """docstring for EmptyEnv"""
  def __init__(self, settings=None, func=None, init_x=None, **kwargs):
    super(EmptyEnv, self).__init__()
    self.settings = settings if settings else CnfSettings()
    self.logger = utils.get_logger(self.settings, 'EmptyEnv')
    self.obs_state_shape = [1,self.settings['state_dim']]
    self.obs_clabels_shape = [self.settings['sat_reduce_base'], self.settings['clabel_dim']]
    self.step_length = self.settings['empty_step_length']
    self.rewards = []

  def random_clabels(self):
    CLABEL_LBD = 3
    clabels = np.zeros(self.obs_clabels_shape)
    clabels[:,CLABEL_LBD] = np.random.randint(2,30,size=(self.obs_clabels_shape[0]))
    return clabels

  def step(self, action):
    reward = random.uniform(-2,2)
    self.rewards.append(reward)
    time.sleep(random.uniform(0,self.step_length))

    return self.EnvObservation(np.random.random(self.obs_state_shape),self.random_clabels(),reward, random.random() < 0.08)

  def reset(self):
    self.rewards = []
    self.finished = False
    return self.EnvObservation(np.random.random(self.obs_state_shape),self.random_clabels() ,0,False)

  def new_episode(self, *args, **kwargs):    
    return self.reset()

  def process_observation(self, last_obs, env_obs, settings=None):
    return State(torch.from_numpy(env_obs.state).float(),None, torch.zeros(2,self.settings['vlabel_dim']), torch.from_numpy(env_obs.clabels).float(), None, None, (0,self.obs_clabels_shape[0]))
