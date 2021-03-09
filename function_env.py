from IPython.core.debugger import Tracer
import utils
import random

from namedlist import namedlist
from settings import *
from envbase import *
from dispatcher import *
from rl_types import *

class FunctionEnv(EnvBase, AbstractProvider):
  EnvObservation = namedlist('EnvObservation', ['state', 'reward', 'done'], default=None)
  """docstring for FunctionEnv"""
  def __init__(self, settings=None, func=None, init_x=None, **kwargs):
    super(FunctionEnv, self).__init__()
    self.settings = settings if settings else CnfSettings()
    self.logger = utils.get_logger(self.settings, 'FunctionEnv')
    if init_x is not None:
      self.init_x = init_x
    else:
      ObserverDispatcher().register('new_batch',self)
      self.init_x = 0
    self.max_step=20
    self.reset()
    if func is None:
      self.func = lambda x: -(x-20)*(x-20)
    else:
      self.func = func

  def step(self, action):
    reward = self.func(self.x+action) - self.func(self.x)
    self.rewards.append(reward)
    self.x += action
    self.local_step += 1
    self.finished = self.local_step > self.max_step
    return self.EnvObservation(self.x,reward, self.finished)

  def reset(self):
    self.x = self.init_x
    self.local_step = 0
    self.rewards = []
    self.finished = False

    return self.EnvObservation(self.x,0,False)

  def new_episode(self, *args, **kwargs):    
    return self.reset()

  def process_observation(self, last_obs, env_obs, settings=None):
    return self.EnvObservation(self.settings.FloatTensor([env_obs.state]),env_obs.reward,env_obs.done)

  def notify(self, *args, **kwargs):
    self.init_x = random.randint(0,40)