import os
import torch.multiprocessing as mp
from enum import Enum
from utils import set_proc_name
import gym
# from gym import *
from gym import spaces


class TrivialSpace(gym.Space):
  def contains(self, x):
    return True

  @property
  def shape(self):
    return ()

  @shape.setter
  def shape(self, value):  
    pass

class EnvCommands(Enum):
	CMD_RESET = 1
	CMD_STEP = 2
	CMD_EXIT = 3
	ACK_RESET = 4
	ACK_STEP = 5
	ACK_EXIT = 6

class EnvBase(gym.Env):
	def __init__(self, config):
		pass

	def step(self,action):
		pass

	def exit(self):
		pass

	def close(self):
		self.exit()
		
	def reset(self,fname):
		pass

	def restart_env(self, timeout=0):
		pass

	@property
	def name(self):
		return 'EnvBase'

