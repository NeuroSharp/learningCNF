import numpy as np
import torch
import time
from IPython.core.debugger import Tracer
import gc
import os
import sys
import signal
import select
import shelve
import torch.multiprocessing as mp
import cProfile
import tracemalloc
import psutil
import logging
import pickle
import Pyro4
from collections import namedtuple, deque
from namedlist import namedlist

from tick import *
from tick_utils import *
from cadet_env import *
from qbf_data import *
from settings import *
from utils import *
from rl_utils import *
from cadet_utils import *
from episode_data import *
from env_factory import *
from env_interactor import *
from policy_factory import *

# this takes an env_interactor and knows to take a list of files (provider) and test it.

TestResultStruct = namedlist('TestResultStruct',
                    ['time', 'steps', 'reward', 'completed'])



class EnvTester:

  def Rewards(rc):
    return np.array([x.reward for [x] in rc.values()]).squeeze()  

  def __init__(self, settings, name, **kwargs):
    super(EnvTester, self).__init__()
    self.name = name
    self.settings = settings
    self.logger = utils.get_logger(self.settings, 'EnvTester-{}'.format(self.name), 
                                    'logs/{}_{}.log'.format(log_name(self.settings), self.name))    
    
  def test_envs(self, provider, model, iters=10, log_name=None, **kwargs):
    self.lmodel = model
    self.interactor = EnvInteractor(self.settings, model, self.name, logger=self.logger, **kwargs)
    if kwargs.get('log_threshold',False):
      self.lmodel.shelf_file = {}     
    self.logger.info('Testing {} envs..\n'.format(provider.get_total()))
    all_episode_files = provider.items
    totals = 0.
    total_srate = 0.
    total_scored = 0
    rc = {}
    kwargs['testing']=True
    tasks = []
    for fname in all_episode_files:
      tasks.extend([fname]*iters)
    while tasks:      
      fname=tasks.pop(0)        
      self.logger.info('Starting {}'.format(fname))
      episode_length, finished = self.interactor.run_episode(fname, **kwargs)
      self.logger.info('Finished {}'.format(fname))
      if episode_length == 0:
        # rc[fname].append(TestResultStruct(0.,0,0.,True))
        continue
      ep = self.interactor.completed_episodes.pop(0)
      total_reward = sum([x.reward for x in ep])                  
      total_time = self.interactor.envstr.end_time - self.interactor.envstr.start_time
      res = TestResultStruct(total_time,episode_length,total_reward,finished)
      if fname not in rc.keys():
        rc[fname] = []
      rc[fname].append(res)
      if len(rc[fname]) == iters:
        mean_steps = np.mean([x.steps for x in rc[fname]])
        mean_reward = np.mean([x.reward for x in rc[fname]])
        mean_time = np.mean([x.time for x in rc[fname]])
        self.logger.info('Finished {}, Averages (time,steps,reward) are {},{},{}'.format(fname,rc[fname],
          mean_time,mean_steps,mean_reward))      

    if kwargs.get('log_threshold',False):
      with open(log_name,'wb') as f:
        pickle.dump(self.lmodel.shelf_file,f)      
    self.interactor.terminate()
    return rc
