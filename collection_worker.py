import numpy as np
import torch
import time
import ipdb
import os
import sys
import signal
import select
import torch.multiprocessing as mp
import torch.optim as optim
import cProfile
from collections import namedtuple, deque
from namedlist import namedlist


from cadet_env import *
from qbf_data import *
from settings import *
from utils import *
from rl_utils import *
from cadet_utils import *
from episode_data import *
from env_factory import *
from worker_base import *
DEF_COST = -1.000e-04

class CollectionWorker(WorkerBase):
  def run_loop(self):
    SYNC_STATS_EVERY = 5+np.random.randint(10)
    local_env_steps = 0
    self.last_grad_step = self.g_grad_steps.value
    self.lmodel.eval()    
    while True:      
      self.main_sem.acquire()       # Get the 'go' signal from main process.
      # if self.g_grad_steps.value > self.last_grad_step:   # If the model weights changed (grad step), load them
      #   print('Reloading model weights')
      #   self.lmodel.load_state_dict(self.gmodel.cpu().state_dict())        
      #   self.last_grad_step = self.g_grad_steps.value
      begin_time = time.time()
      rc = False
      while not rc:
        rc = self.step()
      total_inference_time = time.time() - begin_time
      ep = self.completed_episodes.pop(0)
      total_reward = sum([x.reward for x in ep])
      if self.settings['debug']:
        print('Forward pass in {} got episode with length {} in {} seconds!'.format(self.name,len(ep),total_inference_time))      
      self.main_queue.put(total_reward)