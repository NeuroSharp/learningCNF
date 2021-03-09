import numpy as np
import torch
import time
from IPython.core.debugger import Tracer
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

class WorkerEnv(WorkerBase):
  def discount_episode(self, ep):
    def compute_baseline(formula):
      d = self.ed.get_data()
      if not formula in d.keys() or len(d[formula]) < 3:
        latest_stats = list(x for y in d.values() for x in y[-20:])
        _, r = zip(*latest_stats)        
        return np.mean(r)

      stats = d[formula]
      steps, rewards = zip(*stats)
      return np.mean(rewards[-20:-1])

    if not ep:
      return ep      

    gamma = self.settings['gamma']    
    baseline = compute_baseline(ep[0].formula) if self.settings['stats_baseline'] else 0
    _, _, _,rewards, *_ = zip(*ep)
    r = discount(rewards, gamma) - baseline
    return [Transition(transition.state, transition.action, None, rew, transition.formula, transition.prev_obs) for transition, rew in zip(ep, r)]

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
      transition_data = self.discount_episode(self.completed_episodes.pop(0))
      # print('Forward pass in {} got episode with length {} in {} seconds!'.format(self.name,len(transition_data),total_inference_time))
      local_env_steps += len(transition_data)
      self.main_queue.put(transition_data)