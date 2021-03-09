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

from qbf_data import *
from settings import *
from utils import *
from rl_utils import *
from cadet_utils import *
from episode_data import *
from env_factory import *
from env_interactor import *  
from env_trainer import *
from worker_base import *
from dispatcher import *

class NodeWorker(WorkerBase, IEnvTrainerHook):
  def __init__(self, settings, provider, name, **kwargs):
    super(NodeWorker, self).__init__(settings, name, **kwargs)
    self.index = name
    self.name = 'NodeWorker%i' % name
    self.training_steps = self.settings['training_steps']    
    self.dispatcher = ObserverDispatcher()
    self.provider = provider
    self.whitelisted_keys = []
    self.logger = utils.get_logger(self.settings, 'NodeWorker-{}'.format(self.name), 
                                    'logs/{}_{}.log'.format(log_name(self.settings), self.name))    

  def global_to_local(self, **kwargs):
    global_params = self.node_sync.get_state_dict(**kwargs)
    self.trainer.lmodel.load_state_dict(global_params,strict=False)
    # self.last_grad_steps = self.node_sync.g_grad_steps

  def update_from_worker(self, grads, state_dict):
    self.node_sync.update_grad_and_step(grads)
    # self.logger.info('Grad steps taken before step are {}'.format(self.node_sync.g_grad_steps-self.last_grad_steps))
    if self.whitelisted_keys:    
      local_params = {}
      z = state_dict
      for k in self.whitelisted_keys:
        local_params[k] = z[k]
      self.node_sync.set_state_dict(local_params)

  def init_proc(self, **kwargs):
    super(NodeWorker, self).init_proc(**kwargs)
    self.node_sync = Pyro4.core.Proxy("PYRONAME:{}.node_sync".format(self.settings['pyro_name'])) 
    self.reporter = Pyro4.core.Proxy("PYRONAME:{}.reporter".format(self.settings['pyro_name']))
    self.trainer = EnvTrainer(self.settings, self.provider, self.index, self, reporter=self.reporter, init_pyro=True, **kwargs)
    global_params = self.trainer.lmodel.state_dict()
    for k in global_params.keys():
      if any([x in k for x in self.settings['l2g_whitelist']]):
        self.whitelisted_keys.append(k)    
    
  def run_loop(self):
    clock = GlobalTick()
    SYNC_STATS_EVERY = self.settings['sync_every']
    # SYNC_STATS_EVERY = 5+np.random.randint(10)
    total_step = 0    
    global_steps = 0
    sync_num_episodes = 0
    sync_num_steps = 0
    while global_steps < self.training_steps:
      clock.tick()
      self.dispatcher.notify('new_batch')
      num_env_steps, num_episodes = self.trainer.train_step(**self.kwargs)
      total_step += 1
      sync_num_episodes += num_episodes
      sync_num_steps += num_env_steps
      if total_step % SYNC_STATS_EVERY == 0:
        begin_time = time.time()
        self.node_sync.mod_all(sync_num_steps, sync_num_episodes)
        sync_num_episodes = 0
        sync_num_steps = 0 
        global_steps = self.node_sync.g_grad_steps
        sync_time = time.time() - begin_time
        self.logger.info('Spent {} seconds syncing its stats.'.format(sync_time))


