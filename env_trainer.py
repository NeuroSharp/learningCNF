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


class IEnvTrainerHook:
  def global_to_local(self, **kwargs):
    pass
  def update_from_worker(self, grads, state_dict):
    pass

class EnvTrainer:
  def __init__(self, settings, provider, name, hook_obj, ed=None, model=None, init_model=None, **kwargs):
    super(EnvTrainer, self).__init__()
    self.name = name
    self.settings = settings
    self.init_model = init_model
    self.ed = ed
    self.hook_obj = hook_obj
    self.minimum_episodes = self.settings['minimum_episodes']
    self.provider = provider    
    self.logger = utils.get_logger(self.settings, 'EnvTrainer-{}'.format(self.name), 
                                    'logs/{}_{}.log'.format(log_name(self.settings), self.name))
    if model is None:
      self.lmodel = PolicyFactory().create_policy(**kwargs)
    else:
      self.lmodel = model
    self.lmodel.logger = self.logger    # override logger object with process-specific one
    self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.lmodel.parameters()), weight_decay=self.settings['weight_decay'])
    self.interactor = EnvInteractor(self.settings, self.lmodel, self.name, logger=self.logger, **kwargs)
    if self.init_model is not None:
      self.logger.info('Loading model at runtime!')
      statedict = self.lmodel.state_dict()
      numpy_into_statedict(statedict,self.init_model)
      self.lmodel.load_state_dict(statedict)    
    if self.settings['log_threshold']:
      self.lmodel.shelf_file = shelve.open('thres_proc_{}.shelf'.format(self.name))      

  def train(self,transition_data, **kwargs):
    # print('train: batch size is {} and reward is {}'.format(len(transition_data),sum([t.reward for t in transition_data])))
    if sum([t.reward for t in transition_data]) == 0:
    # if len(transition_data) == self.settings['episodes_per_batch']*(self.settings['max_step']+1):
      self.logger.info('A lost batch, no use training')
      return
    if self.settings['do_not_learn']:
      return
    self.lmodel.train()
    mt = time.time()
    loss, logits = self.lmodel.compute_loss(transition_data, **kwargs)    # TODO: Remove logits from this interface
    mt1 = time.time()
    self.logger.info('Loss computation took {} seconds on {} with length {}'.format(mt1-mt,self.provider.get_next(),len(transition_data)))
    self.optimizer.zero_grad()      # Local model grads are being zeros here!
    loss.backward()                 
    mt2 = time.time()
    self.logger.info('Backward took {} seconds'.format(mt2-mt1))
    grads = [x.grad for x in self.lmodel.parameters()]
    z = self.lmodel.state_dict()
    self.hook_obj.update_from_worker(grads, z)

# Returns the number of steps taken (total length of batch, in steps)

  def train_step(self, **kwargs):
    self.lmodel.eval()
    self.hook_obj.global_to_local()
    begin_time = time.time()
    curr_formula = self.provider.get_next()
    total_episodes = 0
    eps, ns = self.interactor.collect_batch(curr_formula, deterministic=False, **kwargs)   # training is in kwargs
    total_inference_time = time.time() - begin_time
    num_episodes = len(eps)
    transition_data = flatten([discount_episode(x,self.settings) for x in eps])
    lenvec = flatten([[i]*len(eps[i]) for i in range(num_episodes)])
    if ns == 0:
      self.logger.info('Degenerate batch, ignoring')
      if self.settings['autodelete_degenerate']:
        self.provider.delete_item(curr_formula)
        self.logger.debug('After deleting degenerate formula, total number of formulas left is {}'.format(self.provider.get_total()))
      return 0, 0
    elif num_episodes < self.minimum_episodes:
      self.logger.info('too few episodes ({}), dropping batch'.format(num_episodes))
      return 0, 0
    self.logger.info('Forward pass in {} ({}) got batch with length {} ({}) in {} seconds. Ratio: {}'.format(self.name,transition_data[0].formula,len(transition_data),num_episodes,total_inference_time,len(transition_data)/total_inference_time))
    # After the batch is finished, advance the iterator
    begin_time = time.time()
    self.train(transition_data,lenvec=lenvec)
    total_train_time = time.time() - begin_time
    self.logger.info('Backward pass in {} done in {} seconds!'.format(self.name,total_train_time))
    return len(transition_data), num_episodes
