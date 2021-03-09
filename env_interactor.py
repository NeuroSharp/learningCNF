import numpy as np
import torch
import time
from IPython.core.debugger import Tracer
import gc
import os
import sys
import shelve
import cProfile
import tracemalloc
import psutil
import logging
from collections import namedtuple, deque
from namedlist import namedlist

from cadet_env import *
from qbf_data import *
from settings import *
from utils import *
from rl_utils import *
from env_factory import *
from policy_factory import *

# DEF_COST = -1.000e-04
BREAK_CRIT_LOGICAL = 1
BREAK_CRIT_TECHNICAL = 2

MPEnvStruct = namedlist('EnvStruct',
                    ['env', 'last_obs', 'episode_memory', 'env_id', 'fname', 'curr_step', 'active', 'prev_obs', 'start_time', 'end_time'])


class EnvInteractor:
  def __init__(self, settings, model, name, ed=None, reporter=None, logger=None, **kwargs):
    super(EnvInteractor, self).__init__()
    self.name = 'interactor_{}'.format(name)
    self.settings = settings    
    self.ed = ed
    self.lmodel = model
    self.completed_episodes = []
    self.reporter = reporter
    self.max_step = self.settings['max_step']
    self.max_seconds = self.settings['max_seconds']
    self.sat_min_reward = self.settings['sat_min_reward']
    self.drop_technical = self.settings['drop_abort_technical']    
    self.rnn_iters = self.settings['rnn_iters']
    self.restart_solver_every = self.settings['restart_solver_every']    
    self.envstr = MPEnvStruct(EnvFactory().create_env(oracletype=self.lmodel.get_oracletype()), 
        None, None, None, None, None, True, deque(maxlen=self.rnn_iters), time.time(), 0)
    self.reset_counter = 0
    self.total_steps = 0
    self.def_step_cost = self.settings['def_step_cost']
    self.process = psutil.Process(os.getpid())    
    # if self.settings['log_threshold']:
    #   self.lmodel.shelf_file = shelve.open('thres_proc_{}.shelf'.format(self.name))      
    if logger is None:
      self.logger = utils.get_logger(self.settings, 'EnvInteractor-{}'.format(self.name), 
                                    'logs/{}_{}.log'.format(log_name(self.settings), self.name))    
    else: 
      self.logger = logger
    self.lmodel.logger = self.logger

# This discards everything from the old env
  def reset_env(self, fname, **kwargs):
    self.reset_counter += 1
    if self.restart_solver_every > 0 and (self.settings['restart_in_test'] or (self.reset_counter % self.restart_solver_every == 0)):
      self.envstr.env.restart_env(timeout=0)
    self.logger.debug("({0}-{1})reset: {2}/{3}, memory: {4:.2f}MB".format(self.name, self.envstr.fname, self.reset_counter, self.envstr.curr_step, self.process.memory_info().rss / float(2 ** 20)))        
    if self.settings['memory_profiling']:
      print("({0}-{1})reset: {2}/{3}, memory: {4:.2f}MB".format(self.name, self.envstr.fname, self.reset_counter, self.envstr.curr_step, self.process.memory_info().rss / float(2 ** 20)))        
      objects = gc.get_objects()
      print('Number of objects is {}'.format(len(objects)))
      del objects

    env_obs = self.envstr.env.new_episode(fname=fname, **kwargs)
    self.envstr.last_obs = self.envstr.env.process_observation(None,env_obs)
    self.envstr.env_id = fname
    self.envstr.curr_step = 0
    self.envstr.fname = fname
    self.envstr.start_time = time.time()    
    self.envstr.end_time = 0
    self.envstr.episode_memory = []
    # Set up the previous observations to be None followed by the last_obs   
    self.envstr.prev_obs.clear()    
    for i in range(self.rnn_iters):
      self.envstr.prev_obs.append(None)
    if self.lmodel:
      self.lmodel.shelf_key=fname
    return self.envstr.last_obs


# This internal implementation method computes all the episode breaking logic. This should move, at least partially, 
# to the env.

  def check_break(self):
    break_env = False
    break_crit = BREAK_CRIT_LOGICAL
    envstr = self.envstr
    env = envstr.env
    if self.max_seconds:
      if (time.time()-envstr.start_time) > self.max_seconds:
        self.logger.info('Env {} took {} seconds, breaking!'.format(envstr.fname, time.time()-envstr.start_time))
        break_env=True
    elif self.sat_min_reward:        
      if env.rewards is not None and sum(env.rewards) < self.sat_min_reward:
        break_env=True
    if self.max_step:
      if envstr.curr_step > self.max_step:
        break_env=True
        break_crit = BREAK_CRIT_TECHNICAL

    return break_env, break_crit

# Assumes last observation in self.envstr.last_obs

  def step(self, obs, **kwargs):
    envstr = self.envstr
    env = envstr.env
    break_env = False
    break_crit = BREAK_CRIT_LOGICAL
    
    action, ent = self.lmodel.select_action(obs, **kwargs)    
    envstr.episode_memory.append(Transition(obs,action,None, None, ent, envstr.env_id, envstr.prev_obs))    
    next_obs = envstr.env.step(self.lmodel.translate_action(action, obs))    
    # next_obs, *_ = envstr.env.step(self.lmodel.translate_action(action, obs))    
    done = next_obs.done
    self.total_steps += 1
    envstr.curr_step += 1
    envstr.prev_obs.append(next_obs)
    if not done:
      envstr.last_obs = env.process_observation(envstr.last_obs,next_obs)
    return self.envstr.last_obs, None, done

  def run_episode(self, fname, **kwargs):
    envstr = self.envstr
    env = envstr.env
    self.lmodel.eval()
    obs = self.reset_env(fname)    
    if not obs:   # degenerate episode, return 0 actions taken. TODO - delete degenerate episodes
      return 0, False
    done = False
    i = 0
    while not done:      
      obs, _, done = self.step(obs, **kwargs)      
      i += 1
      break_env, break_crit = self.check_break()
      if break_env:
        break
    envstr.end_time = time.time()
    if not done:     # This is an episode where the environment did not finish on its own behalf.
      if env.rewards:
        self.logger.info('Environment {} took too long, aborting it. reward: {}, steps: {}'.format(envstr.fname, sum(env.rewards), len(env.rewards)))
      else:
        self.logger.info('Environment {} took too long, aborting it.'.format(envstr.fname))            
      env.rewards = [0.]*len(envstr.episode_memory)
      if break_crit == BREAK_CRIT_TECHNICAL and self.drop_technical:
        self.logger.info('Environment {} technically dropped.'.format(envstr.fname))
        return 0, False
    for j,r in enumerate(env.rewards):
      envstr.episode_memory[j].reward = r      
    if done or self.settings['learn_from_aborted']:
      self.completed_episodes.append(envstr.episode_memory)
    return i, done

  def run_batch(self, *args, batch_size=0, **kwargs):
    if batch_size == 0:
      batch_size = self.settings['episodes_per_batch']

    total_length = 0
    total_episodes = 0
    batch_stats = []
    for i in range(batch_size):
      episode_length, _ = self.run_episode(*args, **kwargs)
      total_length += episode_length
      if episode_length != 0:
        total_episodes += 1
        ent = np.mean([x.entropy for x in self.envstr.episode_memory])
        stats = (self.envstr.env_id,len(self.envstr.episode_memory),sum(self.envstr.env.rewards), ent, self.total_steps)
        batch_stats.append(stats)

    if self.reporter is not None:
      self.reporter.add_stats(batch_stats)

    return total_length, total_episodes

  def collect_batch(self, *args, **kwargs):
    total_length, bs = self.run_batch(*args, **kwargs)
    if total_length == 0:
      return [], 0

    rc = []
    for i in range(bs):
      rc.append(self.completed_episodes.pop(0))

    return rc, total_length

  def save(self, name):
    torch.save(self.lmodel.state_dict(), name)

  def terminate(self):
    self.envstr.env.exit()

