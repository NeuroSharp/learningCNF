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
from policy_factory import *
from dispatcher import *

# DEF_COST = -1.000e-04
BREAK_CRIT_LOGICAL = 1
BREAK_CRIT_TECHNICAL = 2

MPEnvStruct = namedlist('EnvStruct',
                    ['env', 'last_obs', 'episode_memory', 'env_id', 'fname', 'curr_step', 'active', 'prev_obs', 'start_time'])


class WorkersSynchronizer:
  def __init__(self):
    self.settings = CnfSettings()
    self.logger = utils.get_logger(self.settings, 'workers_sync', 'logs/{}_workers_sync.log'.format(log_name(self.settings)))
    self.workers_to_replace = []
    if self.settings['mp']:
      self.logger.info('MultiProcessing: {} (pid: {})'.format(self.settings['mp'],os.getpid()))
      set_proc_name(str.encode('a3c_workers_sync'))

  def get_total(self):
    return len(self.workers_to_replace)

  def pop(self):
    rc = self.workers_to_replace.pop()
    self.logger.info('Worker {} poped'.format(rc[0]))
    return rc

  def add_worker(self,worker):
    self.logger.info('Worker {} finished and is waiting to be replaced'.format(worker[0]))
    self.workers_to_replace.insert(0,worker)

class WorkerEnv(mp.Process):
  def __init__(self, settings, provider, ed, name, wsync, batch_sem, init_model=None):
    super(WorkerEnv, self).__init__()
    self.index = name
    self.name = 'a3c_worker%i' % name
    self.settings = settings
    self.wsync = wsync
    self.batch_sem = batch_sem
    self.init_model = init_model
    self.ed = ed
    self.completed_episodes = []    
    self.max_step = self.settings['max_step']
    self.max_seconds = self.settings['max_seconds']
    self.sat_min_reward = self.settings['sat_min_reward']
    self.drop_technical = self.settings['drop_abort_technical']
    self.rnn_iters = self.settings['rnn_iters']
    self.training_steps = self.settings['training_steps']
    self.restart_solver_every = self.settings['restart_solver_every']    
    self.check_allowed_actions = self.settings['check_allowed_actions']
    self.memory_cap = self.settings['memory_cap']
    self.stale_threshold = self.settings['stale_threshold']
    self.minimum_episodes = self.settings['minimum_episodes']
    self.reset_counter = 0
    self.env_steps = 0
    self.real_steps = 0
    self.def_step_cost = self.settings['def_step_cost']
    self.provider = provider
    self.dispatcher = ObserverDispatcher()
    self.last_grad_steps = 0
    self.envstr = MPEnvStruct(EnvFactory().create_env(oracletype=self.lmodel.get_oracletype()), 
        None, None, None, None, None, True, deque(maxlen=self.rnn_iters), time.time())        

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

    # if not fname:
    #   if not self.reset_counter % 200:
    #     self.ds.recalc_weights()
    #   (fname,) = self.ds.weighted_sample()
    # last_obs, env_id = self.envstr.env.new_episode(fname=fname, **kwargs)
    env_obs = self.envstr.env.new_episode(fname=fname, **kwargs)
    self.envstr.last_obs = self.envstr.env.process_observation(None,env_obs)
    self.envstr.env_id = fname
    self.envstr.curr_step = 0
    self.envstr.fname = fname
    self.envstr.start_time = time.time()    
    self.envstr.episode_memory = []     
    # Set up the previous observations to be None followed by the last_obs   
    self.envstr.prev_obs.clear()    
    for i in range(self.rnn_iters):
      self.envstr.prev_obs.append(None)
    return self.envstr.last_obs

  def step(self, **kwargs):
    envstr = self.envstr
    env = envstr.env
    if not envstr.last_obs or envstr.curr_step > self.max_step:
      obs = self.reset_env(fname=self.provider.get_next())
      if obs is None:    # degenerate env
        self.logger.info('Got degenerate env: {}'.format(envstr.fname))
        self.completed_episodes.append(envstr.episode_memory)
        return True


    last_obs = collate_observations([envstr.last_obs])
    [action] = self.lmodel.select_action(last_obs, **kwargs)
    envstr.episode_memory.append(Transition(envstr.last_obs,action,None, None, envstr.env_id, envstr.prev_obs))
    allowed_actions = self.lmodel.get_allowed_actions(envstr.last_obs).squeeze() if self.check_allowed_actions else None

    if not self.check_allowed_actions or allowed_actions[action]:
      env_obs = envstr.env.step(self.lmodel.translate_action(action, envstr.last_obs))
      done = env_obs.done
    else:
      print('Chose invalid action, thats not supposed to happen.')
      assert(action_allowed(envstr.last_obs,action))

    self.real_steps += 1
    envstr.curr_step += 1

    if done:      
      for j,r in enumerate(env.rewards):
        envstr.episode_memory[j].reward = r
      self.completed_episodes.append(envstr.episode_memory)
      envstr.last_obs = None      # This will mark the env to reset with a new formula
      if env.finished:
        if self.reporter:
          self.reporter.add_stat(envstr.env_id,len(envstr.episode_memory),sum(env.rewards), 0, self.real_steps)
        if self.ed:
          # Once for every episode going into completed_episodes, add it to stats
          self.ed.ed_add_stat(envstr.fname, (len(envstr.episode_memory), sum(env.rewards))) 
      else:        
        Tracer()()

    else:
      break_env = False
      break_crit = BREAK_CRIT_LOGICAL
      if self.max_seconds:
        if (time.time()-envstr.start_time) > self.max_seconds:
          self.logger.info('Env {} took {} seconds, breaking!'.format(envstr.fname, time.time()-envstr.start_time))
          break_env=True
      elif self.sat_min_reward:        
        if env.rewards is not None and sum(env.rewards) < self.sat_min_reward:
          break_env=True
      elif envstr.curr_step > self.max_step:
        break_env=True
        break_crit = BREAK_CRIT_TECHNICAL
      if break_env:
        envstr.last_obs = None
        try:
          # We set the entire reward to zero all along
          if not env.rewards:
            env.rewards = [0.]*len(envstr.episode_memory)
          self.logger.info('Environment {} took too long, aborting it. reward: {}, steps: {}'.format(envstr.fname, sum(env.rewards), len(env.rewards)))
          env.rewards = [0.]*len(envstr.episode_memory)
          for j,r in enumerate(env.rewards):
            envstr.episode_memory[j].reward = r
        except:
          Tracer()()
        if break_crit == BREAK_CRIT_TECHNICAL and self.drop_technical:
          self.logger.info('Environment {} technically dropped.'.format(envstr.fname))          
          return True
        if self.reporter:
          self.reporter.add_stat(envstr.env_id,len(envstr.episode_memory),sum(env.rewards), 0, self.real_steps)          
        if self.ed:
          if 'testing' not in kwargs or not kwargs['testing']:
            self.ed.ed_add_stat(envstr.fname, (len(envstr.episode_memory), sum(env.rewards)))
        if self.settings['learn_from_aborted']:
          self.completed_episodes.append(envstr.episode_memory)
        return True        

      envstr.prev_obs.append(envstr.last_obs)
      envstr.last_obs = env.process_observation(envstr.last_obs,env_obs)

    return done

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

  def global_to_local(self):
    global_params = self.node_sync.get_state_dict()
    self.lmodel.load_state_dict(global_params,strict=False)
    self.last_grad_steps = self.node_sync.g_grad_steps


  def check_batch_finished(self):
    if self.settings['episodes_per_batch']:
      # print('Here it is:')
      # print(self.completed_episodes)
      # print('returning {}'.format(not (len(self.completed_episodes) < self.settings['episodes_per_batch'])))
      return not (len(self.completed_episodes) < self.settings['episodes_per_batch'])
    else:
      return not (self.episode_lengths() < self.settings['min_timesteps_per_batch'])

  def episode_lengths(self, num=0):
    rc = self.completed_episodes if num==0 else self.completed_episodes[:num]
    return sum([len(x) for x in rc])

  def pop_min(self, num=0):
    if num == 0:
      num = self.settings['min_timesteps_per_batch']
    rc = []
    i=0
    while len(rc) < num:
      ep = self.discount_episode(self.completed_episodes.pop(0))      
      rc.extend(ep)
      i += 1

    return rc, i

  def pop_all(self):
    rc = []
    rc_len = []
    i = 0
    while self.completed_episodes:
      ep = self.discount_episode(self.completed_episodes.pop(0))
      rc.extend(ep)
      rc_len.extend([i]*len(ep))
      i += 1
    return rc, rc_len, i


  def pop_min_normalized(self, num=0):
    if num == 0:
      num = self.settings['episodes_per_batch']
    rc = []
    rc_len = []
    for i in range(num):
      ep = self.discount_episode(self.completed_episodes.pop(0))
      rc.extend(ep)
      rc_len.extend([i]*len(ep))
    return rc, rc_len

  def init_proc(self):    
    set_proc_name(str.encode(self.name))
    utils.seed_all(self.settings,self.name)
    self.reporter = Pyro4.core.Proxy("PYRONAME:{}.reporter".format(self.settings['pyro_name']))
    self.node_sync = Pyro4.core.Proxy("PYRONAME:{}.node_sync".format(self.settings['pyro_name']))
    self.dispatcher.notify('new_batch')
    self.logger = utils.get_logger(self.settings, 'WorkerEnv-{}'.format(self.name), 
                                    'logs/{}_{}.log'.format(log_name(self.settings), self.name))
    self.settings.hyperparameters['cuda']=False         # No CUDA in the worker threads
    self.lmodel = PolicyFactory().create_policy()
    self.lmodel.logger = self.logger    # override logger object with process-specific one
    self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.lmodel.parameters()))
    self.blacklisted_keys = []
    self.whitelisted_keys = []
    global_params = self.lmodel.state_dict()
    for k in global_params.keys():
      if any([x in k for x in self.settings['g2l_blacklist']]):
        self.blacklisted_keys.append(k)    
      if any([x in k for x in self.settings['l2g_whitelist']]):
        self.whitelisted_keys.append(k)    
    if self.init_model is None:
      try:
        a = self.node_sync.get_state_dict(include_all=True)
      except Exception as e:
        print('Uh uh...')
        print(e)      
      self.lmodel.load_state_dict(a)
    else:
      self.logger.info('Loading model at runtime!')
      statedict = self.lmodel.state_dict()
      numpy_into_statedict(statedict,self.init_model)
      self.lmodel.load_state_dict(statedict)
    self.process = psutil.Process(os.getpid())
    if self.settings['log_threshold']:
      self.lmodel.shelf_file = shelve.open('thres_proc_{}.shelf'.format(self.name))      

  def train(self,transition_data, curr_formula=None, **kwargs):
    # print('train: batch size is {} and reward is {}'.format(len(transition_data),sum([t.reward for t in transition_data])))
    if sum([t.reward for t in transition_data]) == 0:
    # if len(transition_data) == self.settings['episodes_per_batch']*(self.settings['max_step']+1):
      self.logger.info('A lost batch, no use training')
      return
    if self.settings['do_not_learn']:
      return
    need_sem = False
    if len(transition_data) >= self.settings['episodes_per_batch']*(self.settings['max_step']+1)*self.settings['batch_size_threshold']:
      self.logger.info('Large batch encountered. Acquiring batch semaphore')
      need_sem = True
    if need_sem:
      self.batch_sem.acquire()
    self.lmodel.train()
    mt = time.time()
    loss, logits = self.lmodel.compute_loss(transition_data, **kwargs)
    mt1 = time.time()
    if self.check_stale():
      self.clean_after_stale(curr_formula)
      self.logger.info('Training aborted after loss computation')
      if need_sem:
        self.batch_sem.release()
      return
    self.logger.info('Loss computation took {} seconds on {} with length {}'.format(mt1-mt,curr_formula,len(transition_data)))
    self.optimizer.zero_grad()      # Local model grads are being zeros here!
    loss.backward()
    mt2 = time.time()
    self.logger.info('Backward took {} seconds'.format(mt2-mt1))
    if self.check_stale():
      self.clean_after_stale(curr_formula)
      self.logger.info('Training aborted after backwards computation')
      if need_sem:
        self.batch_sem.release()
      return

    # torch.nn.utils.clip_grad_norm_(self.lmodel.parameters(), self.settings['grad_norm_clipping'])
    grads = [x.grad for x in self.lmodel.parameters()]
    self.node_sync.update_grad_and_step(grads)
    # self.logger.info('Grad steps taken before step are {}'.format(self.node_sync.g_grad_steps-self.last_grad_steps))
    z = self.lmodel.state_dict()
    # We may want to sync that

    local_params = {}
    for k in self.whitelisted_keys:
      local_params[k] = z[k]
    self.node_sync.set_state_dict(local_params)

    if need_sem:
      self.batch_sem.release()

  def check_stale(self):
    if not self.settings['check_stale']:
      return False
    rc = (self.node_sync.g_grad_steps - self.last_grad_steps)
    if rc > self.stale_threshold:
      self.logger.debug('check_stale: gradient delay is {}'.format(rc))
      return True
    else:
      return False

  def clean_after_stale(self, curr_formula):
    self.completed_episodes = []
    self.dispatcher.notify('new_batch')
    self.logger.info('Batch aborted due to stale gradients in formula {}'.format(curr_formula))

  def run(self):
    if self.settings['memory_profiling']:
      tracemalloc.start(25)
    if self.settings['profiling']:
      cProfile.runctx('self.run_loop()', globals(), locals(), 'prof_{}.prof'.format(self.name))
    else:
      self.run_loop()


  def run_loop(self):
    self.init_proc()
    clock = GlobalTick()
    SYNC_STATS_EVERY = 1
    # SYNC_STATS_EVERY = 5+np.random.randint(10)
    total_step = 0    
    local_env_steps = 0
    global_steps = 0
    is_training = (not self.settings['do_not_learn'])
    # self.episodes_files = self.ds.get_files_list()
    while global_steps < self.training_steps:
      clock.tick()
      self.lmodel.eval()
      self.global_to_local()      
      begin_time = time.time()
      rc = False
      curr_formula = self.provider.get_next()      
      total_episodes = 0
      while (not rc) or (total_episodes < self.settings['episodes_per_batch']):
        if self.settings['log_threshold']:
          k = '{}_{}'.format(global_steps,len(self.completed_episodes))
          print('setting key to {}'.format(k))
          self.lmodel.shelf_key = k
          self.lmodel.shelf_file.sync()          
        rc = self.step(training=is_training)
        if rc:
          total_episodes += 1
          if self.check_stale():
            break
      if self.check_stale():
        self.clean_after_stale(curr_formula)
        continue
      total_inference_time = time.time() - begin_time
      transition_data, lenvec, num_episodes = self.pop_all()
      ns = len(transition_data)
      if ns == 0:
        self.logger.info('Degenerate batch, ignoring')
        if self.settings['autodelete_degenerate']:
          self.provider.delete_item(curr_formula)
          self.logger.debug('After deleting degenerate formula, total number of formulas left is {}'.format(self.provider.get_total()))
        self.dispatcher.notify('new_batch')
        continue
      elif num_episodes < self.minimum_episodes:
        self.logger.info('too few episodes ({}), dropping batch'.format(num_episodes))
        self.dispatcher.notify('new_batch')
        continue              
      self.logger.info('Forward pass in {} ({}) got batch with length {} ({}) in {} seconds. Ratio: {}'.format(self.name,transition_data[0].formula,len(transition_data),num_episodes,total_inference_time,len(transition_data)/total_inference_time))
      # After the batch is finished, advance the iterator
      self.dispatcher.notify('new_batch')
      self.reset_env(fname=self.provider.get_next())
      begin_time = time.time()
      self.train(transition_data,lenvec=lenvec,curr_formula=curr_formula)
      total_train_time = time.time() - begin_time
      self.logger.info('Backward pass in {} done in {} seconds!'.format(self.name,total_train_time))
      if self.check_stale():
        self.clean_after_stale(curr_formula)
        continue      

      total_process_memory = self.process.memory_info().rss / float(2 ** 20)
      if total_process_memory > self.memory_cap:
        self.logger.info('Total memory is {}, greater than memory cap which is {}'.format(total_process_memory,self.memory_cap))
        self.envstr.env.exit()
        self.wsync.add_worker((self.index,statedict_to_numpy(self.lmodel.state_dict())))
        exit()
        print("Shouldn't be here")

      # Sync to global step counts
      total_step += 1
      local_env_steps += len(transition_data)            
      if total_step % SYNC_STATS_EVERY == 0:      
        self.node_sync.mod_g_grad_steps(SYNC_STATS_EVERY)
        self.node_sync.mod_g_episodes(max(lenvec)+1)
        self.node_sync.mod_g_steps(local_env_steps)
        local_env_steps = 0
        global_steps = self.node_sync.g_grad_steps

      if self.settings['memory_profiling']:    
        print("({0})iter: {1}, memory: {2:.2f}MB".format(self.name,total_step, self.process.memory_info().rss / float(2 ** 20)))        
        objects = gc.get_objects()
        print('Number of objects is {}'.format(len(objects)))
        del objects
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        print("[ Top 20 in {}]".format(self.name))
        for stat in top_stats[:20]:
            print(stat)
        del snapshot
        del top_stats
