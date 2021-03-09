import numpy as np
import torch
import time
from IPython.core.debugger import Tracer
import os
import sys
import signal
import select
from collections import namedtuple, deque
from namedlist import namedlist


from cadet_env import *
from qbf_data import *
from episode_data import *
from settings import *
from utils import *
from rl_utils import *
from formula_utils import *
from cadet_utils import *
from env_factory import *

DEF_COST = -1.000e-04

EnvStruct = namedlist('EnvStruct',
                    ['env', 'last_obs', 'episode_memory', 'env_id', 'fname', 'curr_step', 'active', 'prev_obs', 'start_time', 'end_time'])


class EpisodeManager(object):
  def __init__(self, provider, ed=None, parallelism=20, reporter=None):
    self.settings = CnfSettings()
    self.debug = False
    self.parallelism = parallelism
    self.max_seconds = self.settings['max_seconds']
    self.sat_min_reward = self.settings['sat_min_reward']    
    self.max_step = self.settings['max_step']
    self.rnn_iters = self.settings['rnn_iters']
    self.ed = ed
    self.provider = provider
    self.packed = self.settings['packed']
    self.masked_softmax = self.settings['masked_softmax']
    self.check_allowed_actions = self.settings['check_allowed_actions']
    self.envs = []
    self.completed_episodes = []
    self.real_steps = 0
    self.max_reroll = 0
    self.reporter = reporter
    self.bad_episodes = 0
    self.reset_counter = 0
    self.bad_episodes_not_added = 0
    self.INVALID_ACTION_REWARDS = -10

    for i in range(parallelism):
      self.envs.append(EnvStruct(EnvFactory().create_env(), 
        None, None, None, None, None, True, deque(maxlen=self.rnn_iters), 0, 0))

  def check_batch_finished(self):
    if self.settings['episodes_per_batch']:
      return not (len(self.completed_episodes) < self.settings['episodes_per_batch'])
    else:
      return not (self.episode_lengths() < self.settings['min_timesteps_per_batch'])

  def episode_lengths(self, num=0):
    rc = self.completed_episodes if num==0 else self.completed_episodes[:num]
    return sum([len(x) for x in rc])

  def discount_episode(self, ep):

    def compute_baseline(formula):
      if not formula in self.ed.data.keys() or len(self.ed.data[formula]) < 3:
        latest_stats = list(x for y in self.ed.data.values() for x in y[-20:])
        _, r = zip(*latest_stats)        
        return np.mean(r)

      stats = self.ed.data[formula]
      steps, rewards = zip(*stats)
      return np.mean(rewards[-20:-1])

    if not ep:
      return ep      
    gamma = self.settings['gamma']    
    baseline = compute_baseline(ep[0].formula) if self.settings['stats_baseline'] else 0
    _, _, _,rewards, *_ = zip(*ep)
    r = discount(rewards, gamma) - baseline
    return [Transition(transition.state, transition.action, None, rew, transition.formula, transition.prev_obs) for transition, rew in zip(ep, r)]

  def pop_min(self, num=0):
    if num == 0:
      num = self.settings['min_timesteps_per_batch']
    rc = []
    while len(rc) < num:
      ep = self.discount_episode(self.completed_episodes.pop(0))
      rc.extend(ep)

    return rc

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

  def reset_all(self):
    self.provider.reset()
    for envstr in self.envs:
      self.reset_env(envstr,fname=self.provider.get_next())


  def restart_all(self):
    if self.settings['is_sat']:      
      return
    for envstr in self.envs:   
      envstr.env.stop_cadet(timeout=0)
    time.sleep(2)
    for envstr in self.envs:
      envstr.env.start_cadet()
      envstr.last_obs = None          # This lets step_all know its an "empty" env that got to be reset.

# This discards everything from the old env
  def reset_env(self, envstr, fname, **kwargs):
    self.reset_counter += 1
    if self.settings['restart_in_test']:
      envstr.env.restart_env(timeout=0)
    # if not fname:
    #   if not self.reset_counter % 20:
    #     self.ds.recalc_weights()
    #   (fname,) = self.ds.weighted_sample()
    env_obs = envstr.env.new_episode(fname=fname, **kwargs)
    envstr.last_obs = envstr.env.process_observation(None,env_obs)
    envstr.env_id = fname
    envstr.curr_step = 0
    envstr.fname = fname
    envstr.start_time = time.time()
    envstr.end_time = 0
    envstr.episode_memory = []     
    # Set up the previous observations to be None followed by the last_obs   
    envstr.prev_obs.clear()    
    for i in range(self.rnn_iters):
      envstr.prev_obs.append(None)
    return envstr.last_obs

# Step the entire pipeline one step, reseting any new envs. 

  def step_all(self, model, **kwargs):
    step_obs = []
    prev_obs = []    
    rc = []     # the env structure indices that finished and got to be reset (or will reset automatically next step)
    active_envs = [i for i in range(self.parallelism) if self.envs[i].active]
    for i in active_envs:
      envstr = self.envs[i]
      # if not envstr.last_obs or envstr.curr_step > self.max_step:        
      if not envstr.last_obs:        
        obs = self.reset_env(envstr,fname=self.provider.get_next())
        if obs is None:    # degenerate env
          print('Got degenerate env: {}'.format(envstr.fname))
          self.completed_episodes.append(envstr.episode_memory)
          return True

        # print('Started new Environment ({}).'.format(envstr.fname))
      step_obs.append(envstr.last_obs)          # This is an already processed last_obs, from last iteration
      prev_obs.append(list(envstr.prev_obs))

    if step_obs.count(None) == len(step_obs):
      return rc
    if self.packed:
      obs_batch = packed_collate_observations(step_obs)
      vp_ind = obs_batch.pack_indices[1]
    else:
      obs_batch = collate_observations(step_obs)
      prev_obs_batch = [collate_observations(x,replace_none=True, c_size=obs_batch.cmask.shape[1], v_size=obs_batch.vmask.shape[1]) for x in zip(*prev_obs)]
      if prev_obs_batch and prev_obs_batch[0].vmask is not None and prev_obs_batch[0].vmask.shape != obs_batch.vmask.shape:
        Tracer()()
    allowed_actions = model.get_allowed_actions(obs_batch,packed=self.packed) if self.check_allowed_actions else None
    actions = self.packed_select_action(obs_batch, model=model, **kwargs) if self.packed else self.select_action(obs_batch, model=model, prev_obs=prev_obs_batch, **kwargs)    
    for i, envnum in enumerate(active_envs):
      envstr = self.envs[envnum]
      env = envstr.env
      env_id = envstr.env_id
      envstr.episode_memory.append(Transition(step_obs[i],actions[i],None, None, envstr.env_id, prev_obs[i]))
      self.real_steps += 1
      envstr.curr_step += 1      
      if not self.check_allowed_actions:
        action_ok = True
      elif ('cadet_test' in kwargs and kwargs['cadet_test']):
        action_ok = True
      elif self.packed:
        action_ok = allowed_actions[vp_ind[i]+actions[i][0]]
      else:
        action_ok = allowed_actions[i][actions[i]]      # New policies
      if action_ok:
        env_obs = env.step(model.translate_action(actions[i],step_obs[i]))
        done = env_obs.done
      else:
        print('Chose an invalid action! In the packed version. That was not supposed to happen.')
        env.rewards = env.terminate()
        env.rewards = np.append(env.rewards,self.INVALID_ACTION_REWARDS)
        done = True       
      if done:
        if env.rewards is not None:
          print('Finished env {}. reward is: {}'.format(envstr.fname,sum(env.rewards)))
          for j,r in enumerate(env.rewards):
            envstr.episode_memory[j].reward = r
        self.completed_episodes.append(envstr.episode_memory)
        envstr.last_obs = None      # This will mark the env to reset with a new formula
        envstr.end_time = time.time()
        rc.append((envnum,True))
        if env.finished:
          self.reporter.add_stat(env_id,len(envstr.episode_memory),sum(env.rewards), 0, self.real_steps)
          if self.ed:
            # Once for every episode going into completed_episodes, add it to stats
            self.ed.ed_add_stat(envstr.fname, (len(envstr.episode_memory), sum(env.rewards))) 
        else:        
          Tracer()()
      else:        
        break_env = False
        if self.max_seconds:
          if (time.time()-envstr.start_time) > self.max_seconds:
            self.logger.info('Env {} took {} seconds, breaking!'.format(envstr.fname, time.time()-envstr.start_time))
            break_env=True
        elif self.sat_min_reward:        
          if env.rewards is not None and sum(env.rewards) < self.sat_min_reward:
            break_env=True
        elif envstr.curr_step > self.max_step:
          break_env=True

        # break_env = False
        # if max_seconds:
        #   if (time.time()-envstr.start_time) > max_seconds:
        #     print('Env took {} seconds, breaking!'.format(time.time()-envstr.start_time))
        #     break_env=True
        # elif envstr.curr_step > self.max_step:
        #   break_env=True
        if break_env:  
          envstr.end_time = time.time()
          envstr.last_obs = None
          try:
            if env.rewards is None:
              env.rewards = [DEF_COST]*len(envstr.episode_memory)            
            # print('Finished env, rewards are: {}, sum is {}'.format(env.rewards, sum(env.rewards)))
            for j,r in enumerate(env.rewards):
              envstr.episode_memory[j].reward = r
          except:
            Tracer()()
          print('Environment {} took too long, aborting it. reward is: {}'.format(envstr.fname, sum(env.rewards)))
          if self.reporter:
            self.reporter.add_stat(envstr.env_id,len(envstr.episode_memory),sum(env.rewards), 0, self.real_steps)          

          if self.ed:
            # We add to the statistics the envs that aborted, even though they're not learned from
            if 'testing' not in kwargs or not kwargs['testing']:
              self.ed.ed_add_stat(envstr.fname, (len(envstr.episode_memory), sum(env.rewards))) 
          rc.append((envnum,False))
          if ('testing' not in kwargs or not kwargs['testing']) and self.settings['learn_from_aborted']:
            self.completed_episodes.append(envstr.episode_memory)          
        else:
          envstr.prev_obs.append(envstr.last_obs)
          envstr.last_obs = env.process_observation(envstr.last_obs,env_obs)


    return rc



  def select_action(self, obs_batch, model=None, testing=False, random_test=False, activity_test=False, cadet_test=False, **kwargs):        
    bs = len(obs_batch.ground)
    activities = obs_batch.ground.data.numpy()[:,:,IDX_VAR_ACTIVITY]
    allowed_actions = model.get_allowed_actions(obs_batch) if model else get_allowed_actions(obs_batch)
    actions = []
    if random_test:
      for allowed in allowed_actions:
        choices = np.where(allowed.numpy())[0]
        actions.append(np.random.choice(choices))
      return actions
    elif activity_test:
      for i,act in enumerate(activities):
        if np.any(act):
          actions.append(np.argmax(act))
        else:
          choices = np.where(allowed_actions[i].numpy())[0]
          actions.append(np.random.choice(choices))
      return actions
    elif cadet_test:
      return ['?']*bs
      # return '?'
    actions = model.select_action(obs_batch, **kwargs)    
    return actions

  def packed_select_action(self, obs_batch, model=None, testing=False, random_test=False, activity_test=False, cadet_test=False, **kwargs):        
    bs = len(obs_batch.ground)
    activities = obs_batch.ground.data.numpy()[:,IDX_VAR_ACTIVITY]
    allowed_actions = get_allowed_actions(obs_batch, packed=True)
    actions = []
    pack_indices = obs_batch.pack_indices

    if random_test:
      i=0
      while i < len(pack_indices):
        choices = np.where(allowed[pack_indices[i]:pack_indices[i+1]].numpy())[0]
    
      Tracer()()      
      for allowed in allowed_actions:
        choices = np.where(allowed.numpy())[0]
        actions.append(np.random.choice(choices))

      return actions, None
    elif activity_test:
      for i,act in enumerate(activities):
        if np.any(act):
          actions.append(np.argmax(act))
        else:
          choices = np.where(allowed_actions[i].numpy())[0]
          actions.append(np.random.choice(choices))
      return actions, None
    elif cadet_test:
      return ['?']*bs, None
      
    return actions, logits

  def test_envs(self, fnames, model, ed=None, iters=10, **kwargs):
    max_seconds = int(kwargs['max_seconds'])      
    print('Testing {} envs..\n'.format(self.provider.get_total()))
    all_episode_files = self.provider.items
    totals = 0.
    total_srate = 0.
    total_scored = 0
    rc = {}
    kwargs['testing']=True
    self.restart_all()
    available_envs = list(range(self.parallelism))    
    tasks = []
    for fname in all_episode_files:
      tasks.extend([fname]*iters)
    while tasks or len(available_envs) < self.parallelism:
      while available_envs and tasks:
        i = available_envs.pop(0)
        fname=tasks.pop(0)        
        if self.debug:
          print('Starting {} on Env #{}'.format(fname,i))
        envstr = self.envs[i]
        obs = self.reset_env(envstr,fname=fname)
        if not obs:       # Env was solved in 0 steps, just ignore it
          if self.debug:
            print('File {} took 0 steps on solver #{}, ignoring'.format(fname,i))
          available_envs.append(i)
      if not tasks:       # We finished the tasks, mark all current available solvers as inactive
        if self.debug:
          print('Tasks are empty. Available envs: {}'.format(available_envs))
        for i in available_envs:
          if self.debug:
              print('Marking Env #{} as inactive'.format(i))
          self.envs[i].active = False      
      finished_envs = self.step_all(model, **kwargs)
      if finished_envs:
        if self.debug:
          print('Finished Envs: {}'.format(finished_envs))
        for i, finished in finished_envs:
          fname = self.envs[i].fname
          if finished:
            if self.debug:
              print('Finished {} on Solver #{}'.format(fname,i))
            ep = self.completed_episodes.pop(0)
            # res = sum([x.reward for x in ep])            
            if max_seconds:
              res = self.envs[i].end_time - self.envs[i].start_time
            else:
              res = sum([1 for x in ep])
            # Tracer()()
          else:
            print('Env {} took too long on Solver #{}!'.format(fname,i))
            if max_seconds:
              res = self.envs[i].end_time - self.envs[i].start_time
            else:
              res = len(self.envs[i].episode_memory)            
          if ed is not None:
            ed.ed_add_stat(fname,res)
          if fname not in rc.keys():
            rc[fname] = []
          rc[fname].append(res)
          if len(rc[fname]) == iters:
            print('Finished {}, results are: {}, Average/Min are {}/{}'.format(fname,rc[fname],
              np.mean(rc[fname]),min(rc[fname])))
        available_envs.extend([x[0] for x in finished_envs])

    return rc

  def mp_test_envs(self, fnames, model, ed=None, iters=10, **kwargs):
    ds = QbfDataset(fnames=fnames)
    print('Testing {} envs..\n'.format(len(ds)))
    all_episode_files = ds.get_files_list()
    totals = 0.
    total_srate = 0.
    total_scored = 0
    rc = {}
    seed_idx = 0
    poll = select.poll()
    pipes = [None]*self.parallelism
    self.restart_all()
    available_envs = list(range(self.parallelism))    
    busy_envs = [False]*self.parallelism
    pids = [0]*self.parallelism
    tasks = []
    for fname in all_episode_files:
      rc[fname] = []
      tasks.extend([fname]*iters)
    for envstr in self.envs:        # All envs start (and stay) inactive in parent process
      envstr.active = False
    while tasks or any(busy_envs):
      while available_envs and tasks:
        i = available_envs.pop(0)
        fname=tasks.pop(0)
        if self.debug:
          print('Starting {} on Env #{}'.format(fname,i))
        envstr = self.envs[i]
        envstr.fname = fname        # An UGLY HACK. This is for the parent process to also have the file name.
        if pipes[i]:
          poll.unregister(pipes[i][0])
          os.close(pipes[i][0])
        pipes[i] = os.pipe()    # reader, writer
        poll.register(pipes[i][0], select.POLLIN)
        # poll.register(pipes[i][0], select.POLLIN | select.POLLHUP)
        pid = os.fork()
        seed_idx += 1
        if not pid:     # child
          os.close(pipes[i][0])
          self.envs[i].active=True
          np.random.seed(int(time.time())+seed_idx)
          # envstr.env.restart_env(timeout=1)
          self.reset_env(envstr,fname=fname)
          finished_envs = []
          while not finished_envs:      # Just one (the ith) env is actually active and running
            finished_envs = self.step_all(model,**kwargs)
          finished = finished_envs[0][1]
          if finished:
            if self.debug:
              print('Finished {} on Env #{}'.format(fname,i))
            res = len(self.completed_episodes.pop(0))            
          else:
            print('Env {} took too long!'.format(fname,i))
            res = len(self.envs[i].episode_memory)
          os.write(pipes[i][1],str((i,res)).encode())
          os._exit(os.EX_OK)
        
        # Parent continues here
        os.close(pipes[i][1])
        busy_envs[i] = True
        pids[i]=pid

      # We are now most likely out of available solvers, so wait on the busy ones (Which are all until the very end)

      finished_envs = poll.poll()
      for fd, event in finished_envs:
        # print('Got event {}'.format(event))
        if event == select.POLLHUP:
          # print('Why do I get POLLHUP?')
          continue
        i, res = eval(os.read(fd,1000).decode())
        # print('Read the end of env {}'.format(i))
        busy_envs[i] = False
        available_envs.append(i)
        cleanup_process(pids[i])
        envstr = self.envs[i]
        fname = envstr.fname
        rc[fname].append(res)
        # if ed is not None:
        #   ed.ed_add_stat(fname,res)
        if len(rc[fname]) == iters:
          print('Finished {}, results are: {}, Average/Min are {}/{}'.format(fname,rc[fname],
            np.mean(rc[fname]),min(rc[fname])))
        
    return rc


  def workers_test_envs(self, fnames, model, ed=None, iters=10, **kwargs):
    ds = QbfDataset(fnames=fnames)
    print('Testing {} envs..\n'.format(len(ds)))
    all_episode_files = ds.get_files_list()
    totals = 0.
    total_srate = 0.
    total_scored = 0
    rc = {}
    seed_idx = 0
    poll = select.poll()
    pipes = [None]*self.parallelism
    self.restart_all()
    available_envs = list(range(self.parallelism))    
    busy_envs = [False]*self.parallelism
    pids = [0]*self.parallelism
    tasks = []
    for fname in all_episode_files:
      rc[fname] = []
      tasks.extend([fname]*iters)
    for envstr in self.envs:        # All envs start (and stay) inactive in parent process
      envstr.active = False
    while tasks or any(busy_envs):
      while available_envs and tasks:
        i = available_envs.pop(0)
        fname=tasks.pop(0)
        if self.debug:
          print('Starting {} on Env #{}'.format(fname,i))
        envstr = self.envs[i]
        envstr.fname = fname        # An UGLY HACK. This is for the parent process to also have the file name.
        if pipes[i]:
          poll.unregister(pipes[i][0])
          os.close(pipes[i][0])
        pipes[i] = os.pipe()    # reader, writer
        poll.register(pipes[i][0], select.POLLIN)
        # poll.register(pipes[i][0], select.POLLIN | select.POLLHUP)
        pid = os.fork()
        seed_idx += 1
        if not pid:     # child
          os.close(pipes[i][0])
          self.envs[i].active=True
          np.random.seed(int(time.time())+seed_idx)
          # envstr.env.restart_env(timeout=1)
          self.reset_env(envstr,fname=fname)
          finished_envs = []
          while not finished_envs:      # Just one (the ith) env is actually active and running
            finished_envs = self.step_all(model)
          finished = finished_envs[0][1]
          if finished:
            if self.debug:
              print('Finished {} on Env #{}'.format(fname,i))
            res = len(self.completed_episodes.pop(0))            
          else:
            print('Env {} took too long!'.format(fname,i))
            res = len(self.envs[i].episode_memory)
          os.write(pipes[i][1],str((i,res)).encode())
          os._exit(os.EX_OK)
        
        # Parent continues here
        os.close(pipes[i][1])
        busy_envs[i] = True
        pids[i]=pid

      # We are now most likely out of available solvers, so wait on the busy ones (Which are all until the very end)

      finished_envs = poll.poll()
      for fd, event in finished_envs:
        # print('Got event {}'.format(event))
        if event == select.POLLHUP:
          # print('Why do I get POLLHUP?')
          continue
        i, res = eval(os.read(fd,1000).decode())
        # print('Read the end of env {}'.format(i))
        busy_envs[i] = False
        available_envs.append(i)
        cleanup_process(pids[i])
        envstr = self.envs[i]
        fname = envstr.fname
        rc[fname].append(res)
        if ed is not None:
          ed.add_stat(fname,res)
        if len(rc[fname]) == iters:
          print('Finished {}, results are: {}, Average/Min are {}/{}'.format(fname,rc[fname],
            np.mean(rc[fname]),min(rc[fname])))
        
    return rc


