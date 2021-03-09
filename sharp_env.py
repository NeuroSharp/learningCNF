import ipdb
from pysat.solvers import Minisat22, Glucose3, SharpSAT
from pysat.formula import CNF
from subprocess import Popen, PIPE, STDOUT
from collections import deque
from namedlist import namedlist
from scipy.sparse import csr_matrix
import select
import threading
import queue
from IPython.core.debugger import Tracer
import time
import logging
import pickle
import tracemalloc
import torch.multiprocessing as mp
import utils
import pysolvers
import traceback
from gym import *
from gym import spaces
from settings import *
from qbf_data import *
from envbase import *
from rl_types import *
from rl_utils import *
from reduce_base_provider import *

LOG_SIZE = 200
DEF_STEP_REWARD = -0.01     # Temporary reward until Pash sends something from minisat

class SharpSpace(gym.Space):
  def contains(self, x):
    return True

  @property
  def shape(self):
    return ()

  @shape.setter
  def shape(self, value):  
    pass

class SharpActiveEnv:
  EnvObservation = namedlist('SharpEnvObservation', 
                              ['gss', 'vfeatures', 'cfeatures', 'row', 'col', 'efeatures', 'reward', 'done', 'partial'],
                              default=None)

  def __init__(self, server=None, settings=None, **kwargs):
    self.settings = settings if settings else CnfSettings()        
    self.solver = None
    self.server = server
    self.current_step = 0    
    self.disable_gnn = self.settings['disable_gnn']        
    self.formulas_dict = {}
    self._name = 'SharpEnv'

  @property
  def name(self):
    return self._name
  
  # def load_formula(self, fname):
  #   if fname not in self.formulas_dict.keys():
  #     self.formulas_dict[fname] = CNF(fname)
  #     print('Lazily loaded {} in process {}_{}'.format(fname,self._name,os.getpid()))
  #   return self.formulas_dict[fname]

  def start_solver(self):    
    def thunk(row, col, data, vlabels, lit_stack):
      return self.__callback(row, col, data, vlabels, lit_stack)
    if self.solver is None:
      self.solver = SharpSAT(branching_oracle= {"branching_cb": thunk})
    else:
      self.solver.delete()
      self.solver.new(branching_oracle= {"branching_cb": thunk})
    self.current_step = 0
    return True

  def __callback(self, row, col, data, vlabels, lit_stack):
    self.current_step += 1
    if not self.server:
      log.info('Running a test version of SharpEnv')
      ind = np.argmax(vlabels[:,1])
      pick = ind + 1 if (vlabels[ind][0] < vlabels[ind + 1][0]) else ind
      return pick
      
    else:
      try:
        rc = self.server.callback(vlabels, None, row, col, data, lit_stack)
        return rc
      except Exception as e:
        print('SharpEnv: Gah, an exception: {}'.format(e))
        raise e

class SharpEnvProxy(EnvBase):
  def __init__(self, config):
    self.settings = config['settings']
    if not self.settings:
      self.settings = CnfSettings()
    self.state_dim = self.settings['state_dim']
    self.decode = self.settings['sharp_decode']    
    self.max_time = self.settings['sharp_max_time']
    self.observation_space = SharpSpace()
    self.action_space = spaces.Discrete(self.settings['max_variables'])
    self.queue_in = config['queue_in']
    self.queue_out = config['queue_out']
    self.provider = config['provider']
    self.rewards = []
    self.current_step = 0
    self.finished = False    
    self.start_time = None
    self.completion_reward = self.settings['sharp_completion_reward']
    self.max_step = self.settings['max_step']    
    self.disable_gnn = self.settings['disable_gnn']
    self.logger = utils.get_logger(self.settings, 'SharpEnvProxy')


  def process_observation(self, last_obs, env_obs):
    if not env_obs or env_obs.done:
      return EmptyDenseState

    vfeatures = env_obs.vfeatures
    efeatures = env_obs.efeatures
    row = env_obs.row
    col = env_obs.col    
    cmat = csr_matrix((torch.ones(len(efeatures)), (row, col)),shape=(row.max()+1,len(vfeatures)))
    cmat = csr_to_pytorch(cmat)
    ground_embs = torch.from_numpy(vfeatures.to_numpy()[:, 1:]).float()
    extra_data = torch.from_numpy(efeatures).float(), env_obs.partial
    vmask = None
    cmask = None

    return densify_obs(State(None,cmat, ground_embs, None, vmask, cmask, extra_data))


  def step(self, action):    
    self.queue_out.put((EnvCommands.CMD_STEP,action))
    ack, rc = self.queue_in.get()  
    assert ack==EnvCommands.ACK_STEP, 'Expected ACK_STEP'
    env_obs = SharpActiveEnv.EnvObservation(*rc)
    self.finished = env_obs.done
    if env_obs.reward:
      self.rewards.append(env_obs.reward)
    self.current_step += 1
    # if env_obs.done:
    #   print('Env returning DONE, number of rewards is {}'.format(len(self.rewards)))
    return self.process_observation(None,env_obs), env_obs.reward, env_obs.done or self.check_break(), {}

  def reset(self, fname=None):
    if not fname:
      fname = self.provider.get_next()
    # print('reset: Got formula: {}'.format(fname))
    self.finished = False
    self.current_step = 0
    self.rewards = []
    self.start_time = time.time()
    self.queue_out.put((EnvCommands.CMD_RESET,fname))
    ack, rc = self.queue_in.get()
    assert ack==EnvCommands.ACK_RESET, 'Expected ACK_RESET'    
    if rc != None:
      return self.process_observation(None, SharpActiveEnv.EnvObservation(*rc))

  def exit(self):
    self.queue_out.put((EnvCommands.CMD_EXIT,None))
    ack, rc = self.queue_in.get()
    assert ack==EnvCommands.ACK_EXIT, 'Expected ACK_EXIT'

    # For the gym interface, the env itself decides whether to abort.

  def check_break(self):    
    return (self.current_step > self.max_step) or (time.time() - self.start_time) > self.max_time

  def new_episode(self, fname, **kwargs):    
    return self.reset(fname)        

class SharpEnvServer(mp.Process if CnfSettings()['env_as_process'] else threading.Thread):
  def __init__(self, env, settings=None):
    super(SharpEnvServer, self).__init__()
    self.settings = settings if settings else CnfSettings()
    self.state_dim = self.settings['state_dim']    
    self.decode = self.settings['sharp_decode']    
    self.env = env
    self.is_process = self.settings['env_as_process']    
    self.env.server = self
    self.queue_in = mp.Queue() if self.is_process else queue.Queue()
    self.queue_out = mp.Queue() if self.is_process else queue.Queue()
    self.cmd = None
    self.current_fname = None
    self.last_reward = 0
    self.last_orig_clause_size = 0
    self.disable_gnn = self.settings['disable_gnn']
    self.winning_reward = self.settings['sharp_completion_reward']
    self.total_episodes = 0
    self.def_step_cost = self.settings['def_step_cost']
    self.uncache_after_batch = self.settings['uncache_after_batch']
    self.logger = utils.get_logger(self.settings, 'SharpEnvServer')    

  def proxy(self, **kwargs):
    config = kwargs
    config['queue_out'] = self.queue_in
    config['queue_in'] = self.queue_out
    return SharpEnvProxy(config)

  def run(self):
    # print('Env {} on pid {}'.format(self.env.name, os.getpid()))
    while True:
      if self.cmd == EnvCommands.CMD_RESET:
        # We get here only after a CMD_RESET aborted a running episode and requested a new file.
        fname = self.current_fname        
      else:
        self.cmd, fname = self.queue_in.get()
        if self.cmd == EnvCommands.CMD_EXIT:
          # print('Got CMD_EXIT 1')
          self.queue_out.put((EnvCommands.ACK_EXIT,None))
          break
        assert self.cmd == EnvCommands.CMD_RESET, 'Unexpected command {}'.format(self.cmd)
      if self.uncache_after_batch and  fname != self.current_fname:
        self.settings.formula_cache.delete_key(self.current_fname)
      self.current_fname = fname

      # This call does not return until the episodes is done. Messages are going to be exchanged until then through
      # the __callback method

      t1 = time.time()
      if self.env.start_solver():
        self.env.solver.solve(fname)
      else:
        print('Skipping {}'.format(fname))
      if self.settings['sharp_log_actions']:
        actions = self.env.solver.get_branching_seq()
        stats = self.env.solver.get_stats()
        units = self.env.solver.get_problem_units()

        with open('eval_logs/{}_episode_log_{}.pickle'.format(self.settings['name'],os.path.basename(fname)),'wb') as f:
          pickle.dump({'actions': actions, 'stats': stats, 'units': units, 'time': time.time()-t1}, f)

      # print('Solver finished in {}'.format(time.time()-t1))
      if self.cmd == EnvCommands.CMD_STEP:
        last_step_reward = self.def_step_cost     
        # We are here because the episode successfuly finished. We need to mark done and return the rewards to the client.
        msg = self.env.EnvObservation(gss=np.zeros(self.state_dim), reward=self.winning_reward+last_step_reward, done=True)
        # msg = self.env.EnvObservation(None, None, None, None, None, None, self.winning_reward+last_step_reward, True)
        self.queue_out.put((EnvCommands.ACK_STEP,tuple(msg)))
        self.total_episodes += 1

      elif self.cmd == EnvCommands.CMD_RESET:
        if self.env.current_step == 0:
          print('Degenerate episode on {}'.format(fname))
          self.cmd = None
          self.queue_out.put((EnvCommands.ACK_RESET,None))
          # This is a degenerate episodes with no GC
        else:
          pass
          # We are here because the episode was aborted. We can just move on, the client already has everything.
      elif self.cmd == EnvCommands.CMD_EXIT:
        # print('Got CMD_EXIT 2')
        self.queue_out.put((EnvCommands.ACK_EXIT,None))
        break

# adj_matrix = csr_matrix((torch.ones(len(data)), (row, col)),shape=(row.max()+1,len(vlabels)))
  def callback(self, vfeatures, cfeatures, row, col, efeatures, lit_stack):
    # print('clabels shape: {}'.format(cfeatures.shape))    
    # print('reward is {}'.format(self.env.get_reward()))
    partial = None
    if self.decode:
      units = self.env.solver.get_problem_units()
      partial = np.concatenate([units,lit_stack])
    msg = self.env.EnvObservation(None, vfeatures, cfeatures, row, col, efeatures, self.def_step_cost, False, partial)
    if self.cmd == EnvCommands.CMD_RESET:
      ack = EnvCommands.ACK_RESET
    elif self.cmd == EnvCommands.CMD_STEP:
      ack = EnvCommands.ACK_STEP
    elif self.cmd == EnvCommands.CMD_EXIT:
      print('self.cmd is CMD_EXIT, yet we are in the callback again!')
      return 0
    else:
      assert True, 'Invalid last command detected'

    self.queue_out.put((ack,tuple(msg)))
    self.cmd, rc = self.queue_in.get()    
    if self.cmd == EnvCommands.CMD_STEP:
      # We got back an action
      return rc
    elif self.cmd == EnvCommands.CMD_RESET:
      # We were asked to abort the current episode. Notify the solver and continue as usual
      if self.uncache_after_batch and rc != self.current_fname:
        self.settings.formula_cache.delete_key(self.current_fname)
      self.current_fname = rc
      self.env.solver.terminate()
      return 0      # This is just an integer, any integer. Solver is going to terminate anyhow.
    elif self.cmd == EnvCommands.CMD_EXIT:
      # print('Got CMD_EXIT')
      self.env.solver.terminate()
      return 0

