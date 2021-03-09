from pysat.solvers import Minisat22, Glucose3
from pysat.formula import CNF
from subprocess import Popen, PIPE, STDOUT
from collections import deque
from namedlist import namedlist
from scipy.sparse import csr_matrix
import select
from IPython.core.debugger import Tracer
import time
import logging
import pickle
import tracemalloc
import queue
import threading
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
from reduce_base_provider import *

LOG_SIZE = 200
DEF_STEP_REWARD = -0.01     # Temporary reward until Pash sends something from minisat
NUM_ACTIONS = 29
log = mp.get_logger()


CLABEL_LEARNED = 0
CLABEL_LBD = 3
CLABEL_LOCKED = 5



class SatActiveESEnv:
  EnvObservation = namedlist('SatESEnvObservation', 
                              ['state', 'vlabels', 'clabels', 'adj_arrays', 'reward', 'done'],
                              default=None)

  def __init__(self, debug=False, server=None, settings=None, oracletype=None, **kwargs):
    self.settings = settings if settings else CnfSettings()    
    self.debug = debug
    self.tail = deque([],LOG_SIZE)
    self.solver = None
    self.server = server
    self.current_step = 0
    self.oracletype = oracletype
    self.gc_freq = self.settings['sat_gc_freq']
    self.disable_gnn = self.settings['disable_gnn']
    self.cb_type = self.settings['sat_cb_type']
    self.trigger = self.settings['sat_trigger']
    self.trigger_freq = self.settings['sat_trigger_freq']
    self.formulas_dict = {}
    ProviderClass = eval(self.settings['sat_reduce_base_provider'])
    self.rb_provider = ProviderClass(self.settings)
    self._name = 'SatEnv'

  @property
  def name(self):
    return self._name
  
  # def load_formula(self, fname):
  #   if fname not in self.formulas_dict.keys():
  #     self.formulas_dict[fname] = CNF(fname)
  #     print('Lazily loaded {} in process {}_{}'.format(fname,self._name,os.getpid()))
  #   return self.formulas_dict[fname]

  def start_solver(self, fname=None):
    
    def thunk():      
      return self.__callback()


    if self.solver is not None:
      self.solver.delete()
      del self.solver

    kwargs = {}
    if self.cb_type == 'gc_oracle':
      sat_oracle = {"callback": thunk, "policy": self.oracletype}
      kwargs['reduce_base'] = self.rb_provider.get_reduce_base()
      kwargs['gc_freq'] = self.gc_freq
    elif self.cb_type == 'branching_oracle':
      sat_oracle = {"callback": thunk, "trigger": self.trigger, "trigger_freq": self.trigger_freq}
    else:
      assert False, 'Bad oracle type {}'.format(self.cb_type)
    kwargs[self.cb_type] = sat_oracle
    self.solver = Glucose3(**kwargs)
    self.current_step = 0
    if fname:
      try:
        f1 = self.settings.formula_cache.load_formula(fname)
        self.solver.append_formula(f1.clauses)
        del f1
        return True
      except:
        return False
    else:
      print('Got no filename!!')
      return False


  def get_adj_arrays(self):
    return self.solver.get_cl_arr()

  def get_vlabels(self):
    return self.solver.get_var_labels()
  
  def get_lit_labels(self):
    return self.solver.get_lit_labels()

  def get_clabels(self, clause_type='all'):
    return self.solver.get_cl_labels(clause_type)

  def get_global_state(self):
    if self.cb_type == 'branching_oracle':
      return None
    elif self.cb_type == 'gc_oracle':
      return self.solver.get_solver_state()
    else:
      assert False, 'Bad oracle type'


  def get_reward(self):
    return self.solver.reward()

  def __callback(self):
    self.current_step += 1
      # adj_matrix = csr_matrix((data_arr, (rows_arr, cols_arr)))
    gss = self.get_global_state()
    # clabels = self.get_clabels()
    clabels = None
    vlabels = self.get_lit_labels()   # Note, vlabels are actually -literal- labels  
    adj_arrays = self.get_adj_arrays()
    try:
      return self.server.callback(gss, vlabels, clabels, adj_arrays)
    except Exception as e:
      print('SatEnv: Gah, an exception: {}'.format(e))
      raise e

class SatESEnvProxy(EnvBase):
  def __init__(self, config):
    self.settings = config['settings']
    if not self.settings:
      self.settings = CnfSettings()
    self.state_dim = self.settings['state_dim']
    self.observation_space = TrivialSpace()
    self.action_space = spaces.Discrete(NUM_ACTIONS)      # Make a config or take it from somewhere
    self.queue_in = config['queue_in']
    self.queue_out = config['queue_out']
    self.provider = config['provider']
    self.current_step = 0
    self.orig_clabels = None
    self.rewards = None
    self.finished = False
    self.reward_scale = self.settings['sat_reward_scale']
    self.disable_gnn = self.settings['disable_gnn']
    self.sat_min_reward = self.settings['sat_min_reward']    
    self.max_step = self.settings['max_step']    
    self.logger = utils.get_logger(self.settings, 'SatESEnvProxy')

  def step(self, action):
    self.queue_out.put((EnvCommands.CMD_STEP,action))
    ack, rc = self.queue_in.get()  
    assert ack==EnvCommands.ACK_STEP, 'Expected ACK_STEP'
    env_obs = SatActiveESEnv.EnvObservation(*rc)
    self.finished = env_obs.done
    if env_obs.reward:      
      r = env_obs.reward / self.reward_scale
      self.rewards.append(r)    
    self.current_step += 1
    # if env_obs.done:
    #   print('Env returning DONE, number of rewards is {}'.format(len(self.rewards)))
    return env_obs, r, env_obs.done or self.check_break(), {}

  def reset(self, fname=None):
    if not fname:
      fname = self.provider.get_next()
    # print('reset: Got formula: {}'.format(fname))
    self.finished = False
    self.current_step = 0
    self.rewards = []
    self.queue_out.put((EnvCommands.CMD_RESET,fname))
    ack, rc = self.queue_in.get()
    assert ack==EnvCommands.ACK_RESET, 'Expected ACK_RESET'    
    if rc != None:
      return SatActiveESEnv.EnvObservation(*rc)

  def exit(self):
    self.queue_out.put((EnvCommands.CMD_EXIT,None))
    ack, rc = self.queue_in.get()
    assert ack==EnvCommands.ACK_EXIT, 'Expected ACK_EXIT'

    # For the gym interface, the env itself decides whether to abort.

  def check_break(self):
    if self.sat_min_reward:        
      return (sum(self.rewards) < self.sat_min_reward)
    else:
      return (self.current_step > self.max_step)

  def new_episode(self, fname, **kwargs):    
    return self.reset(fname)        

  def process_observation(self, last_obs, env_obs, settings=None):
    if not settings:
      settings = CnfSettings()

    if env_obs == None:
      return None
    self.orig_clabels = [] if self.disable_gnn else env_obs.orig_clause_labels
    if env_obs.orig_clauses is not None:
      self.orig_clauses = None if self.disable_gnn else csr_to_pytorch(env_obs.orig_clauses)
    learned_clauses = None if self.disable_gnn else csr_to_pytorch(env_obs.learned_clauses)
    cmat = None if self.disable_gnn else Variable(concat_sparse(self.orig_clauses,learned_clauses))
    all_clabels = torch.from_numpy(env_obs.clabels if self.disable_gnn else np.concatenate([self.orig_clabels,env_obs.clabels])).float()



    # Replace the first index of the clabels with a marker for orig/learned
    # Take log of clabels[:,4]

    all_clabels[:,0]=0
    all_clabels[-len(env_obs.clabels):,0]=1
    act = all_clabels[:,4]
    act += 10.
    act.log_()
    all_clabels[:,4] = act

    # Take log of vlabels[:,3] and clabels[:,4]
    if not self.disable_gnn:
      activities = env_obs.vlabels[:,3]+10
      env_obs.vlabels[:,3]=np.log(activities)
      vlabels = Variable(torch.from_numpy(env_obs.vlabels).float())   # Remove first (zero) row
    else:
      vlabels = Variable(torch.zeros(2,self.settings['vlabel_dim']))   # Add dummy variables to keep collate_batch happy
    clabels = Variable(all_clabels)
    vmask = last_obs.vmask if last_obs else None
    cmask = last_obs.cmask if last_obs else None
    state = self.settings.FloatTensor(env_obs.state).unsqueeze(0)
    num_orig_clauses = len(self.orig_clabels)
    num_learned_clauses = len(env_obs.clabels)

    return State(state,cmat, vlabels, clabels, vmask, cmask, (num_orig_clauses,num_orig_clauses+num_learned_clauses))

class SatESEnvServer(mp.Process if CnfSettings()['env_as_process'] else threading.Thread):
  def __init__(self, env, settings=None):
    super(SatESEnvServer, self).__init__()
    self.settings = settings if settings else CnfSettings()
    self.state_dim = self.settings['state_dim']    
    self.env = env
    self.is_process = self.settings['env_as_process']
    self.env.server = self
    self.queue_in = mp.Queue() if self.is_process else queue.Queue()
    self.queue_out = mp.Queue() if self.is_process else queue.Queue()
    self.cmd = None
    self.current_fname = None
    self.last_reward = 0
    self.last_orig_clause_size = 0
    self.do_lbd = self.settings['do_lbd']
    self.disable_gnn = self.settings['disable_gnn']
    if self.settings['sat_min_reward']:
      self.winning_reward = -self.settings['sat_min_reward']*self.settings['sat_reward_scale']*self.settings['sat_win_scale']
    else:
      self.winning_reward = self.settings['sat_winning_reward']*self.settings['sat_reward_scale']
    self.total_episodes = 0
    self.uncache_after_batch = self.settings['uncache_after_batch']
    self.logger = utils.get_logger(self.settings, 'SatEnvServer')    

  def proxy(self, **kwargs):
    config = kwargs
    config['queue_out'] = self.queue_in
    config['queue_in'] = self.queue_out
    return SatESEnvProxy(config)

  def run(self):
    print('Env {} on pid {}'.format(self.env.name, os.getpid()))
    set_proc_name(str.encode('{}_{}'.format(self.env.name,os.getpid())))
    # if self.settings['memory_profiling']:
    #   tracemalloc.start(25)    
    while True:
      # if self.settings['memory_profiling'] and (self.total_episodes % 10 == 1):    
      # if self.settings['memory_profiling']:
      #   snapshot = tracemalloc.take_snapshot()
      #   top_stats = snapshot.statistics('lineno')
      #   print("[ Top 20 in {}]".format(self.name))
      #   for stat in top_stats[:20]:
      #       print(stat)            
      #   print('Number of cached formulas: {}'.format(len(self.env.formulas_dict.keys())))
      #   print(self.env.formulas_dict.keys())


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
      if self.uncache_after_batch and fname != self.current_fname:
        self.settings.formula_cache.delete_key(self.current_fname)
      self.current_fname = fname

      # This call does not return until the episodes is done. Messages are going to be exchanged until then through
      # the __callback method

      if self.env.start_solver(fname):
        self.env.solver.solve()
        # print('Done with {} in {} steps'.format(fname,self.env.current_step))
      else:
        print('Skipping {}'.format(fname))

      if self.cmd == EnvCommands.CMD_STEP:
        last_step_reward = -(self.env.get_reward() - self.last_reward)      
        # We are here because the episode successfuly finished. We need to mark done and return the rewards to the client.
        msg = self.env.EnvObservation(state=np.zeros(self.state_dim), reward=self.winning_reward+last_step_reward, done=True)
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
        self.queue_out.put((EnvCommands.ACK_EXIT,None))
        break


  def callback(self, gss, vlabels, clabels, adj_arrays):
    msg = self.env.EnvObservation(gss, vlabels, clabels, adj_arrays, None, False)
    if self.cmd == EnvCommands.CMD_RESET:
      # if not self.disable_gnn:
      #   msg.orig_clauses = self.env.get_orig_clauses()
      self.last_reward = self.env.get_reward()
      ack = EnvCommands.ACK_RESET
    elif self.cmd == EnvCommands.CMD_STEP:
      last_reward = self.env.get_reward()
      msg.reward = -(last_reward - self.last_reward)
      # print('Got reward: {}'.format(msg.reward))
      self.last_reward = last_reward
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
      return 0
    elif self.cmd == EnvCommands.CMD_EXIT:
      print('Got CMD_EXIT')
      self.env.solver.terminate()
      return 0

