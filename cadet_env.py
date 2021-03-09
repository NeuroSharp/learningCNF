import ipdb
import select
import gym
import time
from subprocess import Popen, PIPE, STDOUT
from collections import deque
from collections import namedtuple
from gym import spaces
from IPython.core.debugger import Tracer
from settings import *
from qbf_data import *
from cadet_utils import *
from envbase import *
from rl_utils import *

DEF_GREEDY_ALPHA = 0.01
MAX_EPISODE_LENGTH = 200
DQN_DEF_COST = 1e-4
# DQN_DEF_COST = 0.
BINARY_SUCCESS = 1.
LOG_SIZE = 100
def require_init(f, *args, **kwargs): 
  def inner(instance, *args, **kwargs):
    assert(instance.cadet_proc != None)
    return f(instance,*args,**kwargs)
  return inner

class CadetSpace(gym.Space):
  def contains(self, x):
    return True

  @property
  def shape(self):
    return ()
    
  @shape.setter
  def shape(self, value):  
    pass


# Cadet actions are 1-based. The CadetEnv exposes 0-based actions
    
class CadetEnv(EnvBase):
  def __init__(self, provider=None, **kwargs):
    self.EnvObservation = namedtuple('EnvObservation', 
                    ['state', 'vars_add', 'vars_remove', 'activities', 'decision', 'clause', 
                      'reward', 'vars_set', 'done'])

    self.settings = CnfSettings()
    self.provider = provider
    self.cadet_binary = self.settings['cadet_binary']
    self.debug = self.settings['debug']
    self.qbf = QbfBase(**kwargs)
    self.observation_space = CadetSpace()
    self.action_space = spaces.Discrete(self.settings['max_variables'])
    self.greedy_rewards = self.settings['greedy_rewards']
    self.clause_learning = self.settings['clause_learning']
    self.vars_set = self.settings['vars_set']
    self.fresh_seed = self.settings['fresh_seed']
    self.use_old_rewards = self.settings['use_old_rewards']
    self.use_vsids_rewards = self.settings['use_vsids_rewards']
    self.slim_state = self.settings['slim_state']
    self.def_step_cost = self.settings['def_step_cost']
    self.cadet_completion_reward = self.settings['cadet_completion_reward']
    self.max_step = self.settings['max_step']
    self.restart_solver_every = int(self.settings['restart_solver_every'])
    # self.logger = logger
    self.greedy_alpha = DEF_GREEDY_ALPHA if self.greedy_rewards else 0.    
    self.tail = deque([],LOG_SIZE)
    self.reset_counter = 0
    self.use_activities = self.settings['cadet_use_activities']
    self.last_processed_obs = None
    self.start_cadet()
    

  def start_cadet(self):
    cadet_params = ['--rl', '--cegar', '--sat_by_qbf', '--rl_reward_per_decision', '{}'.format(self.def_step_cost), 
                      '--rl_completion_reward', '{}'.format(self.cadet_completion_reward)]
    if not self.use_old_rewards:
      if self.debug:
        print('Using new rewards!')
      cadet_params.append('--rl_advanced_rewards')
    if self.fresh_seed:
      if self.debug:
        print('Using fresh seed!')
      cadet_params.append('--fresh_seed')
    if self.slim_state:
      if self.debug:
        print('Using slim_state!')
      cadet_params.append('--rl_slim_state')
    if self.use_vsids_rewards:
      if self.debug:
        print('Using vsids rewards!!')
      cadet_params.append('--rl_vsids_rewards')

    if self.debug:
      print(' '.join([self.cadet_binary,  *cadet_params]))
    self.cadet_proc = Popen([self.cadet_binary,  *cadet_params], stdout=PIPE, stdin=PIPE, stderr=STDOUT, universal_newlines=True)
    self.poll_obj = select.poll()
    self.poll_obj.register(self.cadet_proc.stdout, select.POLLIN)  
    self.finished = False
    self.done = True      
    self.current_fname = None

  def stop_cadet(self, timeout):
    assert(self.cadet_proc != None)
    self.cadet_proc.terminate()
    time.sleep(timeout)
    if self.cadet_proc.poll() != None:
      self.cadet_proc.kill()
    self.cadet_proc = None
    self.poll_obj = None

  def restart_env(self, timeout=5):
    if self.debug:
      print('Stopping cadet...')
    self.stop_cadet(timeout)
    if self.debug:
      print('Restarting cadet...')
    self.start_cadet()

  def eat_initial_output(self):
    self.read_line_with_timeout()
    self.read_line_with_timeout()


  def write(self, val):    
    self.cadet_proc.stdin.write(val)
    self.cadet_proc.stdin.flush()


  def terminate(self):
    if not self.done:
      if self.debug:
        print('interrupting mid-episode!')
      self.write_action(-1)
      a = self.read_line_with_timeout()
      if self.debug:
        print(a)
      self.done = True
      rewards = np.asarray(list(map(float,a.split()[1:])))
      return rewards

  def exit(self):
    print('exit()')
    self.terminate()
    self.stop_cadet(5)

  def reset(self):
    self.reset_counter += 1
    if self.restart_solver_every > 0 and (self.settings['restart_in_test'] or (self.reset_counter % self.restart_solver_every == 0)):
      self.restart_env(timeout=0)

    self.terminate()
    fname = self.provider.get_next()
    if self.debug:
      print('Starting Env {}'.format(fname))
    # if fname == 'data/huge_gen1/small-bug1-fixpoint-3.qdimacs':
    #   Tracer()()
    self.qbf.reload_qdimacs(fname)    # This holds our own representation of the qbf graph
    self.vars_deterministic = np.zeros(self.qbf.num_vars)
    self.total_vars_deterministic = np.zeros(self.qbf.num_vars)    
    self.activities = np.zeros(self.qbf.num_vars)
    self.max_rewards = self.qbf.num_existential
    self.timestep = 0
    self.finished = False
    self.running_reward = []
    self.rewards = None

    self.write(fname+'\n')
    self.done = False
    self.current_fname = fname
    rc = self.read_state_update()     # Initial state
    self.last_processed_obs = self.process_observation(None, rc)
    return self.last_processed_obs
    


  def read_line_with_timeout(self, timeout=10.):
    return self.cadet_proc.stdout.readline()
    # entry = time.time()
    # curr = entry
    # line = ''
    # while curr < entry + timeout:
    #   p = self.poll_obj.poll(0) 
    #   if p:
    #     c = self.cadet_proc.stdout.read(1)
    #     if c == '\n':
    #       # Tracer()()
    #       return line
    #     line += c
    #   else:
    #     # print('Poll negative..')
    #     pass
    #   curr = time.time()

    # return None

  # This is where we go from 0-based to 1-based
  def write_action(self, a):
    if self.debug:
      print('Writing action {}'.format(a))
    if a in ['?','r']:
      self.write('{}\n'.format(a))
      return
    if type(a) is tuple:
      cadet_action = int(a[0]) + 1
      if a[1]:
        cadet_action = -cadet_action
    else:
      cadet_action = int(a)+1
    self.write('%d\n' % cadet_action)


  '''
  Returns:

    state - A bunch of numbers describing general solver state
    candidates - an array of (0-based) available actions in the current state
    done - Is this the end of the episode?

  '''
  def read_state_update(self):
    self.vars_deterministic.fill(0)
    self.activities.fill(0)
    clause = None
    reward = None
    decision = None
    vars_set = []
    while True:
      a = self.read_line_with_timeout()
      if not a or a == '\n': continue
      self.tail.append(a)
      if self.debug:
        print(a)
      if False and a == 'UNSAT\n':
        if self.cadet_binary != './cadet':
          a = self.read_line_with_timeout()     # refutation line
        a = self.read_line_with_timeout()     # rewards
        self.rewards = np.asarray(list(map(float,a.split()[1:])))
        if np.isnan(self.rewards).any():
          if np.isnan(self.rewards[:-1]).any():
            Tracer()()
          else:
            self.rewards[-1]=BINARY_SUCCESS
        self.done = True
        state = None
        break
      elif False and a == 'SAT\n':
        Tracer()()
        a = self.read_line_with_timeout()     # rewards
        self.rewards = np.asarray(list(map(float,a.split()[1:])))                
        if np.isnan(self.rewards).any():
          if np.isnan(self.rewards[:-1]).any():
            Tracer()()
          else:
            self.rewards[-1]=BINARY_SUCCESS
        self.done = True
        state = None
        break
      elif a.startswith('rewards') or a.startswith('SATrewards') or a.startswith('UNSATrewards'):
        self.rewards = np.asarray(list(map(float,a.split()[1:])))
        # if self.debug:
        #   Tracer()()
        if np.isnan(self.rewards).any():
          if np.isnan(self.rewards[:-1]).any():
            Tracer()()
          else:
            self.rewards[-1]=BINARY_SUCCESS
        self.done = True
        self.finished = True
        state = None
        if self.debug:
          print('Successfuly finished episode in {} steps!'.format(self.timestep))
        break
      elif a[0] == 'u' and a[1] != 'c':   # New cadet has 'uc'
        update = int(a[3:])-1     # Here we go from 1-based to 0-based
        if a[1] == '+':
          self.vars_deterministic[update] = 1
          self.total_vars_deterministic[update] = 1
        elif a[1] == '-':
          self.vars_deterministic[update] = -1
          self.total_vars_deterministic[update] = 0
      elif a.startswith('delete_clause'):
        if not self.clause_learning:
          continue
        cid = int(a.split()[1])
        # print('Removing clause id {}'.format(cid))
        # print(a)
        self.qbf.remove_clause(cid)
        clause = True
      elif a[0] == 'd':
        decision = [int(x) for x in a[2:].split(',')]
        decision[0] -= 1
      elif a[0] == 's':
        state = np.array([float(x) for x in a[2:].split(',')])
        if self.debug:
          print('Got state!')
        break
      elif a[0] == 'a':
        b = a[2:].split(',')
        update = int(b[0])-1
        activity = float(b[1])
        self.activities[update] = activity
      elif a[0] == 'v' and self.vars_set:
        v, pol = a.split(' ')[1:]        
        vars_set.append((int(v)-1,int(pol)))
      elif self.timestep > 0 and a[0] == 'c':
        if a.startswith('conflict'):
          continue
        elif a.startswith('clause'):      # new cadet version
          if not self.clause_learning:
            continue
          c = a.split()
          cid = int(c[1])
          b = [int(x) for x in c[4:]]
          self.qbf.add_clause(b,cid)
          # print('Adding clause id {}'.format(cid))
          # print(a)
        else:
          print('This version is too old')
          Tracer()()
          b = [int(x) for x in a[2:].split()]        
          
        # clause = (np.array([abs(x)-1 for x in b if x > 0]), np.array([abs(x)-1 for x in b if x < 0]))
        clause = True
      elif self.debug:
        print('Got unprocessed line: %s' % a)
        if a.startswith('Error'):
          return

    if self.timestep > 0:      
      greedy_reward = np.count_nonzero(self.total_vars_deterministic) - self.last_total_determinized
      self.running_reward.append(greedy_reward)
      reward = self.def_step_cost  + (self.cadet_completion_reward if self.done else 0.)
      if self.greedy_rewards:
        reward += self.greedy_alpha*self.running_reward[-1]
    self.last_total_determinized = np.count_nonzero(self.total_vars_deterministic)
    # if self.timestep > 200:
    #   self.debug = True
    if sum(self.running_reward) < -1:
      Tracer()()
    if self.done:
      self.rewards = self.rewards + self.greedy_alpha*np.asarray(self.running_reward)

    pos_vars = np.where(self.vars_deterministic>0)[0]
    neg_vars = np.where(self.vars_deterministic<0)[0]      

    return self.EnvObservation(state, pos_vars, neg_vars, self.activities, decision, clause, reward, np.array(vars_set), self.done)

  # This gets a tuple from stepping the environment:
  # state, vars_add, vars_remove, activities, decision, clause, reward, done = env.step(action)
  # And it returns the next observation.

  def process_observation(self, last_obs, env_obs):
    # import ipdb
    # ipdb.set_trace()
    
    if not env_obs or env_obs.done:
      return EmptyDenseState
    if env_obs.clause or not last_obs:
      # Tracer()()
      cmat = get_input_from_qbf(self.qbf, self.settings, False) # Do not split
      clabels = Variable(torch.from_numpy(self.qbf.get_clabels()).float().unsqueeze(0)).t()
    else:
      last_obs = undensify_obs(last_obs)
      cmat, clabels = last_obs.cmat, last_obs.clabels
    if last_obs:
      ground_embs = np.copy(last_obs.ground.data.numpy().squeeze())
      vmask = last_obs.vmask
      cmask = last_obs.cmask
    else:      
      ground_embs = self.qbf.get_base_embeddings()
      vmask = None
      cmask = None
    if env_obs.decision:
      ground_embs[env_obs.decision[0]][IDX_VAR_POLARITY_POS+1-env_obs.decision[1]] = True
    if len(env_obs.vars_add):
      ground_embs[:,IDX_VAR_DETERMINIZED][env_obs.vars_add] = True
    if len(env_obs.vars_remove):
      ground_embs[:,IDX_VAR_DETERMINIZED][env_obs.vars_remove] = False
      ground_embs[:,IDX_VAR_POLARITY_POS:IDX_VAR_POLARITY_NEG][env_obs.vars_remove] = False
    if self.use_activities:
      ground_embs[:,IDX_VAR_ACTIVITY] = env_obs.activities
    if len(env_obs.vars_set):
      a = env_obs.vars_set
      idx = a[:,0][np.where(a[:,1]==1)[0]]
      ground_embs[:,IDX_VAR_SET_POS][idx] = True
      idx = a[:,0][np.where(a[:,1]==-1)[0]]
      ground_embs[:,IDX_VAR_SET_NEG][idx] = True
      idx = a[:,0][np.where(a[:,1]==0)[0]]
      ground_embs[:,IDX_VAR_SET_POS:IDX_VAR_SET_NEG][idx] = False  

    state = Variable(torch.from_numpy(env_obs.state).float().unsqueeze(0))
    ground_embs = Variable(torch.from_numpy(ground_embs).float().unsqueeze(0))
    # ipdb.set_trace()
    return densify_obs(State(state,cmat, ground_embs, clabels, vmask, cmask, None))
    


# This returns already a State (higher-level) observation, not EnvObs.

  def new_episode(self, fname, settings=None, **kwargs):
    try:
      obs = self.reset(fname)
      # state, vars_add, vars_remove, activities, _, _ , _, vars_set, _ = self.reset(fname)
      if obs.state is not None:   # Env solved in 0 steps
        return obs
    except:
      print('Error reseting with file {}'.format(fname))


  def check_break(self):
    return self.timestep >= self.max_step

  def translate_action(self, action):
    try:
      if action in ['?']:
        return action
    except:
      pass
    return (int(action/2),int(action%2))


  def step(self, action):
    assert(not self.done)
    # if self.greedy_rewards and self.timestep > MAX_EPISODE_LENGTH:      
    #   rewards = self.terminate() + self.greedy_alpha*np.asarray(self.running_reward)
    #   self.rewards = np.concatenate([rewards, [DQN_DEF_COST]])    # Average action
    #   return None, None, None, None, None, None, DQN_DEF_COST, None, True    
    self.timestep += 1
    # print('Got action {}'.format(action))
    # ipdb.set_trace()
    self.write_action(self.translate_action(action))
    env_obs = self.read_state_update()
    obs = self.process_observation(self.last_processed_obs, env_obs)
    self.last_processed_obs = obs
    rc = obs, env_obs.reward, env_obs.done or self.check_break(), {'fname': self.current_fname}
    return rc
            
