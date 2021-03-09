# Code in this file is copied and adapted from
# https://github.com/openai/evolution-strategies-starter.
import ipdb
import gym
import numpy as np
import torch
import ray
import ray.experimental.tf_utils
from ray.rllib.evaluation.sampler import _unbatch_tuple_actions
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.filter import get_filter, NoFilter
from ray.util.sgd.utils import TimerStat
from custom_rllib_utils import *
from rllib_sharp_models import SharpModel
from rllib_sat_models import SatActivityModel
from clause_model import ClausePredictionModel
from settings import *
from graph_utils import *

def rollout(policy, env, fname, timestep_limit=None, add_noise=False):
  """Do a rollout.

  If add_noise is True, the rollout will take noisy actions with
  noise drawn from that stream. Otherwise, no action noise will be added.
  """
  timers = {k: TimerStat() for k in ["reset", "compute", "step"]}
  policy.model.curr_fname = fname           # This is an ugly hack for the extra high level information awareness
  env_timestep_limit = policy.settings['max_step']+10
  timestep_limit = (env_timestep_limit if timestep_limit is None else min(
    timestep_limit, env_timestep_limit))
  rews = []
  t = 0
  with timers['reset']:
    observation = env.reset(fname=fname)
  for _ in range(timestep_limit or 999999):
    with timers['compute']:
      ac = -1 if policy.settings['es_vanilla_policy'] else policy.compute(observation)[0]
    with timers['step']:
      observation, rew, done, _ = env.step(ac)
    rews.append(rew)
    t += 1
    if done:
      break
  rews = np.array(rews, dtype=np.float32)  
  return rews, t


class TorchGNNPolicy:
  def __init__(self, model):
    self.settings = CnfSettings()
    self.model = model
    self.num_params = np.sum([np.prod(x.shape) for x in self.model.parameters()])
    self.observation_filter = NoFilter()

  def get_weights(self):
    return self.get_flat_weights()

  def get_flat_weights(self):
    pre_flat = {k: v.cpu() for k, v in self.model.state_dict().items()}
    rc = torch.cat([v.view(-1) for k,v in self.model.state_dict().items()],dim=0)
    return rc.numpy()

  def set_weights(self, weights):
    curr_weights = torch.from_numpy(weights)
    curr_dict = self.model.state_dict()
    for k,v in curr_dict.items():
      total = np.prod(v.shape)      
      curr_dict[k] = curr_weights[:total].view_as(v)
      curr_weights = curr_weights[total:]

    self.model.load_state_dict(curr_dict)

  def get_filter(self):
    return self.observation_filter

  def set_filter(self, observation_filter):
    self.observation_filter = observation_filter

class SharpPolicy(TorchGNNPolicy):
  def __init__(self):
    super(SharpPolicy, self).__init__(SharpModel())
    self.time_hack = self.settings['sharp_time_random']

  def compute(self, observation):
    if self.settings['sharp_random_policy']:
      return [np.random.randint(observation.ground.shape[0])]
    logits, _ = self.model(observation, state=None, seq_lens=None)
    l = logits.detach().numpy()
    if self.time_hack:
      indices = np.where(l.reshape(-1)==l.max())[0]
      action = [np.random.choice(indices)]
    else:
      dist = TorchCategoricalArgmax(logits, self.model)    
      action = dist.sample().numpy()
    return action


class SATPolicy(TorchGNNPolicy):
  def __init__(self):
    super(SATPolicy, self).__init__(SatActivityModel())

# Returning logits for learnt clauses
  def transform(sample):
    G = sample['graph']
    G.nodes['literal'].data['literal_feats'][:,1] = 2*G.nodes['literal'].data['literal_feats'][:,1] / G.nodes['literal'].data['literal_feats'].shape[0]
    
    return sample

  def compute(self, observation):
    input_dict = {
      'gss': observation.state,
      'graph': graph_from_arrays(observation.vlabels,observation.clabels,observation.adj_arrays)
    }
    scores, _ = self.model(SATPolicy.transform(input_dict), state=None, seq_lens=None)
    return scores.detach().numpy()
