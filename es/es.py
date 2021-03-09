# Code in this file is copied and adapted from
# https://github.com/openai/evolution-strategies-starter.

from collections import namedtuple
import logging
import numpy as np
import time
import itertools
import ray
from ray.rllib.agents import Trainer, with_common_config

from ray.rllib.agents.es import optimizers
from es import policies
from es import es_utils
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.utils.annotations import override
from ray.rllib.utils.memory import ray_get_and_free
from ray.rllib.utils import FilterManager
from settings import *
from episode_data import *
from utils import OneTimeSwitch

logger = logging.getLogger(__name__)

Result = namedtuple("Result", [
  "noise_indices", "noisy_returns", "sign_noisy_returns", "noisy_lengths"
])

# yapf: disable
# __sphinx_doc_begin__
DEFAULT_CONFIG = with_common_config({
  "l2_coeff": 0.005,
  "noise_stdev": 0.02,
  "episodes_per_batch": 1000,
  "train_batch_size": 10000,
  "eval_prob": 0.003,
  "return_proc_mode": "centered_rank",
  "num_workers": 10,
  "stepsize": 0.01,             # NOTE: This overrides settings['init_lr'] !!!
  "observation_filter": "MeanStdFilter",
  "noise_size": 250000000,
  "report_length": 10,
})
# __sphinx_doc_end__
# yapf: enable


@ray.remote
def create_shared_noise(count):
  """Create a large array of noise to be shared by all workers."""
  seed = 123
  print("noise size {}".format(count))
  noise = np.random.RandomState(seed).randn(count).astype(np.float32)
  return noise


class SharedNoiseTable:
  def __init__(self, noise):
    self.noise = noise
    assert self.noise.dtype == np.float32

  def get(self, i, dim):
    return self.noise[i:i + dim]

  def sample_index(self, dim):
    return np.random.randint(0, len(self.noise) - dim + 1)


@ray.remote
class Worker:
  def __init__(self,
               config,
               policy_params,
               env_creator,
               noise,
               is_eval,
               min_task_runtime=0.2):
    self.min_task_runtime = min_task_runtime
    self.config = config
    self.settings = CnfSettings()
    self.settings.hyperparameters = config['env_config']['settings']
    self.is_eval = is_eval
    # self.train_uniform_items = OnePassProvider(self.settings['es_train_data']).items
    self.policy_params = policy_params
    self.noise = SharedNoiseTable(noise) if noise is not None else None
    self.env_creator = env_creator
    self.recreate_env = self.settings['recreate_env']
    if not self.recreate_env:
      self.env = env_creator(config["env_config"])    # Created once and for all for this worker
    if self.settings['solver'] == 'sharpsat':
      self.policy = policies.SharpPolicy()
    elif self.settings['solver'] == 'sat_es':
      self.policy = policies.SATPolicy()

  @property
  def filters(self):
    return {DEFAULT_POLICY_ID: self.policy.get_filter()}

  def sync_filters(self, new_filters):
    for k in self.filters:
      self.filters[k].sync(new_filters[k])

  def get_filters(self, flush_after=False):
    return_filters = {}
    for k, f in self.filters.items():
      return_filters[k] = f.as_serializable()
      if flush_after:
        f.clear_buffer()
    return return_filters

  def rollout(self, fnames, timestep_limit, add_noise=True, params=None):
    if params is not None:
      self.policy.set_weights(params)
    rews = []
    lens = []
    for fname in fnames:
      if self.recreate_env:
        self.env = self.env_creator(self.config["env_config"])
      rollout_rewards, rollout_length = policies.rollout(
        self.policy,
        self.env,        
        fname,
        timestep_limit=timestep_limit,
        add_noise=add_noise)
      rews.append(rollout_rewards)
      lens.append(rollout_length)
      if self.is_eval:
        print('Eval finished {} with {} steps'.format(fname,rollout_length))
      if self.recreate_env:
        self.env.exit()
    return rews, lens

  # def evaluate(params, fnames)

  def do_rollouts(self, params, fnames, timestep_limit=None):
    # Set the network weights.
    self.policy.set_weights(params)
    noise_indices, returns, sign_returns, lengths = [], [], [], []
    eval_returns, eval_lengths = [], []

    # Perform some rollouts with noise.
    task_tstart = time.time()
    while (len(noise_indices) == 0
       or time.time() - task_tstart < self.min_task_runtime):

      # Do a regular run with parameter perturbations.
      noise_index = self.noise.sample_index(self.policy.num_params)

      perturbation = self.config["noise_stdev"] * self.noise.get(
        noise_index, self.policy.num_params)

      # These two sampling steps could be done in parallel on
      # different actors letting us update twice as frequently.                
      self.policy.set_weights(params + perturbation)
      rewards_pos, lengths_pos = self.rollout(fnames, timestep_limit)
      rewards_pos = [np.sum(x) for x in rewards_pos]
      lengths_pos = np.sum(lengths_pos)

      self.policy.set_weights(params - perturbation)
      rewards_neg, lengths_neg = self.rollout(fnames, timestep_limit)
      rewards_neg = [np.sum(x) for x in rewards_neg]
      lengths_neg = np.sum(lengths_neg)

      noise_indices.append(noise_index)
      returns.append([np.sum(rewards_pos), np.sum(rewards_neg)])
      sign_returns.append(
        [np.sign(rewards_pos).sum(),
         np.sign(rewards_neg).sum()])
      lengths.append([lengths_pos, lengths_neg])

    return Result(
      noise_indices=noise_indices,
      noisy_returns=returns,
      sign_noisy_returns=sign_returns,
      noisy_lengths=lengths)

  def do_rollouts_dist(self, params, conf, timestep_limit=None):
    # Set the network weights.
    self.policy.set_weights(params)    
    fname = conf['fname']
    noise_index = conf['noise_id']

    perturbation = self.config["noise_stdev"] * self.noise.get(
      noise_index, self.policy.num_params)

    # These two sampling steps could be done in parallel on
    # different actors letting us update twice as frequently.                
    self.policy.set_weights(params + perturbation)
    rewards_pos, lengths_pos = self.rollout([fname], timestep_limit)
    rewards_pos = [np.sum(x) for x in rewards_pos]
    lengths_pos = np.sum(lengths_pos)

    self.policy.set_weights(params - perturbation)
    rewards_neg, lengths_neg = self.rollout([fname], timestep_limit)
    rewards_neg = [np.sum(x) for x in rewards_neg]
    lengths_neg = np.sum(lengths_neg)

    returns = [np.sum(rewards_pos), np.sum(rewards_neg)]
    lengths = [lengths_pos, lengths_neg]

    return conf, (returns,lengths)

class ESTrainer(Trainer):
  """Large-scale implementation of Evolution Strategies in Ray."""

  _name = "ES"
  _default_config = DEFAULT_CONFIG

  @override(Trainer)
  def _init(self, config, env_creator):
    self.settings = CnfSettings()
    self.num_to_sample = self.settings['es_num_formulas']
    pcls = eval(self.settings['episode_provider'])
    self.provider=pcls(self.settings['es_train_data'])

    policy_params = {"action_noise_std": 0.01}    

    if self.settings['solver'] == 'sharpsat':
      self.policy = policies.SharpPolicy()
    elif self.settings['solver'] == 'sat_es':
      self.policy = policies.SATPolicy()
    print('step size is: {}'.format(config["stepsize"]))
    self.optimizer = optimizers.Adam(self.policy, config["stepsize"])
    self.report_length = config["report_length"]

    # Create the shared noise table.
    logger.info("Creating shared noise table.")
    noise_id = create_shared_noise.remote(config["noise_size"])
    self.noise = SharedNoiseTable(ray.get(noise_id))

    # Create the actors.
    logger.info("Creating actors.")
    self._workers = [
      Worker.remote(config, policy_params, env_creator, noise_id, False)
      for _ in range(config["num_workers"])
    ]

    self.episodes_so_far = 0
    self.reward_list = []
    self.tstart = time.time()

  @override(Trainer)
  def _train(self):
    config = self.config
    theta = self.policy.get_weights()
    # assert theta.dtype == np.float32

    # Put the current policy weights in the object store.
    theta_id = ray.put(theta)
    # Use the actors to do rollouts, note that we pass in the ID of the
    # policy weights.
    # results, num_episodes, num_timesteps = self._collect_results(
    #   theta_id, config["episodes_per_batch"], config["train_batch_size"])
    results, num_episodes, num_timesteps = self._collect_results_dist(
      theta_id, config["episodes_per_batch"])

    all_noise_indices = []
    all_training_returns = []
    all_training_lengths = []

    # Loop over the results.
    for result in results:
      all_noise_indices += result.noise_indices
      all_training_returns += result.noisy_returns
      all_training_lengths += result.noisy_lengths
    
    assert (len(all_noise_indices) == len(all_training_returns) == len(all_training_lengths))

    self.episodes_so_far += num_episodes
    
    # Assemble the results.
    noise_indices = np.array(all_noise_indices)
    noisy_returns = np.array(all_training_returns)
    noisy_lengths = np.array(all_training_lengths)

    # Process the returns.    
    if config["return_proc_mode"] == "centered_rank":
      proc_noisy_returns = es_utils.compute_centered_ranks(noisy_returns)
    else:
      raise NotImplementedError(config["return_proc_mode"])

    # Compute and take a step.
    g, count = es_utils.batched_weighted_sum(
      proc_noisy_returns[:, 0] - proc_noisy_returns[:, 1],
      (self.noise.get(index, self.policy.num_params)
       for index in noise_indices),
      batch_size=500)
    g /= noisy_returns.size
    assert (g.shape == (self.policy.num_params, ) and g.dtype == np.float32
            and count == len(noise_indices))
    # Compute the new weights theta.
    theta, update_ratio = self.optimizer.update(-g +
                                                config["l2_coeff"] * theta)
    # Set the new weights in the local copy of the policy.
    self.policy.set_weights(theta)

    # Now sync the filters
    FilterManager.synchronize({
      DEFAULT_POLICY_ID: self.policy.get_filter()
    }, self._workers)

    info = {
      "weights_norm": np.square(theta).sum(),
      "grad_norm": np.square(g).sum(),
      "update_ratio": update_ratio,
      "episodes_this_iter": noisy_lengths.size,
      "episodes_so_far": self.episodes_so_far,
    }

    result = dict(
      timesteps_this_iter=noisy_lengths.sum(),
      info=info)

    return result

  @override(Trainer)
  def compute_action(self, observation):
    return self.policy.compute(observation)[0]

  @override(Trainer)
  def _stop(self):
    # workaround for https://github.com/ray-project/ray/issues/1516
    for w in self._workers:
      w.__ray_terminate__.remote()

  def _collect_results(self, theta_id, min_episodes, min_timesteps):
    num_episodes, num_timesteps = 0, 0
    results = []
    # do_eval = OneTimeSwitch()
    fnames = self.provider.sample(size=self.num_to_sample,replace=False)
    fnames_id = ray.put(fnames)
    while num_episodes < min_episodes or num_timesteps < min_timesteps:
      # logger.info(
      #     "Collected {} episodes {} timesteps so far this iter".format(
      #         num_episodes, num_timesteps))
      rollout_ids = [
        worker.do_rollouts.remote(theta_id,fnames_id) for worker in self._workers
      ]
      # Get the results of the rollouts.
      for result in ray_get_and_free(rollout_ids):
        results.append(result)
        # Update the number of episodes and the number of timesteps
        # keeping in mind that result.noisy_lengths is a list of lists,
        # where the inner lists have length 2.
        num_episodes += sum(len(pair) for pair in result.noisy_lengths)
        num_timesteps += sum(
          sum(pair) for pair in result.noisy_lengths)
    return results, num_episodes, num_timesteps

  def _collect_results_dist(self, theta_id, sample_size):
    num_episodes, num_timesteps = 0, 0
    results = []
    # do_eval = OneTimeSwitch()    
    fnames = self.provider.sample(size=self.num_to_sample,replace=False)
    # fnames_id = ray.put(fnames)
    all_sample_noise = [self.noise.sample_index(self.policy.num_params) for _ in range(sample_size)]
    confs = list(itertools.product(all_sample_noise,fnames))
    free_workers = list(range(len(self._workers)))
    all_results = {i: {} for i in all_sample_noise}
    rollout_results = []
  # Collect
    while confs or rollout_results:
      # Start new tasks if we have free workers
      while free_workers and confs:
        n_id, fname = confs.pop()
        w_id = free_workers.pop()
        w = self._workers[w_id]
        conf_id = ray.put({'noise_id': n_id, 'fname': fname, 'worker_id': w_id})
        rollout_results.append(w.do_rollouts_dist.remote(theta_id,conf_id))

      # Read anything ready into a dict and free the worker
      ready, _ = ray.wait(rollout_results,timeout=0.2)
      for obj in ready:
        conf, result = ray_get_and_free(obj)        
        rollout_results.remove(obj)
        free_workers.append(conf['worker_id'])
        # print('Saving result for filename {} and index {}'.format(conf['fname'], conf['noise_id']))
        all_results[conf['noise_id']][conf['fname']] = result
                
    # Average returns over different formulas/files

    for nid, val in all_results.items():      
      returns = np.array([x[0] for x in val.values()]).mean(axis=0).tolist()
      lengths = np.array([x[1] for x in val.values()]).sum(axis=0).tolist()      
      results.append(Result(noise_indices=[nid],
      noisy_returns=[returns],
      sign_noisy_returns=None,
      noisy_lengths=[lengths]))
      num_episodes += 2
      num_timesteps += sum(lengths)        

    return results, num_episodes, num_timesteps

  def get_weights(self):
    return self.policy.get_weights()

  def __getstate__(self):
    return {
      "weights": self.policy.get_weights(),
      "filter": self.policy.get_filter(),
      "episodes_so_far": self.episodes_so_far,
    }

  def __setstate__(self, state):
    self.episodes_so_far = state["episodes_so_far"]
    self.policy.set_weights(state["weights"])
    self.policy.set_filter(state["filter"])
    # FilterManager.synchronize({
    #   DEFAULT_POLICY_ID: self.policy.get_filter()
    # }, self._workers)
