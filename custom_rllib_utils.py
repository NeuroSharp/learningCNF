import ipdb
import collections
import numpy as np

from ray.rllib.optimizers import AsyncGradientsOptimizer
from ray.rllib.evaluation.metrics import collect_episodes, summarize_episodes, _partition
from ray.rllib.models.torch.torch_action_dist import TorchCategorical
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID

class TorchCategoricalArgmax(TorchCategorical):
  def sample(self):
    return self.dist.logits.argmax().unsqueeze(0)

class CNFGradientsOptimizer(AsyncGradientsOptimizer):

  def collect_metrics(self,
                      timeout_seconds,
                      min_history=100,
                      selected_workers=None):
    """Returns worker and optimizer stats.

    Arguments:
        timeout_seconds (int): Max wait time for a worker before
            dropping its results. This usually indicates a hung worker.
        min_history (int): Min history length to smooth results over.
        selected_workers (list): Override the list of remote workers
            to collect metrics from.

    Returns:
        res (dict): A training result dict from worker metrics with
            `info` replaced with stats from self.
    """
    episodes, self.to_be_collected = collect_episodes(
      self.workers.local_worker(),
      selected_workers or self.workers.remote_workers(),
      self.to_be_collected,
      timeout_seconds=timeout_seconds)
    orig_episodes = list(episodes)
    missing = min_history - len(episodes)
    if missing > 0:
      episodes.extend(self.episode_history[-missing:])
      assert len(episodes) <= min_history
    self.episode_history.extend(orig_episodes)
    self.episode_history = self.episode_history[-min_history:]
    res = CNFGradientsOptimizer.summarize_episodes(episodes, orig_episodes)
    res.update(info=self.stats())
    return res

  def summarize_episodes(episodes, new_episodes):
    episodes, estimates = _partition(episodes)
    new_episodes, _ = _partition(new_episodes)

    episode_rewards = []
    episode_lengths = []
    policy_rewards = collections.defaultdict(list)
    custom_metrics = collections.defaultdict(list)
    perf_stats = collections.defaultdict(list)
    for episode in episodes:
      episode_lengths.append(episode.episode_length)
      episode_rewards.append(episode.episode_reward)
      for k, v in episode.custom_metrics.items():
        custom_metrics[k].append(v)
      for k, v in episode.perf_stats.items():
        perf_stats[k].append(v)
      for (_, policy_id), reward in episode.agent_rewards.items():
        if policy_id != DEFAULT_POLICY_ID:
          policy_rewards[policy_id].append(reward)
    if episode_rewards:
      min_reward = min(episode_rewards)
      max_reward = max(episode_rewards)
    else:
      min_reward = float("nan")
      max_reward = float("nan")
    avg_reward = np.mean(episode_rewards)
    avg_length = np.mean(episode_lengths)

    policy_reward_min = {}
    policy_reward_mean = {}
    policy_reward_max = {}
    for policy_id, rewards in policy_rewards.copy().items():
      policy_reward_min[policy_id] = np.min(rewards)
      policy_reward_mean[policy_id] = np.mean(rewards)
      policy_reward_max[policy_id] = np.max(rewards)

    for k, v_list in custom_metrics.copy().items():
      if v_list and np.shape(v_list[0]) and np.shape(v_list[0])[0] > 1:
        custom_metrics[k + "_mean"] = np.mean(v_list,axis=0)  
      else:
        custom_metrics[k + "_mean"] = np.mean(v_list)
        filt = [v for v in v_list if not np.isnan(v)]
        if filt:
          custom_metrics[k + "_min"] = np.min(filt)
          custom_metrics[k + "_max"] = np.max(filt)
        else:
          custom_metrics[k + "_min"] = float("nan")
          custom_metrics[k + "_max"] = float("nan")
      del custom_metrics[k]

    for k, v_list in perf_stats.copy().items():
      perf_stats[k] = np.mean(v_list)

    estimators = collections.defaultdict(lambda: collections.defaultdict(list))
    for e in estimates:
      acc = estimators[e.estimator_name]
      for k, v in e.metrics.items():
        acc[k].append(v)
    for name, metrics in estimators.items():
      for k, v_list in metrics.items():
        metrics[k] = np.mean(v_list)
      estimators[name] = dict(metrics)

    return dict(
      episode_reward_max=max_reward,
      episode_reward_min=min_reward,
      episode_reward_mean=avg_reward,
      episode_len_mean=avg_length,
      episodes_this_iter=len(new_episodes),
      policy_reward_min=policy_reward_min,
      policy_reward_max=policy_reward_max,
      policy_reward_mean=policy_reward_mean,
      custom_metrics=dict(custom_metrics),
      sampler_perf=dict(perf_stats),
      off_policy_estimator=dict(estimators))
