from config import *
from settings import *
import ipdb
import os
import numpy as np
import pickle
import itertools
import argparse
import pandas as pd
import plotly.express as px
import torch.multiprocessing as mp
import es
import time
from IPython.core.debugger import Tracer
from collections import Counter
from matplotlib import pyplot as plt
from ray.tune.registry import register_env
# import ray.rllib.agents.ppo as ppo
from ray.rllib.agents import a3c
from ray.rllib.agents.a3c.a3c_torch_policy import *
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.evaluation.rollout_worker import *
from ray.rllib.evaluation.worker_set import *
from ray.tune.logger import *
from ray.tune.result import TIMESTEPS_TOTAL
from ray.rllib.utils.numpy import sigmoid, softmax
from ray.rllib.utils.memory import ray_get_and_free

from custom_rllib_utils import *
from dispatcher import *
from episode_data import *
from episode_reporter import *
from policy_factory import *
from test_envs import *
from rllib_sat_models import *
from rllib_cadet_models import *
from rllib_sharp_models import *


# In[2]:
steps_counter = 0
settings = CnfSettings(cfg())

def log_logits_policy_wrapper(policy):
  def add_logits(policy, input_dict, state_batches, model, action_dist):
    logits, _ = model(input_dict, state_batches, [1])
    probs = softmax(logits.cpu().numpy())
    return {SampleBatch.VF_PREDS: model.value_function().cpu().numpy(), 'probs': probs}
  return policy.with_updates(extra_action_out_fn=add_logits, name='AddLogitsA3CPolicy')

def get_settings_from_file(fname):
  conf = load_config_from_file(fname)
  for (k,v) in conf.items():
    settings.hyperparameters[k]=v

def make_cnf_optimizer(workers, config):
    return CNFGradientsOptimizer(workers, **config["optimizer"])

def on_train_result(info):
  custom_metrics = info['result']['custom_metrics']
  probs = custom_metrics.pop('probs_mean')
  logger = info['trainer']._result_logger._loggers[-1]
  steps = info['result'].get(TIMESTEPS_TOTAL)
  hist_data = (probs*2000).astype(int)
  dummy_data = []
  for idx, value in enumerate(hist_data):
    dummy_data += [idx + 2.001] * value
  values = np.array(dummy_data).astype(float).reshape(-1)
  logger._file_writer.add_histogram('hist_probs', values, global_step=steps, bins=list(range(2,32)))
  logger._file_writer.flush()

def on_episode_end(info):
  ipdb.set_trace()

def get_postprocess_fn(reporter):
  def my_postprocess(info):
    global steps_counter
    ipdb.set_trace()
    settings = CnfSettings()
    batch = info["post_batch"]
    episode = info["episode"]
    reporter.add_stat.remote(batch['infos'][0]['fname'],batch.count,batch['rewards'].sum())
    if 'probs' in batch:
      episode.custom_metrics["probs"] = batch['probs'].mean(axis=0)
    steps_counter += batch.count
    if steps_counter >= int(settings['min_timesteps_per_batch']):
      steps_counter = 0
      ObserverDispatcher().notify('new_batch')

  return my_postprocess

def eval_postprocess(info):
  ObserverDispatcher().notify('new_batch')

def get_logger_creator(settings):
  import tempfile
  from ray.tune.logger import UnifiedLogger, TBXLogger
  logdir_prefix = settings['name']+'_'
  logdir_default = settings['rl_log_dir']

  def named_logger_creator(config):
      """Creates a Unified logger with a default logdir prefix
      containing the agent name and the env id
      """
      if not os.path.exists(logdir_default):
          os.makedirs(logdir_default)
      logdir = tempfile.mkdtemp(
          prefix=logdir_prefix, dir=logdir_default)
      # return TBXLogger(config, logdir)
      return UnifiedLogger(config, logdir, loggers=(JsonLogger, CSVLogger, TBXLogger))

  return named_logger_creator

@ray.remote
def evaluate(steps, config, weights):
  env_config = config['env_config']
  settings = env_config['settings']
  env_config['eval'] = True
  env_config['formula_dir'] = settings['rl_test_data']
  config["model"]['custom_action_dist'] = 'argmax_dist'
  config["sample_batch_size"]=1
  config["train_batch_size"]=1  
  config["callbacks"] = {'on_postprocess_traj': eval_postprocess}
  w = RolloutWorker(env_creator=env_creator, policy=A3CTorchPolicy, batch_steps=1, batch_mode='complete_episodes', 
    callbacks={'on_postprocess_traj': eval_postprocess}, policy_config=config, env_config=config['env_config'])
  w.set_weights(weights['default_policy'])
  results = []
  for (i, _) in enumerate(OrderedProvider(settings['rl_test_data']).items):
    a = w.sample()
    results.append(a['rewards'].sum())
    # if (i % 10) == 0:
    #   print('Eval {}'.format(i))
  res = np.mean(results)
  print('Evaluate finished with reward {}'.format(res))
  # w.stop()
  del w
  return steps, res

class RLLibTrainer():
  def __init__(self):
    self.settings = CnfSettings()   
    self.clock = GlobalTick()
    self.logger = utils.get_logger(self.settings, 'rllib_trainer', 'logs/{}_rllib_trainer.log'.format(log_name(self.settings)))
    self.settings.formula_cache = FormulaCache()
    self.training_steps = self.settings['training_steps']
    self.test_every = int(self.settings['test_every'])
    register_env("sat_env", env_creator)
    register_env("sharp_env", env_creator)
    register_env("cadet_env", env_creator)
    ModelCatalog.register_custom_model("sat_model", SatThresholdModel)
    ModelCatalog.register_custom_model("cadet_model", CadetModel)
    ModelCatalog.register_custom_model("sharp_model", SharpModel)
    ModelCatalog.register_custom_action_dist("argmax_dist", TorchCategoricalArgmax)

  def main(self):    
    if self.settings['do_not_run']:
      print('Not running. Printing settings instead:')
      print(self.settings.hyperparameters)
      return
    print('Main pid is {}'.format(os.getpid()))
    reporter = RLLibEpisodeReporter.remote("{}/{}".format(self.settings['rl_log_dir'], log_name(self.settings)), self.settings)
    if self.settings['solver'] == 'minisat':
      config = a3c.DEFAULT_CONFIG.copy()
      model_name = 'sat_model'
      config["callbacks"] = {'on_postprocess_traj': get_postprocess_fn(reporter), 'on_train_result': on_train_result, }
                            # 'on_episode_end': on_episode_end}
      config["entropy_coeff"]=settings['entropy_alpha']
      custom_policy = log_logits_policy_wrapper(A3CTorchPolicy)
      trainer_class = a3c.A3CTrainer.with_updates(default_policy=custom_policy, name='GilTrainer', get_policy_class=lambda x: custom_policy,
                                                    make_policy_optimizer=make_cnf_optimizer)
      envname = 'sat_env'
    elif self.settings['solver'] == 'cadet':
      config = a3c.DEFAULT_CONFIG.copy()
      model_name = 'cadet_model'
      config["callbacks"] = {'on_postprocess_traj': get_postprocess_fn(reporter), }
      config["entropy_coeff"]=settings['entropy_alpha']
      trainer_class = a3c.A3CTrainer
      envname = 'cadet_env'
    elif self.settings['solver'] == 'sharpsat':
      config = es.DEFAULT_CONFIG.copy()
      model_name = 'sharp_model'
      config["episodes_per_batch"] = self.settings['episodes_per_batch']
      config["train_batch_size"] = self.settings['episodes_per_batch']*10      
      trainer_class = es.ESTrainer
      envname = 'sharp_env'
    else:
      assert False, "Unknown solver: {}".format(self.settings['solver'])
    # config = ppo.DEFAULT_CONFIG.copy()
    config["num_gpus"] = 0
    config["num_workers"] = self.settings['parallelism']
    config["eager"] = False
    config["sample_async"] = False
    config["batch_mode"]='complete_episodes'
    config["sample_batch_size"]=self.settings['min_timesteps_per_batch']
    config["train_batch_size"]=self.settings['min_timesteps_per_batch']
    # config["timesteps_per_iteration"]=10
    config['gamma'] = float(self.settings['gamma'])
    config["model"] = {"custom_model": model_name}
    config['use_pytorch'] = True
    config['lr'] = float(self.settings['init_lr'])
    config["env_config"]={'settings': settings.hyperparameters.copy(), 'formula_dir': self.settings['rl_train_data'], 'eval': False}
    if settings['use_seed']:
      config['seed'] = int(settings['use_seed'])

    trainer = trainer_class(config=config, env=envname, logger_creator=get_logger_creator(settings))
    # ipdb.set_trace()
    # self.result_logger = trainer._result_logger
    self.result_logger = trainer._result_logger._loggers[-1]
    if self.settings['base_model']:
      self.logger.info('Loading from {}..'.format(self.settings['base_model']))
      trainer.restore(self.settings['base_model'])
    print('Running for {} iterations..'.format(self.training_steps))
    eval_results = []
    for i in range(self.training_steps):
      result = trainer.train()      
      print(pretty_print(result))     

      # Do the uniform style reporting
      steps_val, reward_val = ray.get(reporter.report_stats.remote())  
      if steps_val and reward_val:
        self.result_logger._file_writer.add_scalar("ray/uniform/mean_steps", steps_val, global_step=result.get(TIMESTEPS_TOTAL))
        self.result_logger._file_writer.add_scalar("ray/uniform/mean_reward", reward_val, global_step=result.get(TIMESTEPS_TOTAL))
        self.result_logger._file_writer.flush()
      if i % self.test_every == 0 and i > 0:
        weights = ray.put({"default_policy": trainer.get_weights()})
        eval_results.append(evaluate.remote(result.get(TIMESTEPS_TOTAL), config, weights))
        ready, _ = ray.wait(eval_results,timeout=0.01)
        for obj in ready:
          t, result = ray.ray_get_and_free(obj)
          print('Adding evaluation result at timestep {}: {}'.format(t,result))
          # val = [tf.Summary.Value(tag="ray/eval/{}".format(self.settings['rl_test_data']), simple_value=result)]
          # self.result_logger._file_writer.add_summary(tf.Summary(value=val), t)
          self.result_logger._file_writer.add_scalar("ray/eval/{}".format(self.settings['rl_test_data']), result, global_step=t)
          self.result_logger._file_writer.flush()
          eval_results.remove(obj)
      if i % 100 == 0:
        checkpoint = trainer.save()
        print("checkpoint saved at", checkpoint)


def rllib_main(): 
  settings = CnfSettings()
  import ray
  address = settings['ray_address']
  if address:
    print('Running in ray cluster mode')
    ray.init(address=address, redis_password='blabla')
  else:
    ray.init()
  rllib_trainer = RLLibTrainer()
  rllib_trainer.main()

# if __name__=='__main__':
#   parser = argparse.ArgumentParser(description='Process some params.')
#   parser.add_argument('params', metavar='N', type=str, nargs='*', help='an integer for the accumulator')
#   parser.add_argument('-s', '--settings', type=str, help='settings file') 
#   parser.add_argument('-c', '--cluster', action='store_true', default=False, help='settings file') 
#   args = parser.parse_args()
#   get_settings_from_file(args.settings)
#   for param in args.params:
#     k, v = param.split('=')
#     settings.hyperparameters[k]=v
#   rllib_main(args)



# config["num_envs"]=1
# if settings['preload_formulas']:
#     settings.formula_cache.load_files(provider.items)  
# settings.hyperparameters['loglevel']='logging.INFO'
# settings.hyperparameters['sat_min_reward']=-100
# settings.hyperparameters['max_step']=300
# settings.hyperparameters['min_timesteps_per_batch']=100


# config["model"] = {"custom_model": "sat_model"}
# config["use_pytorch"] = True


# Can optionally call trainer.restore(path) to load a checkpoint.

# # w = RolloutWorker(env_creator=env_creator, policy=A3CTorchPolicy, batch_mode='complete_episodes', policy_config=config)
# workers = WorkerSet(
#     policy=A3CTFPolicy,
#     env_creator=env_creator,
#     num_workers=2, 
#     trainer_config=config
#     )

# # In[7]:

# a = ray.get(workers.remote_workers()[0].sample.remote())
# print('total steps: {}'.format(len(a['obs'])))
# b = ray.get(workers.remote_workers()[0].sample.remote())
# print('total steps: {}'.format(len(b['obs'])))