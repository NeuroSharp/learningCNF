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
  settings = CnfSettings()
  settings.hyperparameters = config['env_config']['settings']
  n = settings['test_parallelism']
  workers = [es.es.Worker.remote(config, {"action_noise_std": 0.01}, env_creator, None, True) for _ in range(n)]
  params = ray.put(weights['default_policy'])
  fnames = OnePassProvider(settings['es_validation_data']).items
  parts_ids = [ray.put(list(x)) for x in np.array_split(np.array(fnames), n)]
  res_all = [w.rollout.remote(part,None,add_noise=False,params=params) for (w, part) in zip(workers, parts_ids)]
  results = ray.get(res_all)
  rewards, lengths = zip(*results)
  rc = np.mean(np.concatenate(lengths))
  for w in workers:
    del w

  return steps, rc

class ESEval():
  def __init__(self):
    self.settings = CnfSettings()   
    self.clock = GlobalTick()
    self.logger = utils.get_logger(self.settings, 'rllib_trainer', 'logs/{}_rllib_trainer.log'.format(log_name(self.settings)))
    self.settings.formula_cache = FormulaCache()
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
    config = es.DEFAULT_CONFIG.copy()
    config["episodes_per_batch"] = self.settings['episodes_per_batch']
    config["train_batch_size"] = self.settings['episodes_per_batch']*10      
    trainer_class = es.ESTrainer
    if self.settings['solver'] == 'minisat' or self.settings['solver'] == 'sat_es':
      model_name = 'sat_model'
      envname = 'sat_env'
    elif self.settings['solver'] == 'cadet':
      model_name = 'cadet_model'
      envname = 'cadet_env'
    elif self.settings['solver'] == 'sharpsat':
      model_name = 'sharp_model'
      envname = 'sharp_env'
    else:
      assert False, "Unknown solver: {}".format(self.settings['solver'])
    config["num_gpus"] = 0
    config["eager"] = False
    config["sample_async"] = False
    config["noise_size"]=10
    # config["timesteps_per_iteration"]=10
    config["model"] = {"custom_model": model_name}
    config["env_config"]={'settings': settings.hyperparameters.copy(), 'formula_dir': self.settings['es_train_data'], 'eval': False}
    if settings['use_seed']:
      config['seed'] = int(settings['use_seed'])
    trainer = trainer_class(config=config, env=envname, logger_creator=get_logger_creator(settings))   
    # assert self.settings['base_model'] != None , "Must provide a saved model"
    if self.settings['base_model']:
      self.logger.info('Loading from {}..'.format(self.settings['base_model']))
      trainer.restore(self.settings['base_model'])    
    self.result_logger = trainer._result_logger._loggers[-1]
    print('Beginning eval...')
    weights = ray.put({"default_policy": trainer.get_weights()})
    rc = evaluate.remote(0, config, weights)
    t, result = ray.get(rc)
    print('Adding evaluation result at timestep {}: {}'.format(t,result))

def es_eval_main(): 
  settings = CnfSettings()
  import ray
  address = settings['ray_address']
  if address:
    print('Running in ray cluster mode')
    ray.init(address=address, redis_password='blabla', include_webui=False)
  else:
    ray.init()
  es_loop = ESEval()
  es_loop.main()

