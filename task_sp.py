import os
import os.path
import torch
# from torch.distributions import Categorical
from IPython.core.debugger import Tracer
import pdb
import random
import time
import tracemalloc
import signal
import logging

from multiprocessing.managers import BaseManager
import torch.multiprocessing as mp
from shared_adam import SharedAdam
from settings import *
from utils import *
from rl_utils import *
from tick_utils import *
from episode_reporter import *
from episode_data import *
from formula_utils import *
from policy_factory import *
import torch.nn.utils as tutils
from env_factory import *
from env_interactor import *  
from env_trainer import *
from dispatcher import *


class SingleProcessTrainer(IEnvTrainerHook):
  def __init__(self):
    self.settings = CnfSettings()
    seed_all(self.settings, 'single_process')
    self.settings.hyperparameters['mp']=False
    self.clock = GlobalTick()
    self.SAVE_EVERY = 500
    self.TEST_EVERY = self.settings['test_every']
    self.REPORT_EVERY = self.settings['report_every']
    self.reporter = PGEpisodeReporter("{}/{}".format(self.settings['rl_log_dir'], log_name(self.settings)), self.settings, tensorboard=self.settings['report_tensorboard'])    
    self.training_steps = self.settings['training_steps']        
    self.dispatcher = ObserverDispatcher()
    self.logger = utils.get_logger(self.settings, 'workers_sync', 'logs/{}_workers_sync.log'.format(log_name(self.settings)))    

    self.reporter.log_env(self.settings['rl_log_envs'])  
    ProviderClass = eval(self.settings['episode_provider'])
    self.provider = ProviderClass(self.settings['rl_train_data'])
    self.settings.formula_cache = FormulaCache()
    if self.settings['preload_formulas']:
      self.settings.formula_cache.load_files(self.provider.items)  
    self.trainer = EnvTrainer(self.settings,self.provider,'SingleProcessTrainer', self, reporter=self.reporter, init_pyro=False)
    if self.settings['profiling']:
      pr = cProfile.Profile() 
    if self.settings['memory_profiling']:
      tracemalloc.start(25)

  def main(self):    
    if self.settings['do_not_run']:
      print('Not running. Printing settings instead:')
      print(self.settings.hyperparameters)
      return

    print('Running for {} iterations..'.format(self.training_steps))
    for i in range(self.training_steps):
      clock.tick()
      self.dispatcher.notify('new_batch')
      num_env_steps, num_episodes = self.trainer.train_step()


      if not (i % self.REPORT_EVERY) and i>0:
        self.reporter.report_stats(i, len(self.provider))

      if i % self.SAVE_EVERY == 0 and i>0:
        self.trainer.interactor.save('%s/%s_step%d.model' % (self.settings['model_dir'],utils.log_name(self.settings), i))

      # if i % TEST_EVERY == 0 and i>0:
      #   if settings['rl_validation_data']:
      #     print('Testing envs:')
      #     rc = em.test_envs(settings['rl_validation_data'], policy, iters=2)
      #     z = np.array(list(rc.values()))
      #     val_average = z.mean()
      #     print('Average on {}: {}'.format(settings['rl_validation_data'],val_average))
      #     # log_value('Validation', val_average, total_steps)
      #   if settings['rl_test_data']:                
      #     rc = em.test_envs(settings['rl_test_data'], policy, iters=1, training=False)
      #     z = np.array(list(rc.values()))        
      #     test_average = z.mean()
      #     print('Average on {}: {}'.format(settings['rl_test_data'],test_average))

def sp_main():
  sp_trainer = SingleProcessTrainer()
  sp_trainer.main()
