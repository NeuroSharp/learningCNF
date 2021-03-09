import os
import os.path
import torch
# from torch.distributions import Categorical
import ipdb
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
from cadet_env import *
from rl_model import *
from new_policies import *
from qbf_data import *
from qbf_model import QbfClassifier
from utils import *
from rl_utils import *
from cadet_utils import *
from episode_reporter import *
from mp_episode_manager import *
# from episode_manager import test_envs
from episode_manager import EpisodeManager
from episode_data import *
from formula_utils import *
from policy_factory import *
import torch.nn.utils as tutils
import sat_policies
settings = CnfSettings()

# Units of 30 seconds

UNIT_LENGTH = 30

# 2 = 1 Minute
REPORT_EVERY = 2
SAVE_EVERY = 20
TEST_EVERY = settings['test_every']
init_lr = settings['init_lr']
desired_kl = settings['desired_kl']
stepsize = settings['init_lr']
curr_lr = init_lr
max_reroll = 0

def handleSIGCHLD(a,b):
  os.waitpid(-1, os.WNOHANG)

class MyManager(BaseManager): 
  pass
# MyManager.register('EpisodeData',EpisodeData)
MyManager.register('wsync',WorkersSynchronizer)

def a3c_main():
  # mp.set_start_method('forkserver')
  signal.signal(signal.SIGCHLD, handleSIGCHLD)
  if settings['do_not_run']:
    print('Not running. Printing settings instead:')
    print(settings.hyperparameters)
    return
  logger = logging.getLogger('task_a3c')
  logger.setLevel(eval(settings['loglevel']))    
  fh = logging.FileHandler('logs/{}_a3c_main.log'.format(log_name(settings)))
  fh.setLevel(logging.DEBUG)
  logger.addHandler(fh)    

  total_steps = 0
  global_steps = mp.Value('i', 0)
  global_grad_steps = mp.Value('i', 0)
  global_episodes = mp.Value('i', 0)
  manager = MyManager()
  reporter = PGReporterServer(PGEpisodeReporter("{}/{}".format(settings['rl_log_dir'], log_name(settings)), settings, tensorboard=settings['report_tensorboard']))
  reporter.start()
  manager.start()
  wsync = manager.wsync()
  batch_sem = mp.Semaphore(settings['batch_sem_value'])
  ed = None
  # ed = manager.EpisodeData(name=settings['name'], fname=settings['base_stats'])
  # ds = QbfCurriculumDataset(fnames=settings['rl_train_data'], ed=ed)
  ProviderClass = eval(settings['episode_provider'])
  provider = ProviderClass(settings['rl_train_data'])
  settings.formula_cache = FormulaCache()
  if settings['preload_formulas']:
    settings.formula_cache.load_files(provider.items)

  # em = EpisodeManager(provider, ed=ed, parallelism=settings['parallelism'],reporter=reporter.proxy())  

  policy = PolicyFactory().create_policy()
  policy.share_memory()
  optimizer = SharedAdam(filter(lambda p: p.requires_grad, policy.parameters()), lr=stepsize)    
  num_steps = 100000000
  curr_lr = init_lr
  lr_schedule = PiecewiseSchedule([
                                       (0,                   init_lr),
                                       (num_steps / 20, init_lr),
                                       (num_steps / 10, init_lr * 0.75),
                                       (num_steps / 5,  init_lr * 0.5),
                                       (num_steps / 3,  init_lr * 0.25),
                                       (num_steps / 2,  init_lr * 0.1),
                                  ],
                                  outside_value = init_lr * 0.02) 

  kl_schedule = PiecewiseSchedule([
                                       (0,                   desired_kl),
                                       (num_steps / 10, desired_kl),
                                       (num_steps / 5,  desired_kl * 0.5),
                                       (num_steps / 3,  desired_kl * 0.25),
                                       (num_steps / 2,  desired_kl * 0.1),
                                  ],
                                  outside_value=desired_kl * 0.02) 
  mp.set_sharing_strategy('file_system')
  workers = {i: WorkerEnv(settings,policy,optimizer,provider,ed,global_steps, global_grad_steps, global_episodes, i, wsync, batch_sem, init_model=None, reporter=reporter.proxy()) for i in range(settings['parallelism'])}  
  print('Running with {} workers...'.format(len(workers)))
  for i,w in workers.items():
    w.start()  

  i = 0
  pval = None
  main_proc = psutil.Process(os.getpid())
  set_proc_name(str.encode('a3c_main'))
  while True:
    time.sleep(UNIT_LENGTH)
    logger.info('Round {}'.format(i))
    while wsync.get_total() > 0:
      w = wsync.pop()
      j = w[0]
      logger.info('restarting worker {}'.format(j))
      workers[j] = WorkerEnv(settings,policy,optimizer,provider,ed,global_steps, global_grad_steps, global_episodes, j, wsync, batch_sem, init_model=w[1], reporter=reporter.proxy())
      workers[j].start()
    gsteps = global_steps.value
    try:
      total_mem = main_proc.memory_info().rss / float(2 ** 20)
      children = main_proc.children(recursive=True)
      for child in children:
        child_mem = child.memory_info().rss / float(2 ** 20)
        total_mem += child_mem
        logger.info('Child pid is {}, name is {}, mem is {}'.format(child.pid, child.name(), child_mem))
      logger.info('Total memory is {}'.format(total_mem))
    except:       # A child could already be dead due to a race. Just ignore it this round.
      pass
    if i % REPORT_EVERY == 0 and i>0:
      if type(policy) == sat_policies.SatBernoulliPolicy:
        pval = float(policy.pval.detach().numpy())
      reporter.proxy().report_stats(gsteps, provider.get_total(), pval)
      eps = global_episodes.value
      logger.info('Average number of simulated episodes per time unit: {}'.format(global_episodes.value/i))
    if i % SAVE_EVERY == 0 and i>0:
      torch.save(policy.state_dict(),'%s/%s_step%d.model' % (settings['model_dir'],utils.log_name(settings), gsteps))
      if ed is not None:
        ed.save_file()
    if i % TEST_EVERY == 0 and i>0:      
      em.test_envs(settings['rl_test_data'], policy, iters=1, training=False)
    if settings['rl_decay']:
      new_lr = lr_schedule.value(gsteps)
      if new_lr != curr_lr:
        utils.set_lr(optimizer,new_lr)
        print('setting new learning rate to {}'.format(new_lr))
        curr_lr = new_lr

    i += 1
