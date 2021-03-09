import os
import os.path
import torch
# from torch.distributions import Categorical
import ipdb
import pdb
import random
import time
import copy
import shelve

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
from collection_worker import *
from episode_data import *
from gridparams import *
import torch.nn.utils as tutils

settings = CnfSettings()


# 2 = 1 Minute
REPORT_EVERY = 25
SAVE_EVERY = 100

ProviderClass = eval(settings['episode_provider'])
class MyManager(BaseManager): pass
MyManager.register('EpisodeData',EpisodeData)
MyManager.register(settings['episode_provider'],ProviderClass)

def collect_lbd_main():
  if settings['do_not_run']:
    print('Not running. Printing settings instead:')
    print(settings.hyperparameters)
    return
  settings.hyperparameters['policy']='SatLBDPolicy'
  total_steps = 0
  global_grad_steps = mp.Value('i', 0)
  episodes_per_batch = settings['episodes_per_batch']
  workers_sem = mp.Semaphore(0)
  workers_queue = mp.Queue()
  manager = MyManager()
  reporter = PGReporterServer(PGEpisodeReporter("{}/{}".format(settings['rl_log_dir'], log_name(settings)), settings, tensorboard=settings['report_tensorboard']))
  manager.start()
  reporter.start()
  ed = manager.EpisodeData(name=settings['name'], fname=settings['base_stats'])
  provider = getattr(manager,settings['episode_provider'])(settings['rl_train_data'])
  policy = create_policy()
  policy.share_memory()
  num_steps = 100000000
  shelf_name = 'vis_lbd.shelf'
  # mp.set_sharing_strategy('file_system')
  workers = [CollectionWorker(settings, policy, provider, ed, workers_queue, workers_sem, global_grad_steps, i, reporter=reporter.proxy()) for i in range(settings['parallelism'])]  
  print('Running with {} workers...'.format(len(workers)))
  for w in workers:
    w.start()  
    # Change learning rate according to KL

  gsteps = 0
  num_episodes = 0
  set_proc_name(str.encode('lbd_main'))  
  if settings['profiling']:
    pr = cProfile.Profile() 
  all_results = {}
  while provider.get_next():
    i = 0
    fname = provider.get_next()
    print('entry {}'.format(fname))
    mt1 = time.time()
    all_episodes = []
    for _ in range(episodes_per_batch):
      workers_sem.release()
    for _ in range(episodes_per_batch):
      episode = workers_queue.get()      
      all_episodes.append(episode)
    mt2 = time.time()    
    num_episodes += episodes_per_batch
    i += 1
    with shelve.open(shelf_name) as db:
      db[fname] = all_episodes
      if sum(all_episodes) == 0:
        print('degenerate episode {}'.format(provider.get_next()))

    provider.reset()
