import os
import os.path
import torch
# from torch.distributions import Categorical
from IPython.core.debugger import Tracer
import pdb
import random
import time
import copy

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
from forward_worker import *
from episode_data import *
import torch.nn.utils as tutils

settings = CnfSettings()


# 2 = 1 Minute
REPORT_EVERY = 25
SAVE_EVERY = 100

init_lr = settings['init_lr']
desired_kl = settings['desired_kl']
stepsize = settings['init_lr']
curr_lr = init_lr
max_reroll = 0

ProviderClass = eval(settings['episode_provider'])
class MyManager(BaseManager): pass
MyManager.register('EpisodeData',EpisodeData)
MyManager.register(settings['episode_provider'],ProviderClass)

def parallel_main():
  def train_model(transition_data):
    policy.train()
    mt1 = time.time()        
    loss, logits = policy.compute_loss(transition_data)
    mt2 = time.time()
    if loss is None:
      return
    # break_every_tick(20)
    optimizer.zero_grad()
    loss.backward()
    mt3 = time.time()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), settings['grad_norm_clipping'])
    if any([(x.grad!=x.grad).data.any() for x in policy.parameters() if x.grad is not None]): # nan in grads
      print('NaN in grads!')
      Tracer()()
    # tutils.clip_grad_norm(policy.parameters(), 40)
    optimizer.step()    
    print('Times are: Loss: {}, Grad: {}, Ratio: {}'.format(mt2-mt1, mt3-mt2, ((mt3-mt2)/(mt2-mt1))))
    return logits



  if settings['do_not_run']:
    print('Not running. Printing settings instead:')
    print(settings.hyperparameters)
    return
  total_steps = 0
  global_steps = mp.Value('i', 0)
  global_grad_steps = mp.Value('i', 0)
  global_episodes = mp.Value('i', 0)
  episodes_per_batch = settings['episodes_per_batch']
  workers_sem = mp.Semaphore(episodes_per_batch)
  workers_queue = mp.Queue()
  manager = MyManager()
  reporter = PGReporterServer(PGEpisodeReporter("{}/{}".format(settings['rl_log_dir'], log_name(settings)), settings, tensorboard=settings['report_tensorboard']))
  manager.start()
  reporter.start()
  ed = manager.EpisodeData(name=settings['name'], fname=settings['base_stats'])
  provider = getattr(manager,settings['episode_provider'])(settings['rl_train_data'])
  policy = create_policy()
  if settings['cuda']:
    cpu_policy = create_policy().cpu()    
  else:
    cpu_policy = policy
  cpu_policy.share_memory()
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
  workers = [WorkerEnv(settings, cpu_policy, provider, ed, workers_queue, workers_sem, global_grad_steps, i, reporter=reporter.proxy()) for i in range(settings['parallelism'])]  
  print('Running with {} workers...'.format(len(workers)))
  for w in workers:
    w.start()  
    # Change learning rate according to KL

  i = 0
  gsteps = 0
  num_episodes = 0
  set_proc_name(str.encode('parallel_main'))

  if settings['profiling']:
    pr = cProfile.Profile() 
  while True:
    all_episodes = []
    mt1 = time.time()
    for _ in range(episodes_per_batch):
      episode = workers_queue.get()
      gsteps += len(episode)
      all_episodes.append([undensify_transition(x) for x in episode])
    mt2 = time.time()
    transition_data = [i for x in all_episodes for i in x]
    num_episodes += episodes_per_batch
    print('Collected {} episodes with {} steps in {} seconds'.format(episodes_per_batch,len(transition_data),mt2-mt1))
    provider.reset()
    train_model(transition_data)    
    if settings['cuda']:
      sd = copy.deepcopy(policy.state_dict())
      for k in sd.keys():
        sd[k]=sd[k].cpu()
      cpu_policy.load_state_dict(sd)
    if i % REPORT_EVERY == 0 and i>0:
      reporter.proxy().report_stats(gsteps, provider.get_total())
      # print('Average number of simulated episodes per time unit: {}'.format(global_episodes.value/i))
    if i % SAVE_EVERY == 0 and i>0:
      torch.save(policy.state_dict(),'%s/%s_step%d.model' % (settings['model_dir'],utils.log_name(settings), gsteps))
      ed.save_file()
    if settings['rl_decay']:
      new_lr = lr_schedule.value(gsteps)
      if new_lr != curr_lr:
        utils.set_lr(optimizer,new_lr)
        print('setting new learning rate to {}'.format(new_lr))
        curr_lr = new_lr

    with global_grad_steps.get_lock():
      global_grad_steps.value += 1    
    i += 1
    for _ in range(episodes_per_batch):
      workers_sem.release()
