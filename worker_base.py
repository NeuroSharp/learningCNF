import numpy as np
import torch
import time
from IPython.core.debugger import Tracer
import os
import sys
import signal
import select
import torch.multiprocessing as mp
import torch.optim as optim
import cProfile
from collections import namedtuple, deque
from namedlist import namedlist

from qbf_data import *
from settings import *
from utils import *
from rl_utils import *
from cadet_utils import *
from episode_data import *
from env_factory import *
from env_interactor import *  


class WorkerBase(mp.Process):
  def __init__(self, settings, name, **kwargs):
    super(WorkerBase, self).__init__(**kwargs)
    self.name = 'WorkerBase%i' % name
    self.settings = settings
    self.kwargs = kwargs

  def init_proc(self, **kwargs):
    set_proc_name(str.encode(self.name))    
    utils.seed_all(self.settings, self.name)
    self.settings.hyperparameters['cuda']=False         # No CUDA in the worker threads

  def run(self):
    self.init_proc(**self.kwargs)
    if self.settings['memory_profiling']:
      tracemalloc.start(25)
    if self.settings['profiling']:
      cProfile.runctx('self.run_loop()', globals(), locals(), 'prof_{}.prof'.format(self.name))
    else:
      self.run_loop()

  def run_loop(self):
    pass