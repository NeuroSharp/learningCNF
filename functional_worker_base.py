import time
from IPython.core.debugger import Tracer
import cProfile
from enum import Enum
import logging

from settings import *
from functional_env import *
from worker_base import *

class WorkerCommands(Enum):
  CMD_EXIT = 1
  CMD_TASK = 2
  ACK_EXIT = 3
  ACK_TASK = 4

class FunctionalWorkerBase(WorkerBase):
  def __init__(self, settings, index, task_queue_in, task_queue_out):
    super(FunctionalWorkerBase, self).__init__(settings,index)
    self.index = index
    self.name = 'func_worker_{}'.format(index)
    self.settings = settings
    self.task_queue_in = task_queue_in
    self.task_queue_out = task_queue_out

    self.logger = utils.get_logger(self.settings, 'func_worker_{}'.format(self.index), 
                                    'logs/{}_{}.log'.format(log_name(self.settings), self.name))    

  def run_loop(self):
    while True:
      cmd, params, cookie = self.task_queue_in.get()
      if cmd == WorkerCommands.CMD_EXIT:
        self.task_queue_out.put((WorkerCommands.ACK_EXIT,None, cookie))
        return
      elif cmd == WorkerCommands.CMD_TASK:
        rc = self.do_task(params)
        self.task_queue_out.put((WorkerCommands.ACK_TASK,rc, cookie))        
      else:
        self.logger.error('Received unknown WorkerCommands!')
        return
        
  def do_task(self, params):
    pass
