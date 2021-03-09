from functional_worker_base import *

class ESWorker(FunctionalWorkerBase):
  def __init__(self, func_env, *args, **kwargs):    
    super(ESWorker, self).__init__(*args, **kwargs)
    self.func_env = func_env

  def do_task(self, params):
  	state_dict, fnames = params
  	return self.func_env.forward(state_dict, fnames)