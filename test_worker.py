import time
import random
from functional_worker_base import *
from env_tester import *
from episode_data import *

class TestWorker(FunctionalWorkerBase):
  def __init__(self, settings, index, *args, **kwargs):
    super(TestWorker, self).__init__(settings, index, *args, **kwargs)    
    self.name = 'TestWorker-{}'.format(index)

  def do_task(self, params):
    step, statedict, fnames = params
    policy = PolicyFactory().create_policy()
    policy.load_state_dict(statedict)
    policy.eval()
    tester = EnvTester(self.settings,self.index)
    rc = tester.test_envs(OnePassProvider(fnames), policy, iters=1, deterministic=True)
    return (step, rc)