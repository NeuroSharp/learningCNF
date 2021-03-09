from IPython.core.debugger import Tracer
import time
import torch

import torch.multiprocessing as mp
from settings import *
from episode_reporter import *
from policy_factory import *
from functional_env import *
from es_worker import *
from es_base import *
from formula_utils import *
from episode_data import *
from test_worker import *

settings = CnfSettings()


class NaiveEvolutionaryStrategy(ESPolicyBase):
	def __init__(self, *args, **kwargs):
		super(NaiveEvolutionaryStrategy, self).__init__(*args, **kwargs)		
		self.stepsize = self.settings['es_naive_stepsize']
		self.blacklisted_keys = []
		for k in self.policy.state_dict().keys():
			if any([x in k for x in self.settings['g2l_blacklist']]):
				self.blacklisted_keys.append(k)    

	# We do nothing here, the distribution is implicitly defined by the single point which is the model..

	def initialize_distribution(self):
		pass


	# This takes the current policy and samples in a sphere around it.

	def sample_population(self, num):
		state_dict = self.policy.state_dict()
		for k in self.blacklisted_keys:
			state_dict.pop(k,None)
		rc = []
		for _ in range(num):
			# Sample uniformly seperately for each group of weights
			random_vals = {k:np.random.rand(*x.numpy().shape)-0.5 for (k,x) in state_dict.items()}
			norm = np.linalg.norm([np.linalg.norm(x) for x in random_vals.values()])
			rc.append({k:torch.from_numpy(x*self.stepsize/norm).float() for (k,x) in random_vals.items()})

		return rc
	def evolve(self, results):
		Tracer()()

def naive_es_main():
	system = NaiveEvolutionaryStrategy(settings)
	system.main_loop()




