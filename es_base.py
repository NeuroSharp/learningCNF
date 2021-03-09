from IPython.core.debugger import Tracer
import time

from settings import *
from episode_reporter import *
from policy_factory import *
from functional_env import *
from formula_utils import *
from episode_data import *
from es_worker import *

class ESBase:
	def __init__(self, settings, *args, **kwargs):
		self.settings = settings
		self.population_size = settings['es_population']
		self.num_formulas = settings['es_num_formulas']
		settings.formula_cache = FormulaCache()
		if settings['preload_formulas']:
			settings.formula_cache.load_files(provider.items)	
		ProviderClass = eval(settings['episode_provider'])
		self.provider = ProviderClass(settings['rl_train_data'])
		self.reporter = PGReporterServer(PGEpisodeReporter("{}/{}".format(settings['rl_log_dir'], log_name(settings)), settings, tensorboard=False))

	# Entrypoint

	def main_loop(self):
		pass

	# Initialize the (possibly implicit) distribution of solutions

	def initialize_distribution(self):
		pass


	''' Sample from the distribution of solutions
	
	 Receives:
	
		 num - Number of samples to draw
	
	 Returns:
	
		 samples - A sample of strategies. 

	'''
	def sample_population(self, num):
		pass

	''' Take a step and evolve the distribution of solutions.
	
	 Receives:
	
		 results - A list of tuples (sample, fitness)

	'''

	def evolve(self, results):
		pass

class ESPolicyBase(ESBase):
	def __init__(self, *args, **kwargs):
		super(ESPolicyBase, self).__init__(*args, **kwargs)
		self.policy = PolicyFactory().create_policy(requires_grad=False)
		self.policy.share_memory()

	def main_loop(self):
		workers = []
		ack_queue = mp.Queue()
		task_queue = mp.Queue()
		self.initialize_distribution()

		for i in range(self.settings['parallelism']):
			interactor = EnvInteractor(self.settings, self.policy, None, i, reporter=self.reporter.proxy(), requires_grad=False)
			func_agent = FunctionalEnv(self.settings,interactor)			
			worker = ESWorker(func_agent, self.settings, task_queue, ack_queue, i)
			workers.append(worker)
			worker.start()

		while True:
			fnames = self.provider.sample(size=self.num_formulas)
			samples = self.sample_population(self.population_size)
			for i, sample in enumerate(samples):
				task_queue.put((WorkerCommands.CMD_TASK,(sample,fnames),i))
			rewards = [0]*self.population_size
			for _ in range(self.population_size):
				ack, rc, i = ack_queue.get()
				rewards[i] = rc
			self.evolve(zip(samples,rewards))

