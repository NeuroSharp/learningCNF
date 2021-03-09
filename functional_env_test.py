from IPython.core.debugger import Tracer

from settings import *
from episode_reporter import *
from policy_factory import *
from functional_env import *
from formula_utils import *
from episode_data import *

settings = CnfSettings()

def functional_env_test():
	settings.formula_cache = FormulaCache()
	if settings['preload_formulas']:
		settings.formula_cache.load_files(provider.items)	
	ProviderClass = eval(settings['episode_provider'])
	provider = ProviderClass(settings['rl_train_data'])
	reporter = PGEpisodeReporter("{}/{}".format(settings['rl_log_dir'], log_name(settings)), settings, tensorboard=False)
	policy = PolicyFactory().create_policy()
	policy.share_memory()
	interactor = EnvInteractor(settings, policy, None, '1', reporter=reporter)
	func_agent = FunctionalEnv(settings,interactor)
	z = [provider.get_next()]		
	reward = func_agent.forward(policy.state_dict(),z)
	print('reward for {} is {}'.format(z[0],reward))
