from IPython.core.debugger import Tracer
from env_interactor import *

class FunctionalEnv:
	def __init__(self, settings, interactor):
		self.settings = settings
		self.interactor = interactor

	def setup_model_weights(self, state_dict, relative=True):
		if relative:
			curr_model = self.interactor.lmodel.state_dict()
			for k in state_dict.keys():
				curr_model[k] += state_dict[k]
		else:
			curr_model = state_dict
		self.interactor.lmodel.load_state_dict(curr_model, strict=False)

	# This is the main interface. It runs the model/env using its EnvInteractor instance.
	# Input:
	#
	#   state_dict: An ordered_dict, the model params
	#		fnames: A list of formula names
	#		relative: True if state_dict is relative to the current global model, False if its absolute values
	#
	# Returns:
	#		Total rewards

	def forward(self, state_dict, fnames, **kwargs):
		self.setup_model_weights(state_dict, **kwargs)
		total_reward = 0
		for fname in fnames:
			traces, batch_length = self.interactor.collect_batch(fname, batch_size=1, training=False)
			if traces:
				for trace in traces:
					total_reward += sum([x.reward for x in trace])

		return total_reward