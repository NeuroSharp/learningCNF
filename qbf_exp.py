from config import *
from sacred import Experiment
from qbf_data import *
from rl_model import *


@ex.automain
def main():
	settings = CnfSettings(ex.current_run.config)
	from task_qbf_train import qbf_train_main	
	settings.hyperparameters['name'] = ex.current_run.experiment_info['name']
	qbf_train_main()
