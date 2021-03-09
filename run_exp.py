import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from datautils import *
import utils
import time
import numpy as np
import ipdb
import pdb
from tensorboard_logger import configure, log_value
from sacred import Experiment
from training import *

# EX_NAME = 'trenery_4'

# ex = Experiment(EX_NAME)

@ex.config
def cfg():	
	exp_name = EX_NAME
	exp_time = int(time.time())
	state_dim = 30
	embedding_dim = 10
	ground_dim = 3
	policy_dim1 = 100
	policy_dim2 = 50
	max_variables = 200 
	max_clauses = 600
	num_ground_variables = 3 
	data_mode = DataMode.SAT
	data_dir = 'data/'
	dataset = 'boolean8'
	model_dir = 'saved_models'
	# 'base_model = 'saved_models/run_bigsat_50_4_nc2_bs40_ed4_iters8__1508199570_epoch200.model'
	base_model = None
	base_mode = BaseMode.ALL
	max_iters = 12
	batch_size = 64
	val_size = 100 
	threshold = 10
	init_lr = 0.001
	# 'init_lr = 0.0004
	decay_lr = 0.055
	decay_num_epochs = 6
	cosine_margin = 0
	# 'classifier_type = 'BatchGraphLevelClassifier'
	# 'classifier_type = 'BatchEqClassifier'
	classifier_type = 'TopLevelClassifier'
	combinator_type = 'SymmetricSumCombine'	    
	ground_combinator_type = 'DummyGroundCombinator'	    
	# 'ground_combinator_type = 'GroundCombinator'	
	encoder_type = 'BatchEncoder'	    
	# 'embedder_type = 'TopVarEmbedder'	    
	embedder_type = 'GraphEmbedder'	    
	# 'negate_type = 'minus'
	negate_type = 'regular'
	sparse = True
	# sparse = False
	gru_bias = False
	use_ground = True
	moving_ground = False 
	split = False
	# 'cuda = True 
	cuda = False
	reset_on_save = False


	run_task='train'

	max_edges = 20


	
	DS_TRAIN_FILE = 'expressions-synthetic/split/%s-trainset.json' % dataset
	DS_VALIDATION_FILE = 'expressions-synthetic/split/%s-validationset.json' % dataset
	DS_TEST_FILE = 'expressions-synthetic/split/%s-testset.json' % dataset


@ex.automain
def main(DS_TRAIN_FILE, DS_VALIDATION_FILE, DS_TEST_FILE, data_mode, threshold):	
	settings = CnfSettings(ex.current_run.config)

	if settings['run_task'] == 'train_cadet':
		from task_cadet import cadet_main
		cadet_main(settings)
		exit()


	# pdb.set_trace()
	# ds1 = CnfDataset.from_eqparser(DS_TRAIN_FILE,mode=data_mode, threshold=threshold)
	# ds2 = CnfDataset.from_eqparser(DS_VALIDATION_FILE, threshold=0, ref_dataset=ds1, mode=data_mode)
	ns1 = CnfDataset.from_dimacs('data/train_big_10/sat/', 'data/train_big_10/unsat/', max_size=100, sparse=settings['sparse'])
	ns2 = CnfDataset.from_dimacs('data/validation_10/sat/', 'data/validation_10/unsat/', max_size=100, sparse=settings['sparse'])
	# ds1 = CnfDataset(DS_TRAIN_FILE,threshold,mode=data_mode, num_max_clauses=12)
	# ds2 = CnfDataset(DS_VALIDATION_FILE, threshold, ref_dataset=ds1, mode=data_mode, num_max_clauses=12)
	# ds3 = CnfDataset(DS_TEST_FILE, threshold, ref_dataset=ds1, mode=data_mode)
	# print('Classes from validation set:')
	# print(ds2.labels)
	# train(ds1,ds2)

	print('max_variables, clauses: %d, %d' % (ns1.max_variables, ns1.max_clauses))
	train(ns1,ns2)
