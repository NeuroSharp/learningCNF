import time
# from utils import BaseMode
from sacred import Experiment

ex = Experiment('REINFORCE')

@ex.config
def cfg():	
	name = 'DEF_NAME'
	mongo=False
	mongo_host='russell.eecs.berkeley.edu:27017'
	mongo_dbname='rl_exp'
	exp_time = int(time.time())
	pyro_name = None   		# This will become the experiment name
	pyro_port=None
	pyro_host=None
	ray_address=None  	  # None | auto
	save_config = True
	update_config=None
	state_dim = 90
	embedding_dim = 20  # deprecated
	vemb_dim = 16
	cemb_dim = 64
	ground_dim = 8  # deprecated
	vlabel_dim = 6
	clabel_dim = 6
	max_variables = 200 
	max_clauses = 600
	num_ground_variables = 0  # deprecated
	data_dir = 'data/'
	dataset = 'boolean8'
	model_dir = 'saved_models'
	base_model = None
	# base_mode = BaseMode.ITERS 
	# base_mode = BaseMode.ALL	
	base_iters = 1
	base_stats = None
	# base_mode = BaseMode.EMBEDDING
	training_steps = 100000000
	max_iters = 1
	max_step = 250			# Maximum steps per formula
	max_seconds = 0	
	batch_size = 128  # deprecated
	val_size = 100  # deprecated?
	threshold = 10  # deprecated
	init_lr = 0.0006
	num_epochs = 10
	# init_lr = 0.001
	# 'init_lr = 0.0004
	decay_lr = 0.055
	decay_num_epochs = 6  # deprecated
	cosine_margin = 0  # deprecated
	# 'classifier_type = 'BatchGraphLevelClassifier'
	# 'classifier_type = 'BatchEqClassifier'
	classifier_type = 'TopLevelClassifier'  # deprecated
	combinator_type = 'SymmetricSumCombine'  # deprecated
	# ground_combinator_type = 'DummyGroundCombinator'
	ground_combinator_type = 'GroundCombinator'  # deprecated?
	# encoder_type = 'BatchEncoder'
	encoder_type = 'QbfEncoder'  # deprecated
	ggn_core = 'WeightedNegInnerIteration'  # deprecated
	# ggn_core = 'FactoredInnerIteration'
	# 'embedder_type = 'TopVarEmbedder'
	embedder_type = 'GraphEmbedder'  # deprecated
	# 'negate_type = 'minus'
	negate_type = 'regular'  # deprecated
	sparse = True  # deprecated
	# sparse = False
	gru_bias = False  # deprecated
	use_ground = False  # deprecated
	moving_ground = False  # deprecated
	split = False  # deprecated
	# cuda = True 
	cuda = False
	reset_on_save = False
	run_task = 'train'
	do_not_run = False
	do_not_learn = False
	restart_solver_every = 0				# minisat doesn't need restart
	# restart_solver_every = 3500
	restart_in_test = False
	non_linearity = 'torch.relu'
	policy_non_linearity = 'torch.relu'
	attn_non_linearity = 'F.tanh'
	episode_provider = 'UniformEpisodeProvider'
	evaluation_provider = 'OnePassProvider'
	test_every = 1000
	sync_every = 5
	is_sat=True
# RL - PG
	static_ed='eps/random_easy_train.eps'
	gamma = 0.99
	entropy_alpha = 0.000
	policy = 'Actor1Policy'
	policy_dim1 = 64
	policy_dim2 = 0
	batch_backwards = False  # deprecated					# Are we going to re-feed all states into the network in batch (True) or do the naive solution (False)
	# greedy_rewards = True
	greedy_rewards = False
	rl_log_dir = 'runs_cadet'
	rl_train_data = 'data/qbf_easy_train/'
	rl_validation_data = 'data/qbf_easy_test'
	rl_test_data = 'data/qbf_hard_test/'
	rl_log_envs = []
	rl_log_all = False
	rl_shelve_it = False
	rl_decay = False
	# rl_clip_episode_at = 100
	debug_actions = False
	debug = False
	profiling = False
	memory_profiling = False
	use_old_rewards = True					#  Must be false if using old cadet
	use_vsids_rewards = False
	use_sum = True
	leaky = False  # deprecated
	cadet_binary = './cadet'
	solver = 'cadet'
	# cadet_binary = './old_cadet' 			
	do_test = False
	test_iters = 100
	fresh_seed = False  						# Use a fresh seed in cadet
	use_seed = None
	follow_kl = False
	desired_kl = 1e-6
	min_timesteps_per_batch = 400  # replace by number of formulas per batch after normalization
	episodes_per_batch = 16
	ac_baseline = False
	stats_baseline = False
	use_global_state = True
	clause_learning = True
	vars_set = True
	use_gru = False  # deprecated
	use_bn = False
	use_ln = False
	state_bn = False 	# Note the default
	use_curriculum = False
	normalize_episodes = False
	parallelism = 1
	test_parallelism = 1
	full_pipeline = False
	packed = False
	masked_softmax = True 					# This chooses actions only from allowed actions
	slim_state=False
	episode_cutoff = 200
	use_state_in_vn = False	
	rnn_iters = 0
	learn_from_aborted = True
	check_allowed_actions = False
	grad_norm_clipping = 2
	pre_bias = False  # deprecate
	disallowed_aux = False
	lambda_disallowed = 1.
	lambda_value = 1.
	lambda_aux = 1.
	invalid_bias = -1000  # deprecate
	# invalid_bias = 0
	report_tensorboard = True
	report_window=5000
	report_uniform=True
	short_window=50
	report_every = 100
	mp=False
	main_loop = 'pyro_main'
	gpu_cap = 100
	policy_layers = [30,'r',1,'r']
	policy_initializer = None
	print_every = 0
	vbn_init_fixed = False
	vbn_module = True
	preload_formulas = False
	preload_cnf = False
	g2l_blacklist = []
	l2g_whitelist = []
	def_step_cost = -1.000e-04
	cadet_completion_reward = 1.
	cadet_use_activities = True
	report_last_envs = 10
	uncache_after_batch = False
	loglevel = 'logging.WARNING'
	autodelete_degenerate = True
	memory_cap=3000
	vbn_window=5000
	batch_size_threshold = 1.
	batch_sem_value = 3
	stale_threshold = 50
	check_stale = False
	fixed_bn = False
	fixed_bn_mean = 4.
	fixed_bn_std = 10.
	hp_normalize_shift = True	
	env_as_process = False
	recreate_env = False
	EPS_START = 0.9  # deprecated
	EPS_END = 0.03  # deprecated
	EPS_DECAY = 2000  # deprecated

# SAT
	
	sat_gc_freq = 'fixed'
	sat_reward_scale = 1e7
	sat_winning_reward = 10
	sat_min_reward = None
	sat_win_scale = 2.
	sat_reduce_base_provider = 'FixedReduceBaseProvider'
	sat_add_labels = False
	sat_add_embedding = True	
	sat_encoder_type = 'GINEncoder'
	sat_reduce_base = 2000
	sat_rb_min = 1000
	sat_rb_max = 4000	
	sat_cb_type = 'gc_oracle'							# 'branching_oracle'
	sat_trigger = 'step_cnt'							# 'op_cnt'
	sat_trigger_freq = 200
	sat_discrete_threshold_base = 2				# LBD thresholds start from this in SatDiscreteThresholdPolicy/SatFreeDiscretePolicy
	sat_num_free_actions = 29 						# Number of lbd thresholds in SatFreeDiscretePolicy
	minimum_episodes = 2
	do_lbd = False
	sat_random_p = 0.5
	disable_gnn = False
	threshold_sigma = 1.0
	threshold_scale = 1.0
	threshold_shift = 0.
	percentage_sigma = 0.1
	log_threshold = False
	init_threshold = None
	drop_abort_technical = False
	sat_balanced_override = False
	sat_balanced_sat = 'data/sc_synth_sat_combined'
	sat_balanced_unsat = 'data/sc_synth_unsat_combined'
	sat_init_percentage = False
	empty_step_length = 0.3
# RL - DQN
	learning_starts = 50000
	replay_size = 400000
	learning_freq = 4
	target_update_freq = 3000
	unit_length = 30 					# number of seconds per "round"
	weight_decay = 0.
# ES
	
	es_save_every = 10
	es_population = 50
	es_num_formulas = 5
	es_naive_stepsize = 1.
	es_train_data = 'data/sat_100'
	es_validation_data = 'data/sat_100'
	es_vanilla_policy = False

# Clause prediction
	cp_cmask_features = [2]		# By default mask lbd
	cp_vmask_features = []
	cp_invert_cmask = False
	cp_invert_vmask = False
	cp_nsat_use_labels = False
	cp_nsat_relaxed_msg = False
	cp_nsat_relaxed_update = False
	cp_num_samples = 1
	cp_num_layers = 3
	cp_normalization = None
	cp_encoder_type = 'NSATEncoder'
	cp_add_labels = True
	cp_add_gss = True
	cp_add_embedding = True
	cp_emb_dim = 128
	cp_use_gpu = False
	cp_save_every=5
	cp_cap_graph = False
	cp_num_categories = 2
	cp_clauses_sampled = 10
	cp_task = 'ever_used'			# 'ever_used' | lbd'
	smoke_test = False

# SharpSAT

	sharp_relaxed_msg = True
	sharp_relaxed_update = False
	sharp_decode = False
	sharp_decode_size = 9
	sharp_decode_class = 'SudokuModel1'
	sharp_completion_reward = 1
	sharp_encoder_type = 'GINEncoder'
	sharp_add_labels = False
	sharp_add_embedding = True
	sharp_emb_dim = 32
	sharp_decoded_emb_dim = 32
	sharp_log_actions = False
	sharp_random_policy = False
	sharp_time_random = False
	sharp_max_time = 5000
# Localization
	max_edges = 20

	DS_TRAIN_FILE = 'expressions-synthetic/split/%s-trainset.json' % dataset
	DS_VALIDATION_FILE = 'expressions-synthetic/split/%s-validationset.json' % dataset
	DS_TEST_FILE = 'expressions-synthetic/split/%s-testset.json' % dataset

	# return vars()

# ex.add_config({'gil': 'moshe'})