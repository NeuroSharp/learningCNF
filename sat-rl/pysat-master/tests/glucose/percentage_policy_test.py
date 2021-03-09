from pysat.solvers   import Glucose3
from pysat.formula   import CNF

import numpy as np

# import torch

""" This tests runs pysat once with the default gc and once with
	percentage policy with parameters (50% and 2). This should result
	in the same behaviour. We test that here.
"""

nbReduceDB = 0
reward_buffer = []
gss_buffer = []

# def cb_glucose(cl_label_arr, rows_arr, cols_arr, data_arr):
# 	global cl_label_buffer
# 	global gss_buffer

# 	gss_buffer.append(m.get_solver_state())
# 	reward_buffer.append(m.reward())

# 	return 123456789


# def cb_percentage(cl_label_arr, rows_arr, cols_arr, data_arr):
# 	global reward_buffer
# 	global gss_buffer
# 	global nbReduceDB

# 	current_gss = m.get_solver_state()
# 	saved_gss = gss_buffer[nbReduceDB]

# 	current_reward = m.reward()
# 	saved_reward = reward_buffer[nbReduceDB]

# 	passed = True
# 	for cur, saved in zip(current_gss, saved_gss):
# 		try:
# 			np.testing.assert_array_almost_equal(cur, saved, decimal=6)
# 		except AssertionError as e:
# 			print("[Test Failed!]")
# 			print(e)
# 			passed = False

# 	if current_reward != saved_reward:
# 		print("[Test Failed!]")
# 		print("Reward mismatch. Expected: {}, observed: {}.".format(saved_reward, current_reward))
# 		passed = False

# 	if (passed):
# 		print("[Test Passed!]")

# 	nbReduceDB += 1
# 	# return torch.FloatTensor([0.5, 2]).tolist()
# 	return np.array([0.5, 2])
np.set_printoptions(suppress=True)

import pprint
pp = pprint.PrettyPrinter(indent=4)
def cb_percentage():
	gss = m.get_solver_state()
	# pp.pprint(gss)

	# exit(0)

	# print(f"dl: {gss['decision_level_avg']}")
	# print(f"tr: {gss['trail_size_avg']}")
	print(f"hist_r_0: {gss['lbd_hist_recent_0']} : sum({gss['lbd_hist_recent_0'].sum()})")
	print(f"hist_r_1: {gss['lbd_hist_recent_1']} : sum({gss['lbd_hist_recent_1'].sum()})")
	print(f"hist_t_0: {gss['lbd_hist_total_0']}  : sum({gss['lbd_hist_total_0'].sum()}), avg({gss['lbd_avg']})")
	# print("==================================================")
	# print(f"sum py: {gss['lbd_hist_total_0'].sum()}")
	# return np.array([0.5, 2])
	return np.array([5])

# cnf_file = 'datasets/10pipe_k[3-30]-89064.cnf.bz2'
# cnf_file = 'datasets/ecarev-37-200-11-84-26.cnf'
cnf_file = "datasets/mp1-9_38[38-1000]-75490.cnf"
# cnf_file = '/Users/Haddock/Desktop/fast_glucose/test/UCG-20-5p1[1-175]-11325.cnf.bz2'
f1 = CNF(from_file = cnf_file)

# m = Glucose3(gc_oracle = {"callback":cb_glucose, "policy": "glucose"})
# m.append_formula(f1.clauses)
# m.solve()

m = Glucose3(gc_oracle = {"callback": cb_percentage, "policy": "glucose"}, gc_freq="fixed", reduce_base=2000)
# m = Glucose3()
m.append_formula(f1.clauses)
m.solve()
# print(m.get_stats())
