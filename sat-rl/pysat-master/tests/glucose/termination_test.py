import sys

from pysat.solvers import Glucose3
from pysat.formula import CNF

import numpy as np
from scipy.sparse import csr_matrix

np.set_printoptions(threshold=np.nan)

counter =0
reward = 0
counter_max = 5

def cb(cl_label_arr, rows_arr, cols_arr, data_arr):
	adj_matrix = csr_matrix((data_arr, (rows_arr, cols_arr)))
	nLearnts = len(cl_label_arr)
	
	global counter
	global reward

	counter += 1
	print("callback calls counter: {}".format(counter))
	reward = m.reward()
	if (counter == counter_max): m.terminate()

	return [0.5, 2]

cnf_file = 'datasets/10pipe_k[3-30]-89064.cnf.bz2'
f1 = CNF(from_file=cnf_file)

m = Glucose3(gc_oracle = {"callback": cb, "policy": "percentage"}, gc_freq='fixed', reduce_base=300)
m.append_formula(f1.clauses)

m.solve()

if (m.reward() != reward):
	print("[Test Failed!]")
	print("Asynchronous termination! The Solver did not quit right after terminate() call.")
	print("The rewards mismatch:")
	print("Reward from callback: {}".format(reward))
	print("Reward after termination: {}".format(m.reward()))
elif (counter != counter_max):
	print("[Test Failed!]")
	print("Asynchronous termination! The Solver did not quit right after terminate() call.")
	print("More callback calls than counter_max")
else:
	print("[Test Passed!]")
# Even though we called terminate we still have access to solver methods. Uncomment to run:
# print("No of clauses: {}".format(m.nof_clauses()))
# print("Clause labels: {}".format(m.get_cl_labels()))
# print("Variable Labels: {}".format(m.get_var_labels()))

m.delete()
