import sys
import os
import psutil

from pysat.solvers   import Glucose3
from pysat.formula   import CNF
from pysat.callbacks import lbd_based_cb

import numpy as np
from scipy.sparse import csr_matrix

# ----------------------------------------------
# NOTE: Make sure your build is NOT IN DEBUGMODE
# ----------------------------------------------

# A simple stress test to check pressence of memeory leaks
# Check the rate of growth of the memory consumption. It should
# plateau around 50MB...

process = psutil.Process(os.getpid())
def cb_glucose():
	global m
	m.get_solver_state()

	return 123

# def cb_lbd():
# 	global m
# 	m.get_solver_state()

# 	return lbd_based_cb(cl_label_arr, rows_arr, cols_arr, data_arr)

def cb_perc_term():
	global m
	m.get_solver_state()

	# m.terminGate()
	return [0.5, 2]

def cb_perc():
	global m
	m.get_solver_state()
	return [0.5, 2]

f1 = CNF(from_file='datasets/frb30-15-1.cnf')

def test(gc_oracle):
	for i in range(0,500):
		print("iter: {0}, memory: {1:.2f}MB".format(
			i, process.memory_info().rss / float(2 ** 20)))
		
		global m
		global counter
		m = Glucose3(gc_oracle = gc_oracle, reduce_base=50)
		m.append_formula(f1.clauses)

		# Add the functions to test for memory leak here:
		# adj_matrix 	   = m.get_cl_arr()
		# orig_cl_labels = m.get_cl_labels()
		# var_labels     = m.get_var_labels()
		# -----------------------------------------------

		m.solve()
		m.delete()
		m = None

print("Testing with percentage policy with terminate() call...")
test({"callback":cb_perc_term, "policy":"percentage"})

print("Testing with percentage policy...")
test({"callback":cb_perc, "policy":"percentage"})

# print("Testing with lbd_threshold policy...")
# test({"callback":cb_lbd, "policy":"lbd_threshold"})

print("Testing with glucose (default) policy...")
test({"callback":cb_glucose, "policy":"glucose"})
