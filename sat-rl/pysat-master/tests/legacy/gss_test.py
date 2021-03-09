import math
import subprocess
import json

from pysat.solvers   import Glucose3
from pysat.formula   import CNF
from pysat.callbacks import *

import numpy as np
from scipy.sparse import csr_matrix


nbReduceDB = 0

""" This tests runs both deep and pysat glucose versions and 
    makes sure that they see the same Global Solver State (GSS)
    values.
	
	In order to avoid descrepencies based on the GC model, we 
	test both version in vanilla mode (no-deep-reduce-db).
"""

def cb():
	passed = True
	printGss = False

	global nbReduceDB
	nbReduceDB += 1

	gss = m.get_solver_state()
	# with np.printoptions(precision=3, suppress=True):
		# print(np.shape(gss[21:]))

	# if (printGss):
	# 	print("learnts/orig ratio: {}".format(gss[0][0]))
	# 	print("conflicts/vars ratio: {}".format(gss[0][1]))
	# 	print("avg sumLBD: {}".format(gss[0][2]))
	# 	print("------------------")
	# 	print("avg trail_sz: {}".format(repr(gss[1])))
	# 	print("avg decl_sz: {}".format(repr(gss[2])))
	# 	print("avg trail/decl: {}".format(repr(gss[3])))
	# 	print("------------------")

	# 1. Test LBD histograms:
	# hists = np.split(gss[21:], 6)
	hists = np.reshape(gss[21:], (30, 6)).T
	print(np.shape(hists))
	for hist in hists:
		# print(np.shape(hist))
		# print(hist)
		hist_sum = np.sum(hist, axis=0)
		# 1.1. Values in LBD historgram should sum to 1 (except for the beginning when they can be 0).
		if (not (math.isclose(hist_sum, 1, rel_tol=1e-6) 
				or hist_sum == 0.0)): # hist_sum can be zero at the beginning
			print("[Test Failed!] Values in LBD histogram should sum up to 1. Observed: {}".format(hist_sum))
			print(hist)
			passed = False

		# 1.2. All values in LBD historgram should be between 0 and 1.
		if (np.any(hist < 0)):
			print("[Test Failed!] Values in LBD histogram should be >= 0.")
			passed = False

	if (passed): print("[Test Passed!]")

	return 123456

# cnf_file = '/Users/Haddock/Desktop/fast_glucose/unsat/newton_3_4_true-unreach-call.i.cnf'
# cnf_file = '/Users/Haddock/Desktop/fast_glucose/final/CNP-5-1500.cnf'

# cnf_file = 'datasets/10pipe_k[3-30]-89064.cnf'
cnf_file = "datasets/mp1-9_38[38-1000]-75490.cnf"

print("Running the tests...")
f1 = CNF(from_file=cnf_file)
m  = Glucose3(gc_oracle = {"callback": cb, "policy": "glucose"}, gc_freq='fixed', reduce_base=2000)
m.append_formula(f1.clauses)
m.solve()