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

def cb(cl_label_arr, rows_arr, cols_arr, data_arr):
	passed = True
	printGss = False

	global nbReduceDB
	nbReduceDB += 1

	gss = m.get_solver_state()

	if (printGss):
		print("learnts/orig ratio: {}".format(gss[0][0]))
		print("conflicts/vars ratio: {}".format(gss[0][1]))
		print("avg sumLBD: {}".format(gss[0][2]))
		print("------------------")
		print("avg trail_sz: {}".format(repr(gss[1])))
		print("avg decl_sz: {}".format(repr(gss[2])))
		print("avg trail/decl: {}".format(repr(gss[3])))
		print("------------------")

	# Test LBD histograms
	hists = gss[4]
	for hist in hists.T:
		if (printGss):
			print(repr(hist))

		hist_sum = np.sum(hist, axis=0)
		if (not (math.isclose(hist_sum, 1, rel_tol=1e-6) 
				or hist_sum == 0.0)): # hist_sum can be zero at the beginning
			print("[Test Failed!] Values in LBD histogram should sum up to 1. Observed: {}".format(hist_sum))
			passed = False

		if (np.any(hist < 0)):
			print("[Test Failed!] Values in LBD histogram should be >= 0.")
			passed = False

	saved_feat  = json.load(open("_glucose_rdb_{}.tmp".format(nbReduceDB)))['feat']
	deep_feat   = np.ravel(np.array(saved_feat))
	python_feat = np.array([item for sublist in gss[:4] for item in sublist] + gss[4].flatten().tolist())

	try:
		np.testing.assert_array_almost_equal(deep_feat, python_feat, decimal=6)
	except AssertionError as e:
		print("[Test Failed!]")
		print(e)
		passed = False


	if (passed):
		print("[Test Passed!]")
	if (printGss):
		print("============================================")


	return [0.5, 2]

# cnf_file = '/Users/Haddock/Desktop/fast_glucose/unsat/newton_3_4_true-unreach-call.i.cnf'
# cnf_file = '/Users/Haddock/Desktop/fast_glucose/final/CNP-5-1500.cnf'

cnf_file = 'datasets/10pipe_k[3-30]-89064.cnf'
glucose_path = '../deep-solvers/deep-glucose-3.0/'

glucose_task = [glucose_path + "core/glucose_debug", "-firstReduceDB=5000", "-use_deep_rdb", "-gcFreqPolicy=1", "-gcPolicy=2", "-verb=2", cnf_file]
print("Solving the instance using deep-glucose (Constant Percentage):")
print(">>> {}".format(" ".join(glucose_task)))
subprocess.run(glucose_task)

print("Running the tests...")
f1 = CNF(from_file=cnf_file)
m  = Glucose3(gc_oracle = {"callback": cb, "policy": "glucose"}, gc_freq='glucose', reduce_base=5000)
m.append_formula(f1.clauses)
m.solve()

subprocess.run(["rm _glucose_rdb_*"], shell=True)