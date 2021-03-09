import sys

from pysat.solvers   import Minisat22
from pysat.formula   import CNF
from pysat.callbacks import *

import numpy as np
from scipy.sparse import csr_matrix

counter = 0
np.set_printoptions(threshold=np.nan)

def testify(callback):
	def testified_cb(cl_label_arr, rows_arr, cols_arr, data_arr):
		adj_matrix = csr_matrix((data_arr, (rows_arr, cols_arr)))
		nLearnts = len(cl_label_arr)
		
		global counter

		counter += 1
		print("reward: {}".format(m.reward()))
		print("callback calls counter: {}".format(counter))
		if (counter == 5): m.terminate()

		return callback(cl_label_arr, rows_arr, cols_arr, data_arr)

	return testified_cb

print("============================")
print("     RUN IN DEBUGMODE       ")
print("============================")
print()

f1 = CNF()
f1.from_file('datasets/CNP-5-0.cnf.bz2', compressed_with="bzip2")

# Keeps all the clauses
print("Keep_all callback:")
print("____________________________")
m = Minisat22(callback=testify(keep_learnts_cb))
m.append_formula(f1.clauses)
print(m.solve())
m.delete()
counter = 0
print()

# This 'lbd_based_cb' callback mimics the logic of LBD
print("LBD-based reduction callback:")
print("_____________________________")
m = Minisat22(callback=testify(lbd_based_cb))
m.append_formula(f1.clauses)
print(m.solve())
m.delete()
counter = 0
print()

# This one calls the LBD-based reduction function but does not set reduce_base
print("Empty callback (reduce_base=50):")
print("________________________________")
m = Minisat22(callback=testify(empty_cb))
m.append_formula(f1.clauses)
print(m.solve())
m.delete()
counter = 0
print()

# This one calls the LBD-based reduction function and sets reduce_base=500
print("Empty callback (reduce_base=500):")
print("_________________________________")
m = Minisat22(callback=testify(empty_cb), reduce_base=500)
m.append_formula(f1.clauses)
print(m.solve())
m.delete()
counter = 0
print()

# This is the default Minisat LBD-based reduction
print("No Callback (Default LBD):")
print("__________________________")
m = Minisat22()
m.append_formula(f1.clauses)
print(m.solve())
m.delete()
counter = 0