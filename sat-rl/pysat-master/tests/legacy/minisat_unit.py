from __future__ import print_function
import sys, os

from pysat.solvers import Minisat22
from pysat.formula import CNF
from pysat.callbacks import *

import numpy as np
from scipy.sparse import csr_matrix

TEMP_FILE = "_minisat.tmp"

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def callback(cl_label_arr, rows_arr, cols_arr, data_arr):
    # var_label_arr: activity and partial assignment
    # cl_label_arr:  index, num_used, size, lbd, activity and locked
    # TODO: we can also send "recent" num_used and "recent" activity
    expected = []
    passed   = True

    if os.path.exists(TEMP_FILE):
        file = open(TEMP_FILE, "r")
        expected = [line.rstrip('\n') for line in file]
    else:
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), TEMP_FILE)
        eprint("[Test Failed!] File {} does not exist.".format(path))
        eprint("Make sure the solver is compiled in DEBUGMODE")

        return empty_cb(cl_label_arr, rows_arr, cols_arr, data_arr)

    adj_matrix = csr_matrix((data_arr, (rows_arr, cols_arr)))

    def formatf(x):
        return '{0:.3f}'.format(x)
    for cl_i, cl in enumerate(cl_label_arr):
        lits = [(a+1)*b for a,b in zip(adj_matrix[cl_i,:].nonzero()[1], 
                                        adj_matrix[cl_i,:].data)]
        observed = "{},{}[{}]".format(
            ",".join(map(lambda x: str(int(x)), cl[1:4])), 
            formatf(cl[4]), # Activity is a float
            ",".join(map(str, lits)))
        if (expected[cl_i] != observed):
            passed = False
            eprint("[Test Failed!] Bad clause information on python side.")
            eprint(cl[1:])
            eprint("Expected:{};".format(expected[cl_i]))
            eprint("Observed:{};".format(observed))
            break

    if (passed): 
        print("[Test Passed!]")

    if os.path.exists(TEMP_FILE):
        os.remove(TEMP_FILE)

    return empty_cb(cl_label_arr, rows_arr, cols_arr, data_arr)


# f1 = CNF(from_file='datasets/CBS_k3_n100_m403_b10_42.cnf')
f1 = CNF(from_file='datasets/frb30-15-1.cnf')
m = Minisat22(callback=callback)
m.append_formula(f1.clauses)
m.solve()
