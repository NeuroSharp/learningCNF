import unittest

import sys, os

import time

from pysat.solvers import Glucose3
from pysat.formula import CNF

import numpy as np
from scipy.sparse import csr_matrix


np.set_printoptions(suppress=True, threshold=sys.maxsize)

c_map = ["id", "num_used", "size", "lbd", "activity", "blocked", "learnt", "tagged", "del"]

class ThreeValuedCallbackTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cnf = CNF(from_file='/Users/Haddock/Desktop/SAT-ML/Experiments/sat-rl/pysat-master/datasets/ecarev-37-200-11-84-26.cnf')
        cls.cb_counter = 0
        print("Finished Reading CNF file.")

    def prob_callback(self):
        self.cb_counter += 1
        if (self.cb_counter == 2): self.solver.terminate()

        num_cla = self.solver.nof_clauses(learnts=True)
        return np.array([1] * num_cla, dtype=np.double)

    def test_compare_to_default_oracle(self):
        # If all probs are set to be between 20 and 80 percent then the oracle should behave the same as the default oracle
        self.solver = Glucose3(gc_oracle = {"callback": self.prob_callback, "policy":"three_val"}, gc_freq="fixed", reduce_base=2000)
        self.solver.append_formula(self.cnf.clauses)
        self.solver.solve()
        num_learnts_prob = self.solver.nof_clauses(learnts=True)
        self.solver.delete()

        self.cb_counter = 0

        self.solver = Glucose3(gc_oracle = {"callback": self.prob_callback, "policy":"glucose"}, gc_freq="fixed", reduce_base=2000)
        self.solver.append_formula(self.cnf.clauses)
        self.solver.solve()
        num_learnts_default = self.solver.nof_clauses(learnts=True)

        self.assertEqual(num_learnts_prob, num_learnts_default, "Failure!")

    def tearDown(self):
        self.solver.delete()

if __name__ == '__main__':
    unittest.main()
