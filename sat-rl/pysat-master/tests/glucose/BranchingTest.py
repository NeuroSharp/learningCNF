import unittest

import sys, time
import numpy as np
import pandas as pd

from pysat.solvers import Solver, Glucose3
from pysat.formula import CNF


np.set_printoptions(suppress=True, threshold=sys.maxsize)

class BranchingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cnf = CNF(from_file='/Users/Haddock/Desktop/SAT-ML/Experiments/sat-rl/pysat-master/datasets/ecarev-37-200-11-84-26.cnf')
        print("Finished Reading CNF file.")

    def trigger_callback(self):
        if (self.trigger =="step_cnt"):
            self.assertEqual(self.solver.reward()%self.trigger_freq, 0,
                    "The step_cnt trigger is not fired at the proper time.")

        if (self.trigger =="op_cnt"):
            self.assertEqual(self.solver.reward()%self.trigger_freq, 0,
                    "The op_cnt trigger is not fired at the proper time.")

        return -1


    def test_step_cnt_trigger(self):
        self.trigger = "step_cnt"
        self.trigger_freq = 200
        self.solver = Glucose3(branching_oracle = {"callback": self.trigger_callback,
            "trigger": self.trigger, "trigger_freq": self.trigger_freq})
        self.solver.append_formula(self.cnf.clauses)
        self.solver.solve()

    def test_step_cnt_trigger_default(self):
        self.trigger = "step_cnt"
        self.trigger_freq = 1
        self.solver = Glucose3(branching_oracle = {"callback": self.trigger_callback})
        self.solver.append_formula(self.cnf.clauses)
        self.solver.solve()

    # def test_op_cnt_trigger(self):
    #     self.trigger = "op_cnt"
    #     self.trigger_freq = 2000
    #     self.solver = Glucose3(branching_oracle = {"callback": self.trigger_callback,
    #         "trigger": self.trigger, "trigger_freq": self.trigger_freq})
    #     self.solver.append_formula(self.cnf.clauses)
    #     self.solver.solve()

    def tearDown(self):
        self.solver.delete()

if __name__ == '__main__':
    unittest.main()
