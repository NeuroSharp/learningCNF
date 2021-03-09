import unittest

import sys
import numpy as np

from pysat.solvers import Glucose3
from pysat.formula import CNF


np.set_printoptions(suppress=True, threshold=sys.maxsize)
np.random.seed(43)

c_map = ["id", "num_used", "size", "lbd", "activity", "blocked", "learnt", "tagged", "del"]

class CounterFactualTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.step_counter = 0
        cls.cnf = CNF(from_file='/Users/Haddock/Desktop/SAT-ML/Experiments/sat-rl/pysat-master/datasets/ecarev-37-200-11-84-26.cnf')
        print("Finished Reading CNF file.")


    def tagger_callback(self):
        self.step_counter += 1

        labels = self.solver.get_cl_labels(clause_type="learnt")
        self.nLearnts_tagged += [self.solver.nof_clauses(learnts=True) - np.count_nonzero(labels[:, 8])]

        action = [0.0] * self.solver.nof_clauses(learnts=True)

        if (self.step_counter == self.snapshot): # Tag clauses
            action = np.array([1.0] * self.tag_cnt + [0.0] * (self.solver.nof_clauses(learnts=True) - self.tag_cnt))
            np.random.shuffle(action)

        return action

    def glucose_callback(self):
        self.nLearnts_normal += [self.solver.nof_clauses(learnts=True)]
        return 123

    def test_tag_clause_effect(self): # Tagging clauses should not change the behaviour of Glucose
        self.snapshot = 2

        for self.tag_cnt in [0, 100, 200, 500]:
            print(f"Testing with tag count: {self.tag_cnt}")
            self.step_counter = 0
            self.nLearnts_tagged = []
            self.nLearnts_normal = []

            self.solver = Glucose3(gc_oracle = {"callback": self.tagger_callback, "policy":"counter_factual"}, gc_freq="fixed", reduce_base=2000)
            self.solver.append_formula(self.cnf.clauses)
            self.solver.solve()
            tagged_reward = self.solver.reward()
            self.solver.delete()


            self.solver = Glucose3(gc_oracle = {"callback": self.glucose_callback, "policy":"glucose"}, gc_freq="fixed", reduce_base=2000)
            self.solver.append_formula(self.cnf.clauses)
            self.solver.solve()
            normal_reward = self.solver.reward()
            self.solver.delete()

            self.assertEqual(tagged_reward, normal_reward, """The rewards should be the same in both tagged and untagged case.""")

            self.assertSequenceEqual(self.nLearnts_tagged, self.nLearnts_normal, """The number of learnt clauses in CF that are not marked for del
                should match the number of learnt clauses in default Glucose.""")



if __name__ == '__main__':
    unittest.main()
