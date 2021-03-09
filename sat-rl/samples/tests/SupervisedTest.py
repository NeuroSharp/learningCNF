import unittest

import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from supervised.capture import capture
from pysat.formula import CNF

# np.random.seed(42)

c_map = {"id": 0, "num_used": 1, "size": 2, "lbd": 3,
         "activity": 4, "blocked": 5, "learnt": 6, "tagged": 7,
         "del": 8, "lbl:total_used": 9, "lbl:step_killed": 10}

class SupervisedTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.step_counter = 0
        cls.cnf = CNF(from_file='/Users/Haddock/Desktop/SAT-ML/Experiments/sat-rl/pysat-master/datasets/ecarev-37-200-11-84-26.cnf')
        print("Finished Reading CNF file.")

    def test_tagged(self):
        self.tag_cnt = 500
        tagged_res = capture(self.cnf, 2, tag_cnt=self.tag_cnt)
        labels_tagged = tagged_res["clause_feat_and_labels"]

        # print(labels_tagged[:, c_map["tagged"]])
        self.assertEqual(np.count_nonzero(labels_tagged[:, c_map["tagged"]]), self.tag_cnt,
            "The number of tagged clauses should be preserved.")

        # This checks: del -> (step_killed > 0). Which is equivalent to: ~del v step_killed/
        condition = np.logical_or(np.logical_not(labels_tagged[:, c_map["del"]] == 1.0),
                                  labels_tagged[:, c_map["lbl:step_killed"]] > 0)
        self.assertEqual(np.all(condition), True,
                                 "If a clause is marked for del then it should have a proper lbl:step_killed > 0.")

        # This checks: del -> (tagged > 0). Which is equivalent to: ~del v tagged/
        condition = np.logical_or(np.logical_not(labels_tagged[:, c_map["del"]] == 1.0),
                                  labels_tagged[:, c_map["tagged"]] > 0)
        self.assertEqual(np.all(condition), True,
                                 "If a clause is marked for del then it must be tagged.")

        # Compare with the tag_cnt = 0
        untagged_res = capture(self.cnf, 2, tag_cnt=0)
        labels_untagged = untagged_res["clause_feat_and_labels"]
        self.assertSequenceEqual(labels_tagged[:, c_map["lbl:step_killed"]].tolist(),
                                 labels_untagged[:, c_map["lbl:step_killed"]].tolist(),
                                 "The same step_killed should be recorded for tagged and untagged cases.")

        # If nothing is marked then the del should be empty

    # def test_no_tag(self):
    #     f1 = CNF(from_file='../pysat-master/datasets/ecarev-37-200-11-84-26.cnf.bz2')
    #     a = capture(f1, 2)


if __name__ == '__main__':
    unittest.main()
