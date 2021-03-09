import unittest

import sys, os
import dgl
import networkx as nx

import time
import pandas as pd
import matplotlib.pyplot as plt

from pysat.solvers import Glucose3
from pysat.formula import CNF

import numpy as np
from scipy.sparse import csr_matrix


np.set_printoptions(suppress=True, threshold=sys.maxsize)

c_map = ["id", "num_used", "size", "lbd", "activity", "blocked", "learnt", "tagged", "del"]

class ClauseCallbackTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cnf = CNF(from_file='/Users/Haddock/Desktop/SAT-ML/Experiments/sat-rl/pysat-master/datasets/ecarev-37-200-11-84-26.cnf')
        print("Finished Reading CNF file.")

    def lbd_size(self, levels):
        (unique, counts) = np.unique(levels["level"], return_counts=True)
        return f"{len(levels)}, {len(counts)}"

    def edge_callback(self):
        cl_arr_temp = self.solver.get_cl_arr()
        cl_arr_df = pd.DataFrame(data=cl_arr_temp["data_arr"], columns=["id", "level"])
        cl_arr_df = cl_arr_df.groupby(['id']).apply(self.lbd_size).reset_index(name='calc').set_index("id")
        cl_arr_df = cl_arr_df.calc.str.split(", ", expand=True)
        cl_arr_df["calc_size"] = cl_arr_df[0].astype('float64')
        cl_arr_df["calc_lbd"] = cl_arr_df[1].astype('float64')
        cl_arr_df.drop(columns =[0, 1], inplace = True)

        cl_label_arr = self.solver.get_cl_labels(clause_type='all')
        labels_df = pd.DataFrame(data=cl_label_arr, index=cl_label_arr[:,0], columns=c_map)

        joined = labels_df.join(cl_arr_df)
        self.assertTrue((joined['lbd']  == joined['calc_lbd']).all(),
            "The C-side LBD is not equal to the LBD calculated using the edge features on Python side.")
        self.assertTrue((joined['size'] == joined['calc_size']).all(),
            "The size of the clause from the clause labels is different than the number of literals in the edge feature.")

        self.solver.terminate()
        return 123

    def alignment_callback(self):
        cl_arr_temp = self.solver.get_cl_arr()
        cl_arr = csr_matrix((cl_arr_temp["data_arr"][:, 0], (cl_arr_temp["rows_arr"], cl_arr_temp["cols_arr"])))

        cl_label_arr = self.solver.get_cl_labels(clause_type='all')

        self.assertEqual(cl_label_arr.shape[0], cl_arr.shape[0],
            "Size mismatch between the clause_label_array and clause_array.")

        old_row = -1
        for (row, col) in zip(cl_arr_temp["rows_arr"], cl_arr_temp["cols_arr"]):
            # avoid searching the same row (i.e. clause in the dense version of the csr matrix)
            if (old_row == row): continue

            self.assertEqual(cl_label_arr[row][0], cl_arr[row, col],
                "Clause id mismatch: clause_label_array is no aligned with the clause_array.")

            old_row = row

        self.solver.terminate()
        return 123

    def test_edge_features(self):
        self.solver = Glucose3(gc_oracle = {"callback": self.edge_callback, "policy":"glucose"}, gc_freq="fixed", reduce_base=2000)
        self.solver.append_formula(self.cnf.clauses)
        self.solver.solve()

    def test_cvig_label_alignment(self):
        self.solver = Glucose3(gc_oracle = {"callback": self.alignment_callback, "policy":"glucose"}, gc_freq="fixed", reduce_base=2000)
        self.solver.append_formula(self.cnf.clauses)
        self.solver.solve()

    def tearDown(self):
        self.solver.delete()

if __name__ == '__main__':
    unittest.main()
