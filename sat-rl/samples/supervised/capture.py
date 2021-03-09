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

c_map = {"id": 0, "num_used": 1, "size": 2, "lbd": 3,
         "activity": 4, "blocked": 5, "learnt": 6, "tagged": 7,
         "del": 8, "lbl:total_used": 9, "lbl:step_killed": 10}

def get_graph(adj_arrays, clause_labels, lit_labels):
    # print(np.shape(list(zip(adj_arrays["cols_arr"], adj_arrays["rows_arr"]))))
    G = dgl.heterograph(
                {('literal', 'l2c', 'clause') : list(zip(adj_arrays["cols_arr"], adj_arrays["rows_arr"])),
                 ('clause', 'c2l', 'literal') : list(zip(adj_arrays["rows_arr"], adj_arrays["cols_arr"]))},
                {'literal': len(lit_labels),
                 'clause': len(clause_labels)})

    G.nodes['literal'].data['lit_labels'] = lit_labels

    G.nodes['clause'].data['clause_ids']  = clause_labels[:, 0]
    G.nodes['clause'].data['clause_labels']  = clause_labels[:, 1:-4]
    G.nodes['clause'].data['clause_targets'] = clause_labels[:, -4:]

    return G

class Capture(object):
    def __init__(self, cnf, step, no_gc=-1, tag_cnt=500, reduce_base=2000):
        self.cnf = cnf
        self.no_gc = no_gc
        self.step = step
        self.tag_cnt = tag_cnt
        self.reduce_base = reduce_base
        self.step_counter  = 0

    def cold_run_callback(self):
        self.step_counter += 1

        return 1


    def snapshot_callback(self):
        self.step_counter += 1
        action = [0.0] * self.solver.nof_clauses(learnts=True)

        if (self.step_counter == self.snapshot):
            self.adj_arrays     = self.solver.get_cl_arr()
            self.cl_label_arr_l = self.solver.get_cl_labels(clause_type="learnt")
            # add the placeholder for future features here: step killed & number of times used
            self.cl_label_arr_l = np.c_[self.cl_label_arr_l, np.zeros((len(self.cl_label_arr_l), 2))]
            self.cl_label_arr_o = self.solver.get_cl_labels(clause_type="orig")
            self.cl_label_arr_o = np.c_[self.cl_label_arr_o, np.zeros((len(self.cl_label_arr_o), 2))]
            self.lit_label_arr  = self.solver.get_lit_labels()
            self.gss            = self.solver.get_solver_state()

            action = np.array([1.0] * self.tag_cnt + [0.0] * (self.solver.nof_clauses(learnts=True) - self.tag_cnt))
            np.random.shuffle(action)


        if (self.step_counter > self.snapshot):
            self.update_labels()

        return action

    def update_labels(self):
        cc_np = self.solver.get_cl_labels(clause_type="learnt")
        current_clauses = pd.DataFrame(data=cc_np, index=cc_np[:,0])
        for old_cl in self.cl_label_arr_l:
            if (old_cl[c_map["id"]] in current_clauses.index):
                new_cl = current_clauses.loc[old_cl[c_map["id"]]]
                old_cl[c_map['del']] = new_cl[c_map['del']]
                old_cl[c_map['tagged']] = new_cl[c_map['tagged']]
                old_cl[c_map['lbl:total_used']] = new_cl[c_map['num_used']]

                if (old_cl[c_map["del"]] and old_cl[c_map["lbl:step_killed"]] == 0.0): # the tagged clause is marked for delete. Record the step
                    # print(f"""{new_cl[c_map["del"]]} and {old_cl[c_map["lbl:step_killed"]]}""")
                    old_cl[c_map["lbl:step_killed"]] = self.step_counter
            elif (old_cl[c_map["lbl:step_killed"]] == 0.0):
                old_cl[c_map["lbl:step_killed"]] = self.step_counter


    def capture(self):
        if (self.no_gc < 0):
            print("Running a cold run to find the number of GC calls (no_gc)...")

            self.solver = Glucose3(gc_oracle = {"callback": self.cold_run_callback, "policy": "glucose"}, gc_freq="fixed", reduce_base=self.reduce_base)
            self.solver.append_formula(self.cnf.clauses)
            self.solver.solve()
            self.solver.delete()
            self.solver = None
            self.no_gc = self.step_counter
            print("Total number of GC calls: %i" % self.no_gc)

        self.snapshot = self.step if (self.step > 0) else (self.no_gc + self.step)
        if (self.snapshot <= 0 or self.snapshot >= self.no_gc):
            raise Exception("Bad step value. The resulting snapshot (%i) is not between 0 and total GC counts (%i)." % (self.snapshot, self.no_gc))
            return None

        self.step_counter = 0
        self.solver = Glucose3(gc_oracle = {"callback": self.snapshot_callback, "policy": "counter_factual"}, gc_freq="fixed", reduce_base=self.reduce_base)
        self.solver.append_formula(self.cnf.clauses)
        self.solver.solve()
        self.update_labels()

        cl_label_arr = np.r_[self.cl_label_arr_o, self.cl_label_arr_l] # appending the learnt clause labels to the end of the problem clause labels
        # g_dgl = get_graph(self.adj_arrays, cl_label_arr, self.lit_label_arr)

        return {
            "no_gc": self.no_gc,
            "snapshot": self.snapshot,
            "gss": self.gss,
            "adjacency": self.adj_arrays,
            "lit_feat": self.lit_label_arr,
            "clause_feat_and_labels": cl_label_arr,
            # "graph": g_dgl
        }


def capture(cnf, step, no_gc=-1, tag_cnt=500, reduce_base=2000):
    """
    Captures a snapshot of the solver's graph and features at a GC steps and fills in
    the total number of times a clause was used (total_used) as well as the step in
    which the clause was deleted (step_killed). If the clause is not deleted until the
    end that value will remain 0.

    Input:
        cnf: a CNF object
        step: the step to take the snapshot at:
            - if step < 0: count steps from the end
            - if step > 0: count the steps from the beginning
            (The method first runs the solver to find the number of steps (no_gc).)
        no_gc: if the total number of GC steps is known it can be passed in to save the method
            from having to do the 'cold_run' which is running the solver to find the number of steps.

    Output:
        A dictionary:
            no_gc: the number of steps (GC calls). This can be used to speed up process
                in the next call
            snapshot: The step at which the current snapshot was taken.
            gss: The Global Solver State in a blob format.
            graph: DGL graph containing the adjacency graph as well as clause/literal labels:
                G.nodes['literal'].data['lit_labels']: literal labels.
                G.nodes['clause'].data['clause_ids']:  clause uid's.
                G.nodes['clause'].data['clause_labels']: clause features
                G.nodes['clause'].data['clause_targets']: supervised learning labels (total_used, step_killed)

    Example usage:

        >>> # from supervised.capture import capture
        >>> from pysat.formula import CNF
        >>>
        >>> f1 = CNF(from_file='../pysat-master/datasets/ecarev-37-200-11-84-26.cnf.bz2')
        >>> a = capture(f1, 3)
        Running a cold run to find the number of GC calls (no_gc)...
        Total number of GC calls: 10
        >>> G = get_graph(a["adjacency"], a["clause_feat_and_labels"], a["lit_feat"])
        >>> print("uniq  lits %i" % len(np.unique(a["adjacency"]["cols_arr"])))
        uniq  lits 33484
        >>> print("total lits %i" % G.number_of_nodes('literal'))
        total lits 34000
        >>> print("uniq  clas %i" % len(np.unique(a["adjacency"]["rows_arr"])))
        uniq  clas 102310
        >>> print("total clas %i" % G.number_of_nodes('clause'))
        total clas 102310

        >>> G.nodes['clause'].data['clause_targets'][101827]
        array([143.,   0.])
        >>> G.nodes['clause'].data['clause_labels'][101827]
        array([   5.        ,    4.        ,    2.        , 1126.28295898,
                  0.        ,    1.        ])
        >>> G.out_degree(101827, 'c2l')
        4
        >>> G.out_degree(101, 'l2c')
        8
        >>> a["gss"]
        array([0.13372331, 0.44786695, 0.5688547 , 0.51578989, 0.34173734,
               0.20801403, 0.15707182, 0.07429073, 0.03183888, 0.02122592,
               0.01061296, 0.00636778, 0.00424518, 0.00424518, 0.        ,
               0.        , 0.        , 0.        , 0.        , 0.        ,
               0.        , 0.        , 0.        , 0.        , 0.        ,
               0.        , 0.        , 0.        , 0.        , 0.        ,
               0.13333759, 0.42964334, 0.61165973, 0.52911742, 0.29418928,
               0.18413286, 0.12063877, 0.07830938, 0.03809645, 0.02328117,
               0.01481529, 0.00634941, 0.00634941, 0.00634941, 0.00423294,
               0.00211647, 0.        , 0.00634941, 0.        , 0.        ,
               0.        , 0.        , 0.        , 0.        , 0.        ,
               0.        , 0.        , 0.        , 0.        , 0.        ,
               0.20827442, 0.79721039, 0.45756287, 0.26210534, 0.16021109,
               0.09997172, 0.06985204, 0.04357742, 0.0173028 , 0.01345773,
               0.00640844, 0.00512675, 0.00320422, 0.00320422, 0.00128169,
               0.00064084, 0.        , 0.00192253, 0.        , 0.        ,
               0.        , 0.        , 0.        , 0.        , 0.        ,
               0.        , 0.        , 0.        , 0.        , 0.        ])
"""
    return Capture(cnf, step, no_gc, tag_cnt, reduce_base).capture()


if __name__ == '__main__':
    import doctest
    doctest.testmod()




