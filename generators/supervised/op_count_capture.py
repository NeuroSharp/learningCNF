import sys
import os
import dgl
import pickle
import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd

from pysat.solvers import Glucose3
from pysat.formula import CNF

from gen_types import FileName
from gen_utils import random_string


np.set_printoptions(suppress=True, threshold=sys.maxsize)

c_map = {"id": 0, "num_used": 1, "size": 2, "lbd": 3,
         "activity": 4, "blocked": 5, "learnt": 6, "tagged": 7,
         "del": 8, "lbl:total_used": 9, "lbl:step_killed": 10}

def get_graph(adj_arrays, clause_labels, lit_labels):
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

class OpCountCapture(object):
    def __init__(self, fname: FileName, no_gc=-1, reduce_base=2000, **kwargs):
        self.cnf = fname
        self.no_gc = no_gc
        self.reduce_base = reduce_base
        self._step_counter = 0
        self.snapshot = -1
        self.tagged_id = -1

        if (self.no_gc < 0):
            print("Running a cold run to find the number of GC calls (no_gc)...")

            self.solver = Glucose3(gc_oracle = {"callback": self.cold_run_callback, "policy": "glucose"}, gc_freq="fixed", reduce_base=self.reduce_base)
            self.solver.append_formula(CNF(from_file=self.cnf).clauses)
            self.solver.solve()
            self.solver.delete()
            self.solver = None
            self.no_gc = self._step_counter
            print(f"Total number of GC calls: {self.no_gc}")


    def cold_run_callback(self):
        self._step_counter += 1

        return 1

    def snapshot_callback(self):
        self._step_counter += 1
        n_learnts = self.solver.nof_clauses(learnts=True)
        action = [0.0] * n_learnts

        if (self._step_counter == self.snapshot):
            # Record all the features
            self.adj_arrays     = self.solver.get_cl_arr()
            self.cl_label_arr_l = self.solver.get_cl_labels(clause_type="learnt")
            # Add the placeholder for future features here: step killed & number of times used
            self.cl_label_arr_l = np.c_[self.cl_label_arr_l, np.zeros((len(self.cl_label_arr_l), 2))]
            self.cl_label_arr_o = self.solver.get_cl_labels(clause_type="orig")
            self.cl_label_arr_o = np.c_[self.cl_label_arr_o, np.zeros((len(self.cl_label_arr_o), 2))]
            self.lit_label_arr  = self.solver.get_lit_labels()
            self.gss            = self.solver.get_solver_state()

            # Tag a clause at random
            blocked = True
            while (blocked): # Don't choose blocked clause
                self.tagged_id = np.random.choice(n_learnts, 1)[0]
                blocked = self.cl_label_arr_l[self.tagged_id][c_map["blocked"]]
            action[self.tagged_id] = 1.0

        if (self._step_counter > self.snapshot):
            self.update_labels()

        return action

    def delete_callback(self):
        self._step_counter += 1
        n_learnts = self.solver.nof_clauses(learnts=True)
        action = [0.0] * n_learnts

        if (self._step_counter == self.snapshot):
            assert (n_learnts == len(self.cl_label_arr_l)), "Mismatch in the number of clauses between snapshot step and delete step"

            action[self.tagged_id] = -1.0
        return action

    def update_labels(self):
        cc_np = self.solver.get_cl_labels(clause_type="learnt")
        current_clauses = pd.DataFrame(data=cc_np, index=cc_np[:,0])
        for old_cl in self.cl_label_arr_l: # Find the old learnt clauses in the new cl_labels array
            if (old_cl[c_map["id"]] in current_clauses.index): # If you find it, update its labels
                new_cl = current_clauses.loc[old_cl[c_map["id"]]]
                old_cl[c_map['del']] = new_cl[c_map['del']]
                old_cl[c_map['tagged']] = new_cl[c_map['tagged']]
                old_cl[c_map['lbl:total_used']] = new_cl[c_map['num_used']]

                if (old_cl[c_map["del"]] and old_cl[c_map["lbl:step_killed"]] == 0.0): # the tagged clause is marked for delete. Record the step
                    # print(f"""{new_cl[c_map["del"]]} and {old_cl[c_map["lbl:step_killed"]]}""")
                    old_cl[c_map["lbl:step_killed"]] = self._step_counter
            elif (old_cl[c_map["lbl:step_killed"]] == 0.0): # If you can't find it, record the step where it was deleted at
                old_cl[c_map["lbl:step_killed"]] = self._step_counter

    def pick_snapshot(self, step=None):
        if (not step): # pick a random snapshot between (1, no_gc)
            self.snapshot = np.random.choice(range(1, self.no_gc), 1)[0]
        else:
            self.snapshot = step if (step > 0) else (self.no_gc + step)
        if (self.snapshot <= 0 or self.snapshot >= self.no_gc):
            raise Exception("Bad step value. The resulting snapshot (%i) is not between 0 and total GC counts (%i)." % (self.snapshot, self.no_gc))
            return None

        print(f"Snapshot is going to be taken at: {self.snapshot}")

    def capture(self, step=None):
        self.pick_snapshot(step)

        self._step_counter = 0
        self.solver = Glucose3(gc_oracle = {"callback": self.snapshot_callback, "policy": "counter_factual"}, gc_freq="fixed", reduce_base=self.reduce_base)
        self.solver.append_formula(CNF(from_file=self.cnf).clauses)
        self.solver.solve()
        self.update_labels()
        self.op_cnt_with = self.solver.reward()

        self._step_counter = 0
        self.solver = Glucose3(gc_oracle = {"callback": self.delete_callback, "policy": "three_val"}, gc_freq="fixed", reduce_base=self.reduce_base)
        self.solver.append_formula(CNF(from_file=self.cnf).clauses)
        self.solver.solve()
        self.op_cnt_without = self.solver.reward()

        cl_labels = pd.DataFrame(data=self.cl_label_arr_l, index=self.cl_label_arr_l[:,0], columns=c_map)
        tagged  = cl_labels.iloc[self.tagged_id]

        return tagged['lbd'], self.snapshot, self.op_cnt_with, self.op_cnt_without

    def dump(self, dump_dir):
        cl_label_arr = np.r_[self.cl_label_arr_o, self.cl_label_arr_l] # appending the learnt clause labels to the end of the problem clause labels
        adjusted_tagged_id = len(self.cl_label_arr_o) + self.tagged_id
        # g_dgl = get_graph(self.adj_arrays, cl_label_arr, self.lit_label_arr)
        result = {
            "no_gc": self.no_gc,
            "snapshot": self.snapshot,
            "tagged_id": adjusted_tagged_id,
            "op_cnt_with": self.op_cnt_with,
            "op_cnt_without": self.op_cnt_without,
            "gss": self.gss,
            "adjacency": self.adj_arrays,
            "lit_feat": self.lit_label_arr,
            "clause_feat_and_labels": cl_label_arr,
            # "graph": g_dgl
        }
        with open(f"{dump_dir}/{os.path.basename(self.cnf)}-{random_string(8)}.pickle",'wb') as f:
            pickle.dump(result, f)

    @staticmethod
    def load(fname: FileName):
        with open(fname,'rb') as f:
            result = pickle.load(f)
            result["graph"] = get_graph(result["adjacency"], result["clause_feat_and_labels"], result["lit_feat"])

            return result

    @staticmethod
    def dump_stats(stats, dest: FileName):
        from matplotlib import pyplot as plt
        import math

        lbd       = [int(item[0]) for item in stats]
        snapshot  = [int(item[1]) for item in stats]
        reward_w  = np.array([abs(item[2]) for item in stats])
        reward_wo = np.array([abs(item[3]) for item in stats])
        reward    = np.abs(reward_w - reward_wo) / np.maximum(reward_w, reward_wo)

        fig = plt.figure()
        fig.suptitle(f"The effect of clause removal on OP_CNT ({len(lbd)} samples)")

        scatter = plt.scatter(lbd, reward, c=snapshot)
        xint = range(min(lbd), math.ceil(max(lbd))+1)
        plt.xticks(xint)
        plt.xlabel("LBD")
        plt.ylabel(r'$|\Delta OP\_CNT|/max(OP\_CNT)$')
        plt.legend(*scatter.legend_elements(num=max(snapshot) - min(snapshot)),
                        loc='upper center', ncol=7, bbox_to_anchor=(0.5, -0.11), title="GC step where the snapshot was taken")
        fig.savefig(os.path.join(dest, 'stats_abs.png'), bbox_inches='tight')



        fig = plt.figure()
        fig.suptitle(f"The effect of clause removal on OP_CNT ({len(lbd)} samples)")

        scatter = plt.scatter(lbd, reward_w - reward_wo, c=snapshot)
        xint = range(min(lbd), math.ceil(max(lbd))+1)
        plt.xticks(xint)
        plt.xlabel("LBD")
        plt.ylabel(r'$OP\_CNT_{w} - OP\_CNT_{wo}$')
        plt.legend(*scatter.legend_elements(num=max(snapshot) - min(snapshot)),
                        loc='upper center', ncol=7, bbox_to_anchor=(0.5, -0.11), title="GC step where the snapshot was taken")
        fig.savefig(os.path.join(dest, 'stats.png'), bbox_inches='tight')
