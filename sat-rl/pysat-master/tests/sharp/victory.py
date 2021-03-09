import sys, os
import networkx as nx
from pyvis.network import Network
import matplotlib.pyplot as plt
import itertools
import numpy as np

from pysat.solvers import SharpSAT

np.set_printoptions(suppress=True, threshold=sys.maxsize)

def sharpSAT_cb(row, col, data, labels_df, lit_stack):
    act_space = labels_df["id_dimacs"].values

    rlookup = {v: k for k, v in enumerate(act_space)}
    pos_act_sp = np.abs(np.array(labels_df["id_dimacs"].values))
    pos_act_sp = np.max(pos_act_sp) - pos_act_sp
    act_prob  = pos_act_sp / np.sum(pos_act_sp)

    ind = np.random.choice(act_space, p=act_prob)

    # print(act_space[:30])
    # print(act_prob[:30])
    # print("===================")

    # ret = np.argmin(act_space)
    ret = rlookup[ind]

    # print(rlookup)
    # print(ret)
    return ret


ds_dir="/Users/Haddock/Desktop/sharp_cell9_test"
files = [os.path.join(ds_dir, f) for f in os.listdir(ds_dir)]
seq_len = []
for i, file in enumerate(files):
    if(not file.endswith(".cnf")): continue

    print(f"{i}. Solving: {file}...", end="")
    m = SharpSAT(branching_oracle = {"branching_cb": sharpSAT_cb})
    m.solve(file)
    seq_len += [m.reward()]
    print(f"steps: {m.reward()}")
    m.delete()

print(seq_len)
print(f"avg: {np.mean(seq_len)}")
