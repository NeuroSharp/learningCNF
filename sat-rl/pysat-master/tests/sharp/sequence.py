import sys
import pickle

from pyvis.network import Network
import matplotlib.pyplot as plt
import itertools
import numpy as np

from pysat.solvers import SharpSAT

index = -1

np.set_printoptions(suppress=True, threshold=sys.maxsize)

def sharpSAT_cb(row, col, data, labels_df, lit_stack):
    global index
    index += 1

    # if (index >= len(sequence)):
    #     action = -1
    # else:
    action = labels_df.index[labels_df['id_dimacs'] == sequence[index]].tolist()[0]

    print(index, action)

    return action

def default_cb(row, col, data, labels_df, lit_stack):
    return -1


cnf_id = "grid_10_7_a01OpVL2_1590726613"
cnf = f"/Users/Haddock/Desktop/alldata/sharp_grid_annt_huge/{cnf_id}.cnf"

# ds_dir = "TIMEDECODE_GRID_HUGE"
# seq_file = f"/Users/Haddock/Desktop/alldata/{ds_dir}/{ds_dir}_EVAL_TEST_episode_log_{cnf_id}.cnf.pickle"

ds_dir = "VANILLA_GRID_HUGE"
seq_file = f"/Users/Haddock/Desktop/alldata/{ds_dir}/{ds_dir}_EVAL_TEST_episode_log_{cnf_id}.cnf.pickle"

###################################

# cnf_id = "grid_10_12_TwRTSSbE_1590899060"
# cnf = f"/Users/Haddock/Desktop/alldata/sharp_grid_10_12_test/{cnf_id}.cnf"

# # ds_dir = "timedecode_grid_10_12"
# # seq_file = f"/Users/Haddock/Desktop/alldata/{ds_dir}/TIMEDECODE_HUGE12_EVAL_TEST_episode_log_{cnf_id}.cnf.pickle"

# ds_dir = "vanilla_grid_10_12"
# seq_file = f"/Users/Haddock/Desktop/alldata/{ds_dir}/VANILLA_10_12_EVAL_TEST_episode_log_{cnf_id}.cnf.pickle"


infile = open(seq_file,'rb')
new_dict = pickle.load(infile)
# print(new_dict)
infile.close()
sequence = new_dict["actions"]

m = SharpSAT(branching_oracle = {"branching_cb": sharpSAT_cb})
m.solve(cnf)
print("step count:", m.reward())
print("index:", index)
print("len_seq:", len(sequence))
m.delete()

if(index != len(sequence) - 1):
    print(f"Missmatch: {index} != {len(sequence) - 1}")

