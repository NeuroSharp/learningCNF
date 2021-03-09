import numpy as np
from scipy.sparse import csr_matrix


"""
	This is a collection of useful callbacks, used in tests.

	The input array to all the callbacks has the following shape (see Oracle.h):
	Index | Num_Used | Size | LBD | Activity | Locked
	cl_label_arr: learnts x 6
"""

# This callback mimics the logic of LBD (but deletes more clauses)
def lbd_based_cb(cl_label_arr, rows_arr, cols_arr, data_arr):
	adj_matrix = csr_matrix((data_arr, (rows_arr, cols_arr)))
	# Trying to mimic the LBD based clause reduction (not exactly the same!)
	# if(c.size() > 2 && c.lbd() > 2 && !locked(c)) --> mark for remove!
	size_cond = np.greater(cl_label_arr[:,2], 2)
	lbd_cond = np.greater(cl_label_arr[:,3], 2)
	locked_cond = np.logical_not(cl_label_arr[:,5])
	
	return np.logical_not(np.logical_and(np.logical_and(size_cond, lbd_cond), locked_cond))

# Keeps all the learnt clauses
def keep_learnts_cb(cl_label_arr, rows_arr, cols_arr, data_arr):
	nLearnts = len(cl_label_arr)

	return np.full((nLearnts, 1), True)

# Delete all the learnt clauses (except for the locked ones)
def del_learnts_cb(cl_label_arr, rows_arr, cols_arr, data_arr):
	return (cl_label_arr[:, 5].reshape((-1 , 1)) == 1)

# This callback passes an empty np array to the reduceDB function which
# causes that function to fallback to the LBD-based reduction
def empty_cb(cl_label_arr, rows_arr, cols_arr, data_arr):
	return np.empty(shape=(0, 0), dtype=bool)
