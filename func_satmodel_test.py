import ipdb
from sat_code.capture import *
from cp_utils import *


model_path = '/home/gil/ray_results/counterfact/counterfact_iter2_1/ClausePredictionTrainable_2c735952_2020-03-29_14-06-35r33nf4g2/checkpoint_35/model.pth'
settings_path = './experiments_config/config_counterfact_iter2_1.json'

magic_function = wrap_sat_model(model_path, settings_path)

def callback():
  # global magic_function
  adj_arrays     = solver.get_cl_arr()
  cl_label_arr   = torch.Tensor(solver.get_cl_labels())
  lit_label_arr  = torch.Tensor(solver.get_lit_labels())
  gss            = torch.Tensor(solver.get_solver_state())

  probs = magic_function(adj_arrays, cl_label_arr, lit_label_arr, gss)

  return -1


solver = Glucose3(gc_oracle = {"callback": callback, "policy": "glucose"}, gc_freq="fixed")
f1 = CNF(from_file='data/glucose_test/ecarev-37-200-11-20-20.cnf')
solver.append_formula(f1.clauses)
solver.solve()
print('Thats it')