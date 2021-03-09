import random
import os
import torch
from IPython.core.debugger import Tracer
from torch.autograd import Variable
from collections import namedtuple

from settings import *
from qbf_data import *
from rl_types import *
from utils import *


#   # We return a ground embedding of (self.num_vars,7), where embedding is 0 - universal, 1 - existential, 2 - pad, 
#   # 3 - determinized, 4 - activity, [5,6] - pos/neg determinization
#   # 3 is obviously empty here

# def get_base_ground(qbf, settings=None):
#   if not settings:
#     settings = CnfSettings()
#   rc = np.zeros([qbf.num_vars,settings['ground_dim']]).astype(float)
#   for j, val in enumerate(qbf.var_types):
#     rc[j][val] = True
#   return rc

def get_input_from_qbf(qbf, settings=None, split=True):
  if not settings:
    settings = CnfSettings()
  a = qbf.as_np_dict()  
  rc_i = a['sp_indices']
  rc_v = a['sp_vals']
  return create_sparse_adjacency(rc_i,rc_v,torch.Size([qbf.num_clauses,qbf.num_vars]),split)
  # sp_ind_pos = torch.from_numpy(rc_i[np.where(rc_v>0)])
  # sp_ind_neg = torch.from_numpy(rc_i[np.where(rc_v<0)])
  # sp_val_pos = torch.ones(len(sp_ind_pos))
  # sp_val_neg = torch.ones(len(sp_ind_neg))
  # cmat_pos = Variable(torch.sparse.FloatTensor(sp_ind_pos.t(),sp_val_pos,torch.Size([qbf.num_clauses,qbf.num_vars])))
  # cmat_neg = Variable(torch.sparse.FloatTensor(sp_ind_neg.t(),sp_val_neg,torch.Size([qbf.num_clauses,qbf.num_vars])))  
  # # if settings['cuda']:
  # #   cmat_pos, cmat_neg = cmat_pos.cuda(), cmat_neg.cuda()
  # return cmat_pos, cmat_neg

def new_episode(env, fname, settings=None, **kwargs):
  if not settings:
    settings = CnfSettings()
  env_gen = EnvIdGen()
  try:
    env_id = int(os.path.split(fname)[1].split('_')[-2])
  except:
    env_id = os.path.split(fname)[1]
    # env_id = env_gen.get_id()
  # Set up ground_embeddings and adjacency matrices
  state, vars_add, vars_remove, activities, _, _ , _, vars_set, _ = env.reset(fname)
  if state is None:   # Env solved in 0 steps
    return None, env_id, fname
  assert(len(state)==settings['state_dim'])
  ground_embs = env.qbf.get_base_embeddings()
  ground_embs[:,IDX_VAR_DETERMINIZED][vars_add] = True
  ground_embs[:,IDX_VAR_ACTIVITY] = activities
  if len(vars_set):
    a = vars_set
    idx = a[:,0][np.where(a[:,1]==1)[0]]
    ground_embs[:,IDX_VAR_SET_POS][idx] = True
    idx = a[:,0][np.where(a[:,1]==-1)[0]]
    ground_embs[:,IDX_VAR_SET_NEG][idx] = True
    idx = a[:,0][np.where(a[:,1]==0)[0]]
    ground_embs[:,IDX_VAR_SET_POS:IDX_VAR_SET_NEG][idx] = False  
  cmat_pos, cmat_neg = get_input_from_qbf(env.qbf, settings)
  
  state = Variable(torch.from_numpy(state).float().unsqueeze(0))
  ground_embs = Variable(torch.from_numpy(ground_embs).float().unsqueeze(0))
  clabels = Variable(torch.from_numpy(env.qbf.get_clabels()).float().unsqueeze(0))
  # if settings['cuda']:
  #   cmat_pos, cmat_neg = cmat_pos.cuda(), cmat_neg.cuda()
  #   state, ground_embs = state.cuda(), ground_embs.cuda()
  rc = State(state,cmat_pos, cmat_neg, ground_embs, clabels, None, None, None)
  return rc, env_id

def get_ground_index(obs, idx, packed=False):
  if packed:
    return obs.ground.long().data[:,idx].byte()
  else:    
    return obs.ground.long().data[:,:,idx].byte()

def get_determinized(obs, **kwargs):
  return get_ground_index(obs,IDX_VAR_DETERMINIZED, **kwargs)  

# A dirty hack implementing !determinized && existential

def get_allowed_actions(obs, **kwargs):
  return (get_determinized(obs, **kwargs)^1 + get_ground_index(obs,IDX_VAR_EXISTENTIAL, **kwargs)) == 2

def action_allowed(obs, action):
  if action == '?':
    return True
  if type(action) is tuple:       # double-sided actions
    action = action[0]
  return get_allowed_actions(obs)[0,action]

