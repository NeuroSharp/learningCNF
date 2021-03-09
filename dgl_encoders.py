import ipdb
import torch
import torch.nn as nn
from torch.nn import init as nn_init
import torch.nn.functional as F
import utils
import numpy as np
import rnn_util
import dgl
import dgl.function as fn
from collections import namedtuple
from IPython.core.debugger import Tracer
from settings import *
from rl_utils import *
from dgl_layers import *
from common_components import *

class CNFVarLayer(nn.Module):
  def __init__(self, in_size, clause_size, out_size, activation=None, settings=None, **kwargs):
    super(CNFVarLayer, self).__init__()
    self.ntypes = ['variable', 'clause']
    self.etypes = ['v2c_+', 'v2c_-', 'c2v_+', 'c2v_-']
    # W_r for each relation
    self.weight = nn.ModuleDict({
      self.etypes[0] : nn.Linear(in_size, clause_size),
      self.etypes[1] : nn.Linear(in_size, clause_size),
      self.etypes[2] : nn.Linear(clause_size+1, out_size),
      self.etypes[3] : nn.Linear(clause_size+1, out_size)
    })
    self.settings = settings if settings else CnfSettings()
    self.activation = activation if activation else eval(self.settings['non_linearity'])
    
  def forward(self, G, feat_dict):
    # the input is a dictionary of node features for each type
    
    ###### v2c propagation
    Wh_v2c_p = self.weight['v2c_+'](feat_dict['variable'])
    G.nodes['variable'].data['Wh_v2c_p'] = Wh_v2c_p
    Wh_v2c_n = self.weight['v2c_-'](feat_dict['variable'])
    G.nodes['variable'].data['Wh_v2c_n'] = Wh_v2c_n
    
    G['v2c_+'].update_all(fn.copy_src('Wh_v2c_p', 'm_v2c_p'), fn.sum('m_v2c_p', 'h_v2c_p'))
    G['v2c_-'].update_all(fn.copy_src('Wh_v2c_n', 'm_v2c_n'), fn.sum('m_v2c_n', 'h_v2c_n'))
    
    cembs = G.nodes['clause'].data['h_v2c_p'] + G.nodes['clause'].data['h_v2c_n']

    # normalize, for each clause node
    combined_degrees = G.in_degrees(etype="v2c_+") + G.in_degrees(etype="v2c_-")
    combined_degrees[combined_degrees == 0] = 1
    combined_degrees = combined_degrees.float()
    cembs = ((cembs.T)/combined_degrees).T
    cembs = self.activation(cembs)
    G.nodes['clause'].data['h_v2c'] = cembs
    cembs = torch.cat([cembs,feat_dict['clause']], dim=1)
    
    ###### c2v propagation  
    Wh_c2v_p = self.weight['c2v_+'](cembs)
    G.nodes['clause'].data['Wh_c2v_p'] = Wh_c2v_p
    Wh_c2v_n = self.weight['c2v_-'](cembs)
    G.nodes['clause'].data['Wh_c2v_n'] = Wh_c2v_n
    
    G['c2v_+'].update_all(fn.copy_src('Wh_c2v_p', 'm_c2v_p'), fn.sum('m_c2v_p', 'h_c2v_p'))
    G['c2v_-'].update_all(fn.copy_src('Wh_c2v_n', 'm_c2v_n'), fn.sum('m_c2v_n', 'h_c2v_n'))
    
    cembs = G.nodes['variable'].data['h_c2v_p'] + G.nodes['variable'].data['h_c2v_n']

    # normalize, for each clause node
    combined_degrees = G.in_degrees(etype="c2v_+") + G.in_degrees(etype="c2v_-")
    combined_degrees[combined_degrees == 0] = 1
    combined_degrees = combined_degrees.float()
    vembs = ((cembs.T)/combined_degrees).T
    vembs = self.activation(vembs)
    G.nodes['variable'].data['h_c2v'] = vembs
                        
    return vembs
    
class DGLEncoder(nn.Module):
  def __init__(self, settings=None, **kwargs):
    super(DGLEncoder, self).__init__()
    self.settings = settings if settings else CnfSettings()
    
    self.vlabel_dim = self.settings['vlabel_dim']
    self.clabel_dim = self.settings['clabel_dim']
    self.vemb_dim = self.settings['vemb_dim']
    self.cemb_dim = self.settings['cemb_dim']
    self.max_iters = self.settings['max_iters']        
    self.non_linearity = eval(self.settings['non_linearity'])

  def tie_literals(embs, interleaved=True, concat_origin=True):
    feat_size = embs.shape[1]
    if interleaved:
      z = embs.view(-1,2,feat_size)
      rc = z[:,(1,0),:].view(-1,feat_size)
    else:
      n = int(embs.shape[0]/2)
      rc = torch.cat([embs[n:],embs[:n]],dim=0)
    if concat_origin:
      rc = torch.cat([embs,rc],dim=1)
    return rc

  def forward(self, G, feat_dict, **kwargs):
    raise NotImplementedError

  def output_size(self):
    raise NotImplementedError


class CNFEncoder(DGLEncoder):
  def __init__(self, settings=None, **kwargs):
    super(CNFEncoder, self).__init__(settings=settings, **kwargs)
    norm_class = None
    if self.settings['cp_normalization'] == 'batch':
      norm_class = nn.BatchNorm1d
    elif self.settings['cp_normalization'] == 'layer':
      norm_class = nn.LayerNorm
    self.use_norm = self.settings['cp_normalization'] != None
    cnf_layer = CNFLayer
    if self.max_iters > 0:
      self.vnorm_layers = nn.ModuleList([norm_class(self.vemb_dim)]) if self.use_norm else nn.ModuleList([])
      self.layers = nn.ModuleList([cnf_layer(self.vlabel_dim, self.cemb_dim, self.vemb_dim, activation=self.non_linearity, **kwargs)])
      for i in range(1,self.max_iters):
        self.layers.append(cnf_layer(2*self.vemb_dim, self.cemb_dim, self.vemb_dim, activation=self.non_linearity, norm_class=norm_class, **kwargs))
        if self.use_norm:
          self.vnorm_layers.append(norm_class(self.vemb_dim))

  def forward(self, G, feat_dict, **kwargs):    
    vlabels = feat_dict['literal']
    if self.max_iters == 0:
      z = torch.zeros(1)
      vembs = z.expand(G.number_of_nodes('literal'),2*self.vemb_dim)
      cembs = z.expand(G.number_of_nodes('clause'),self.cemb_dim)      
      return vembs, cembs
    for i in range(self.max_iters):
      pre_embs = self.layers[i](G, feat_dict)
      if self.use_norm:
          pre_embs = self.vnorm_layers[i](pre_embs)
      embs = DGLEncoder.tie_literals(pre_embs)
      feat_dict['literal'] = embs
    vembs = embs
    cembs = G.nodes['clause'].data['cembs']
    return vembs, cembs

  def output_size(self):
    return self.cemb_dim  

class CNFVarEncoder(DGLEncoder):
  def __init__(self, settings=None, **kwargs):
    super(CNFVarEncoder, self).__init__(settings=settings, **kwargs)

    self.layers = nn.ModuleList([CNFVarLayer(self.vlabel_dim, self.cemb_dim, self.vemb_dim, activation=self.non_linearity, **kwargs)])
    for i in range(1,self.max_iters):
      self.layers.append(CNFVarLayer(self.vemb_dim, self.cemb_dim, self.vemb_dim, activation=self.non_linearity, **kwargs))

  def forward(self, G, feat_dict, **kwargs):    
    embs = self.layers[0](G,feat_dict)
    for i in range(1,self.max_iters):      
      embs = self.layers[i](G, feat_dict)
    return embs


class GINEncoder(DGLEncoder):
  def __init__(self, settings=None, **kwargs):
    super(GINEncoder, self).__init__(settings=settings, **kwargs)
    self.settings = settings if settings else CnfSettings()    
    self.aggregate = fn.sum
    self.d = self.settings['sharp_emb_dim']
    self.num_layers = self.settings['cp_num_layers']
    self.epsilon = nn.Parameter(torch.Tensor([0.2]))
    
    if self.vlabel_dim:
      self.literal_features_layer = MLPModel([self.vlabel_dim,self.d,self.d])
    else:
      self.L_init = nn.Parameter(torch.normal(torch.zeros(self.d)))
    if self.clabel_dim:
      self.clause_features_layer = MLPModel([self.clabel_dim,self.d,self.d])
    else:
      self.C_init = nn.Parameter(torch.normal(torch.zeros(self.d)))

    self.L_update = nn.ModuleList([])
    self.C_update = nn.ModuleList([])
    for i in range(self.max_iters):  
      self.L_update.append(MLPModel([2*self.d, self.d, self.d], layernorm=True))
      self.C_update.append(MLPModel([self.d, self.d, self.d], layernorm=True))

# We use torch.Tensor just in case, its idempotent.

  def forward(self, G, **kwargs):
    if self.vlabel_dim:
      literals = self.literal_features_layer(torch.Tensor(G.nodes['literal'].data['literal_feats']) / np.sqrt(self.d))
    else:
      literals = self.L_init.expand(G.number_of_nodes('literal'), self.d) / np.sqrt(self.d)
    if self.clabel_dim:
      clauses = self.clause_features_layer(torch.Tensor(G.nodes['clause'].data['clause_feats']) / np.sqrt(self.d))
    else:
      clauses = self.C_init.expand(G.number_of_nodes('clause'), self.d) / np.sqrt(self.d)
    
    G.nodes['literal'].data['literal_embs'] = literals
    G.nodes['clause'].data['clause_embs'] = clauses

    # ipdb.set_trace()
    for i in range(self.max_iters):        
      G['l2c'].update_all(fn.copy_src('literal_embs', 'm'), self.aggregate('m', 'h'))
      pre_cembs = (self.epsilon+1)*G.nodes['clause'].data['clause_embs'] + G.nodes['clause'].data['h']      
      G.nodes['clause'].data['clause_embs'] = self.C_update[i](pre_cembs)
      G['c2l'].update_all(fn.copy_src('clause_embs', 'm'), self.aggregate('m', 'h'))
      pre_lembs = (self.epsilon+1)*G.nodes['literal'].data['literal_embs'] + G.nodes['literal'].data['h']
      pre_lembs = DGLEncoder.tie_literals(pre_lembs)
      G.nodes['literal'].data['literal_embs'] = self.L_update[i](pre_lembs)

    return G.nodes['literal'].data['literal_embs'], G.nodes['clause'].data['clause_embs']

  def output_size(self):
    return self.d    



# Neurosat-style encoder

class NSATEncoder(DGLEncoder):
  def __init__(self, settings=None, **kwargs):
    super(NSATEncoder, self).__init__(settings=settings, **kwargs)
    self.settings = settings if settings else CnfSettings()
    self.d = self.settings['cp_emb_dim']
    self.relaxed_msg = self.settings['cp_nsat_relaxed_msg']
    self.relaxed_update = self.settings['cp_nsat_relaxed_update']
    self.use_labels = self.settings['cp_nsat_use_labels']
    self.aggregate = fn.sum if self.settings['use_sum'] else fn.mean
    self.num_layers = self.settings['cp_num_layers']
    if self.relaxed_msg:      
      self.LC_msg = nn.ModuleList([])
      self.CL_msg = nn.ModuleList([])
      for i in range(self.max_iters):
        self.LC_msg.append(MLPModel([self.d]*self.num_layers))
        self.CL_msg.append(MLPModel([self.d]*self.num_layers))
    else:
      self.LC_msg = MLPModel([self.d]*self.num_layers)
      self.CL_msg = MLPModel([self.d]*self.num_layers)
    self.my_zero = nn.Parameter(torch.zeros(1),requires_grad=False)

    if self.use_labels:
      self.literal_features_layer = MLPModel([self.vlabel_dim,self.d,self.d])
      self.clause_features_layer = MLPModel([self.clabel_dim,self.d,self.d])
    else:
      self.L_init = nn.Parameter(torch.normal(torch.zeros(self.d)))
      self.C_init = nn.Parameter(torch.normal(torch.zeros(self.d)))

    if self.relaxed_update:
      self.L_update = nn.ModuleList([])
      self.C_update = nn.ModuleList([])
      for i in range(self.max_iters):  
        self.L_update.append(rnn_util.LayerNormLSTMCell(2*self.d, self.d))
        self.C_update.append(rnn_util.LayerNormLSTMCell(self.d, self.d))
    else:
      self.L_update = rnn_util.LayerNormLSTMCell(2*self.d, self.d)
      self.C_update = rnn_util.LayerNormLSTMCell(self.d, self.d)


    # like tie_literals before the cat
  def flip(self,L):
    z = L.view(-1,2,self.d)
    return z[:,(1,0),:].view(-1,self.d)
    
  def forward(self, G, feat_dict, **kwargs):
    if self.use_labels:
      literals = self.literal_features_layer(feat_dict['literal']) / np.sqrt(self.d)
      clauses = self.clause_features_layer(feat_dict['clause']) / np.sqrt(self.d)
    else:      
      literals = self.L_init.expand(G.number_of_nodes('literal'), self.d) / np.sqrt(self.d)
      clauses = self.C_init.expand(G.number_of_nodes('clause'), self.d) / np.sqrt(self.d)
      
    L_state = (literals, self.my_zero.expand(literals.shape[0], self.d))
    C_state = (clauses, self.my_zero.expand(clauses.shape[0], self.d))

    for i in range(self.max_iters):
      if self.relaxed_msg:
        LC_msg = self.LC_msg[i]
        CL_msg = self.CL_msg[i]
      else:
        LC_msg = self.LC_msg
        CL_msg = self.CL_msg
      if self.relaxed_update:
        L_update = self.L_update[i]
        C_update = self.C_update[i]
      else:
        L_update = self.L_update
        C_update = self.C_update
      G.nodes['literal'].data['l2c_msg'] = LC_msg(L_state[0])
      G['l2c'].update_all(fn.copy_src('l2c_msg', 'm'), self.aggregate('m', 'h'))
      pre_cembs = G.nodes['clause'].data['h']
      C_state = C_update(pre_cembs, C_state)
      G.nodes['clause'].data['c2l_msg'] = CL_msg(C_state[0])
      G['c2l'].update_all(fn.copy_src('c2l_msg', 'm'), self.aggregate('m', 'h'))
      pre_lembs = G.nodes['literal'].data['h']
      L_state = L_update(torch.cat([pre_lembs,self.flip(L_state[0])],axis=1), L_state)

    return L_state[0], C_state[0]

  def output_size(self):
    return self.d    

class CnfAagEncoder(DGLEncoder):
  def __init__(self, settings=None, **kwargs):
    super(CnfAagEncoder, self).__init__(settings=settings, **kwargs)
    
    self.cnf_layers = nn.ModuleList([CNFLayer(self.vlabel_dim, self.cemb_dim, self.vemb_dim, activation=self.non_linearity, **kwargs)])
    self.aag_layers = nn.ModuleList([AAGLayer(2*self.vemb_dim, self.vemb_dim, activation=self.non_linearity, **kwargs)])
    for i in range(1,self.max_iters):
      self.cnf_layers.append(CNFLayer(2*self.vemb_dim, self.cemb_dim, self.vemb_dim, activation=self.non_linearity, **kwargs))
      self.aag_layers.append(AAGLayer(2*self.vemb_dim, self.vemb_dim, activation=self.non_linearity, **kwargs))

  def forward(self, G, feat_dict, **kwargs):
    pre_embs_cnf = self.cnf_layers[0](G,feat_dict)
    embs_cnf = DGLEncoder.tie_literals(pre_embs_cnf)
    feat_dict['literal'] = embs_cnf
    pre_embs = self.aag_layers[0](G, feat_dict)
    embs = DGLEncoder.tie_literals(pre_embs)
    for i in range(1,self.max_iters):      
      feat_dict['literal'] = embs
      pre_embs_cnf = self.cnf_layers[i](G,feat_dict)
      embs_cnf = DGLEncoder.tie_literals(pre_embs_cnf)
      feat_dict['literal'] = embs_cnf
      pre_embs = self.aag_layers[i](G, feat_dict)
      embs = DGLEncoder.tie_literals(pre_embs)
    return embs

###############################################################################
    
# class DGLPolicy(PolicyBase):
#   def __init__(self, encoder=None, **kwargs):
#     super(DGLPolicy, self).__init__(**kwargs)
#     self.final_embedding_dim = 2*self.vemb_dim+self.vlabel_dim
#     self.hidden_dim = 50
#     if encoder:
#       print('Bootstraping Policy from existing encoder')
#       self.encoder = encoder
#     else:      
#       self.encoder = CNFEncoder(**kwargs)
#     if self.settings['use_global_state']:
#       self.linear1 = nn.Linear(self.state_dim+self.final_embedding_dim, self.policy_dim1)
#     else:
#       self.linear1 = nn.Linear(self.final_embedding_dim, self.policy_dim1)

#     if self.policy_dim2:
#       self.linear2 = nn.Linear(self.policy_dim1,self.policy_dim2)
#       self.action_score = nn.Linear(self.policy_dim2,1)
#     else:
#       self.action_score = nn.Linear(self.policy_dim1,1)
#     if self.state_bn:
#       self.state_vbn = MovingAverageVBN((self.snorm_window,self.state_dim))
#     self.use_global_state = self.settings['use_global_state']
#     self.activation = eval(self.settings['policy_non_linearity'])

  
#   # state is just a (batched) vector of fixed size state_dim which should be expanded. 
#   # ground_embeddings are batch * num_lits * ground_embedding

#   # cmat_net and cmat_pos are already "batched" into a single matrix

#   def forward(self, obs, G, **kwargs):
#     state = obs.state 
#     self.batch_size=obs.state.shape[0]

#     feat_dict = {'literal': G.nodes['literal'].data['lit_labels'], 'clause': torch.zeros(G.number_of_nodes('clause'),1)}
#     # feat_dict = {'literal': G.nodes['literal'].data['lit_labels'], 'clause': G.nodes['clause'].data['clause_labels']}
#     aux_losses = []    
    
#     num_lits = int(feat_dict['literal'].shape[0] / self.batch_size)
    
#     if 'vs' in kwargs.keys():
#       vs = kwargs['vs']   
#     else:
#       vs = self.encoder(G, feat_dict, **kwargs)
#       if 'do_debug' in kwargs:
#         Tracer()()

#     if self.use_global_state:
#       if self.state_bn:
#         state = self.state_vbn(state)
#       reshaped_state = state.unsqueeze(1).expand(state.shape[0],num_lits,self.state_dim).contiguous().view(-1,self.state_dim)
#       inputs = torch.cat([reshaped_state, vs],dim=1).view(-1,self.state_dim+self.final_embedding_dim)
#     else:
#       inputs = vs.view(-1,self.final_embedding_dim)

#     if self.policy_dim2:      
#       outputs = self.action_score(self.activation(self.linear2(self.activation(self.linear1(inputs)))))
#     else:
#       outputs = self.action_score(self.activation(self.linear1(inputs)))
#     outputs = outputs.view(self.batch_size,-1)

#     return outputs, vs

#   def get_allowed_actions(self, obs, **kwargs):
#     rc = cadet_utils.get_allowed_actions(obs,**kwargs)
#     s = rc.shape
#     rc = rc.unsqueeze(2).expand(*s,2).contiguous()
#     rc = rc.view(s[0],-1)
#     return rc

#   def translate_action(self, action, obs, **kwargs):
#     try:
#       if action in ['?']:
#         return action
#     except:
#       pass
#     return (int(action/2),int(action%2))

#   def combine_actions(self, actions, **kwargs):
#     return self.settings.LongTensor(actions)

#   def select_action(self, obs, **kwargs):
#     [logits], *_ = self.forward(obs, obs.ext_data.G.local_var())
#     allowed_actions = self.get_allowed_actions(obs)[0]
#     allowed_idx = self.settings.cudaize_var(torch.from_numpy(np.where(allowed_actions.numpy())[0]))
#     l = logits[allowed_idx]
#     probs = F.softmax(l.contiguous().view(1,-1),dim=1)
#     dist = probs.data.cpu().numpy()[0]
#     choices = range(len(dist))
#     aux_action = np.random.choice(choices, p=dist)
#     action = allowed_idx[aux_action]
#     return action, 0

#   def compute_loss(self, transition_data, **kwargs):
#     _, _, _, rewards, *_ = zip(*transition_data)    
#     collated_batch = collate_transitions(transition_data,settings=self.settings)
#     collated_batch.state = cudaize_obs(collated_batch.state)
#     G = batched_combined_graph([x.G for x in collated_batch.state.ext_data])
#     logits, *_ = self.forward(collated_batch.state, G, prev_obs=collated_batch.prev_obs)
#     allowed_actions = self.get_allowed_actions(collated_batch.state)
#     if self.settings['cuda']:
#       allowed_actions = allowed_actions.cuda()    
#     effective_bs = len(logits)    

#     if self.settings['masked_softmax']:
#       allowed_mask = allowed_actions.float()      
#       probs, debug_probs = masked_softmax2d(logits,allowed_mask)
#     else:
#       probs = F.softmax(logits, dim=1)
#     all_logprobs = safe_logprobs(probs)
#     if self.settings['disallowed_aux']:        # Disallowed actions are possible, so we add auxilliary loss
#       aux_probs = F.softmax(logits,dim=1)
#       disallowed_actions = Variable(allowed_actions.data^1).float()      
#       disallowed_mass = (aux_probs*disallowed_actions).sum(1)
#       disallowed_loss = disallowed_mass.mean()
#       # print('Disallowed loss is {}'.format(disallowed_loss))

#     returns = self.settings.FloatTensor(rewards)
#     actions = collated_batch.action    
#     try:
#       logprobs = all_logprobs.gather(1,actions.view(-1,1)).squeeze()
#     except:
#       Tracer()()
#     entropies = (-probs*all_logprobs).sum(1)    
#     adv_t = (returns - returns.mean()) / (returns.std() + float(np.finfo(np.float32).eps))
#     if self.settings['use_sum']:
#       pg_loss = (-adv_t*logprobs).sum()
#     else:
#       pg_loss = (-adv_t*logprobs).mean()
    
#     loss = pg_loss

#     if self.use_global_state and self.state_bn:
#       self.state_vbn.recompute_moments(collated_batch.state.state.detach())

#     return loss, logits
    
###############################################################################
"""Tests"""
###############################################################################
#a = CombinedGraph1Base()
#a.load_paired_files(aag_fname = './data/words_test_ryan_mini_1/w.qaiger', qcnf_fname = './data/words_test_ryan_mini_1/w.qaiger.qdimacs')
#feat_dict = {
#        'literal': a.G.nodes['literal'].data['lit_labels'],   
#        'clause' : a.G.nodes['clause'].data['clause_labels'] 
#}
#in_size = feat_dict['literal'].shape[1]
#clause_size = 11
#out_size = 7
#C = CNFLayer(in_size, clause_size, out_size, activation=F.relu)
#C_f = C(a.G, feat_dict)  
#in_size = C_f.shape[1]
#out_size = 5
#A = AAGLayer(in_size, out_size, activation=F.relu)
#feat_dict = {'literal' : C_f}
#A_f = A(a.G, feat_dict)
#    ##########################################################################
#    ## TO GO INSIDE AAGLayer() Forward()
#    """
#    EXAMPLE:   files found in words_test_ryan_mini_1
#    10 literals            0,1,2,...,9
#    6 aag_forward_edges   (4,0) (4,2) (6,1) (6,4) (8,2) (8,7)
#    6 aag_backward_edges  (0,4) (2,4) (1,6) (4,6) (2,8) (7,8)
#    The following is what the literal embeddings should be (before the nonlinearity),
#        given that we have already passed the cnf_output through a linear layer
#        to produce Wh_af, Wh_ab:
#    """
#    embs = torch.zeros(10, 5) 
#    embs[0] = Wh_af[4]
#    embs[1] = Wh_af[6]
#    embs[2] = (Wh_af[4] + Wh_af[8])/2
#    embs[4] = (Wh_af[6] + Wh_ab[0] + Wh_ab[2])/3
#    embs[6] = (Wh_ab[1] + Wh_ab[4])/2
#    embs[7] = Wh_af[8]
#    embs[8] = (Wh_ab[2] + Wh_ab[7])/2
#    
##    import ipdb
##    ipdb.set_trace()
##    print(lembs)
##    print(should_be)
#    ##########################################################################
###############################################################################
### TEST tie_literals()
#e = torch.zeros(10,4)
#for i in range(10):
#  e[i] = i
#e
#DGLEncoder.tie_literals(e)
###############################################################################
### Test the Encoders:
#a = CombinedGraph1Base()
#a.load_paired_files(aag_fname = './data/words_test_ryan_mini_1/w.qaiger', qcnf_fname = './data/words_test_ryan_mini_1/w.qaiger.qdimacs')
#### put the following inside DGLEncoder __init__()
#self.vlabel_dim = 4
#self.clabel_dim = 1
#self.vemb_dim = 3
#self.cemb_dim = 11 #hidden
#self.max_iters = 2   
#self.non_linearity = F.relu
####
#G = a.G
#feat_dict = {'literal' : torch.ones(10,4), 'clause': torch.ones(10,1)}
#e = CnfAagEncoder()
#v = e(G, feat_dict)
#import ipdb
#ipdb.set_trace()
###############################################################################
#### MAILBOX testing
#def reduce_af(nodes):
#    import ipdb
#    ipdb.set_trace()
#    return { 'haf' : torch.sum(nodes.mailbox['m_af'], dim=1) }
#def reduce_ab(nodes):
#    import ipdb
#    ipdb.set_trace()
#    return { 'hab' : torch.sum(nodes.mailbox['m_ab'], dim=1) }
#
#    G['aag_forward'].update_all(fn.copy_src('Wh_af', 'm_af'), reduce_af)
#    G['aag_backward'].update_all(fn.copy_src('Wh_ab', 'm_ab'), reduce_ab)
#    import ipdb
#    ipdb.set_trace()

###############################################################################
#### Test CNFVarLayer()
#a = DGL_Graph_Base()
#a.load_paired_files(
#        aag_fname = './data/words_test_ryan_0/words_0_SAT.qaiger', 
#        qcnf_fname = './data/words_test_ryan_0/words_0_SAT.qaiger.qdimacs',
#        var_graph =  True)
#feat_dict = {
#        'variable': a.G.nodes['variable'].data['var_labels'],   
#        'clause' : a.G.nodes['clause'].data['clause_labels'] 
#}
#in_size = feat_dict['variable'].shape[1]
#clause_size = 11
#out_size = 7
#C = CNFVarLayer(in_size, clause_size, out_size, activation=F.relu)
#C_f = C(a.G, feat_dict)  
##### Test CNFVarEncoder()
### put the following inside DGLEncoder __init__()
#self.vlabel_dim = 9
#self.clabel_dim = 1
#self.vemb_dim = 9
#self.cemb_dim = 11 #hidden
#self.max_iters = 2   
#self.non_linearity = F.relu
####
#G = a.G
###feat_dict = {'variable' : torch.ones(10,4), 'clause': torch.ones(10,1)}
#e = CNFVarEncoder()
#v = e(G, feat_dict)
###############################################################################
