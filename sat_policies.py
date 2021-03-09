import torch
import shelve
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import init as nn_init
import torch.nn.functional as F
import torch.optim as optim
import utils
import numpy as np
from collections import namedtuple
from IPython.core.debugger import Tracer
from settings import *
from vbn import *
from sat_env import *
from sat_encoders import *
from policy_base import *
from rl_utils import *
from tick_utils import *
from torch.distributions import Categorical
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal

class SatPolicy(PolicyBase):
  def __init__(self, encoder=None, **kwargs):
    super(SatPolicy, self).__init__(**kwargs)
    self.final_embedding_dim = self.cemb_dim+self.clabel_dim        
    if encoder:
      print('Bootstraping Policy from existing encoder')
      self.encoder = encoder
    else:
      self.encoder = SatEncoder(**kwargs)
      # self.encoder = SatSimpleEncoder(**kwargs)
    if self.settings['use_global_state']:
      self.linear1 = nn.Linear(self.state_dim+self.final_embedding_dim, self.policy_dim1)
    else:
      self.linear1 = nn.Linear(self.final_embedding_dim, self.policy_dim1)

    if self.policy_dim2:
      self.linear2 = nn.Linear(self.policy_dim1,self.policy_dim2)
      self.action_score = nn.Linear(self.policy_dim2,2)
    else:
      self.action_score = nn.Linear(self.policy_dim1,2)
    if self.state_bn:
      self.state_bn = nn.BatchNorm1d(self.state_dim)

    self.gpu_cap = self.settings['gpu_cap']
      
  # state is just a (batched) vector of fixed size state_dim which should be expanded. 
  # vlabels are batch * max_vars * vlabel_dim

  # cmat is already "batched" into a single matrix

  def forward(self, obs, **kwargs):
    mt0 = 0
    mt1 = 0
    mt2 = 0
    mt3 = 0
    mt4 = 0
    state = obs.state
    vlabels = obs.ground
    clabels = obs.clabels
    size = clabels.size()
    self.batch_size=size[0]
    if size[0] > 1:
      mt0 = time.time()
    cmat_pos, cmat_neg = split_sparse_adjacency(obs.cmat)
    aux_losses = []
    if size[0] > 1:
      mt1 = time.time()

    # In MP the processes take care of cudaizing, because each Worker thread can have its own local model on the CPU, and
    # Only the main process does the training and has a global model on GPU.
    # In SP reinforce, we only have one model, so it has to be in GPU (if we're using CUDA)

    if self.settings['cuda'] and not self.settings['mp']:
      cmat_pos, cmat_neg = cmat_pos.cuda(), cmat_neg.cuda()
      state, vlabels, clabels = state.cuda(), vlabels.cuda(), clabels.cuda()

    num_learned = obs.ext_data
    cembs = self.encoder(vlabels.view(-1,self.vlabel_dim), clabels.view(-1,self.clabel_dim), cmat_pos=cmat_pos, cmat_neg=cmat_neg, **kwargs)
    if size[0] > 1:
      mt2 = time.time()
    cembs = cembs.view(self.batch_size,-1,self.final_embedding_dim)
    cembs_processed = []

    # WARNING - This is being done in a loop. gotta' change that.

    for i, (nl1, nl2) in enumerate(num_learned):
      cembs_processed.append(cembs[i,nl1:nl2,:])
    if 'do_debug' in kwargs:
      Tracer()()
    
    
    inputs = []
    if self.settings['use_global_state']:
      if self.state_bn:
        state = self.state_bn(state)
      for i, (s,emb) in enumerate(zip(state,cembs_processed)):
        a = s.view(1,self.state_dim)
        reshaped_state = a.expand(len(emb),self.state_dim)
        inputs.append(torch.cat([reshaped_state,emb],dim=1))
      inputs = torch.cat(inputs,dim=0)
    else:
      inputs = torch.cat(cembs_processed,dim=0)

    # if self.batch_size > 1:
    #   Tracer()()  
    if self.policy_dim2:      
      outputs = self.action_score(self.activation(self.linear2(self.activation(self.linear1(inputs)))))
    else:
      outputs = self.action_score(self.activation(self.linear1(inputs)))
    
    if size[0] > 1:
      mt3 = time.time()
    outputs_processed = []
    for i, (nl1, nl2) in enumerate(num_learned):
      s = nl2-nl1
      outputs_processed.append(outputs[:s])
      outputs = outputs[s:]
    assert(outputs.shape[0]==0)
    if any((x!=x).any() for x in outputs_processed):    # Check nans
      Tracer()()
    if size[0] > 1:
      mt4 = time.time()
    value = None

    if size[0] > 1:
      print('Times are: split: {}, encoder: {}, policy: {}, post_process: {}'.format(mt1-mt0,mt2-mt1,mt3-mt2,mt4-mt3))
    return outputs_processed, value, cembs, aux_losses

  def get_allowed_actions(self, obs, **kwargs):
    pass

  def translate_action(self, action, obs, **kwargs):
    # print('Action is: {}'.format(action[:10]))
    return action

  def combine_actions(self, actions, **kwargs):    
    return actions
    # return torch.cat(actions)

  def select_action(self, obs_batch, **kwargs):
    logits, *_ = self.forward(obs_batch)
    assert(len(logits)==1)
    logits = logits[0]
    probs = F.softmax(logits,dim=1)
    ps = probs[:,0].cpu().detach().numpy()
    action = torch.from_numpy(np.random.binomial(1,p=ps)).unsqueeze(0)
    num_learned = obs_batch.ext_data[0]
    locked = obs_batch.clabels[0,num_learned[0]:num_learned[1],CLABEL_LOCKED].long().view(1,-1)
    final_action = torch.max(action,locked)
    return final_action

  def compute_loss(self, transition_data):
    mt1 = time.time()
    if self.settings['cuda'] and len(transition_data) > self.gpu_cap:
      gpu_batches = []
      td = transition_data
      while td:
        b = collate_transitions(td[:self.gpu_cap],self.settings, cudaize_state=True)
        gpu_batches.append(b)
        td = td[self.gpu_cap:]
    else:
      gpu_batches = [collate_transitions(transition_data,self.settings, cudaize_state=True)]       
    actions = flatten([b.action for b in gpu_batches])
    returns = self.settings.cudaize_var(torch.cat([b.reward for b in gpu_batches]))
    mt2 = time.time()
    batched_logits = []
    for b in gpu_batches:
      logits, values, _, aux_losses = self.forward(b.state, prev_obs=b.prev_obs,do_timing=True)
      batched_logits.append(logits)
    batched_logits = flatten(batched_logits)    
    mt3 = time.time()
    logprobs = []
    batched_clabels = flatten([b.state.clabels.unbind() for b in gpu_batches])
    # collated_batch.state.clabels
    num_learned = flatten((b.state.ext_data for b in gpu_batches))
    for (action, logits, clabels, learned_idx) in zip(actions,batched_logits, batched_clabels, num_learned):
      probs = F.softmax(logits,dim=1)
      locked = self.settings.cudaize_var(clabels[learned_idx[0]:learned_idx[1],CLABEL_LOCKED])
      pre_logprobs = probs.gather(1,self.settings.cudaize_var(action).view(-1,1)).log().view(-1)
      logprobs.append(((1-locked)*pre_logprobs).sum())
    adv_t = returns
    value_loss = 0.
    # Tracer()()
    logprobs = torch.stack(logprobs)
    # entropies = (-probs*all_logprobs).sum(1)    
    adv_t = (adv_t - adv_t.mean())
    if self.settings['use_sum']:
      pg_loss = (-Variable(adv_t)*logprobs).sum()
    else:
      pg_loss = (-Variable(adv_t)*logprobs).mean()

    loss = pg_loss    # + self.lambda_value*value_loss + self.lambda_aux*total_aux_loss
    mt4 = time.time()
    print('compute_loss: Collate: {}, forward: {}, compute: {}'.format(mt2-mt1,mt3-mt2,mt4-mt3))
    return loss, logits

class SatLinearPolicy(PolicyBase):
  def __init__(self, encoder=None, **kwargs):
    super(SatLinearPolicy, self).__init__(**kwargs)
    non_linearity = self.settings['policy_non_linearity']
    if self.policy_dim1:
      self.linear1 = nn.Linear(self.clabel_dim, self.policy_dim1)

    if self.policy_dim2:
      self.linear2 = nn.Linear(self.policy_dim1,self.policy_dim2)
      self.action_score = nn.Linear(self.policy_dim2,2)
    elif self.policy_dim1:
      self.action_score = nn.Linear(self.policy_dim1,2) 
    else:
      self.action_score = nn.Linear(self.clabel_dim,2) 

    if non_linearity is not None:
      self.activation = eval(non_linearity)
    else:
      self.activation = lambda x: x

  
  # state is just a (batched) vector of fixed size state_dim which should be expanded. 
  # vlabels are batch * max_vars * vlabel_dim

  # cmat is already "batched" into a single matrix

  def forward(self, obs, **kwargs):
    clabels = obs.clabels
    size = obs.clabels.shape

    aux_losses = []

    if self.settings['cuda'] and not self.settings['mp']:
      clabels = clabels.cuda()

    num_learned = obs.ext_data
    # Tracer()()
    inputs = clabels.view(-1,self.clabel_dim)

    # if size[0] > 1:
    #   break_every_tick(20)
    if self.policy_dim2:      
      outputs = self.action_score(self.activation(self.linear2(self.activation(self.linear1(inputs)))))
    elif self.policy_dim1:
      outputs = self.action_score(self.activation(self.linear1(inputs)))
    else:
      outputs = self.action_score(inputs)
    outputs_processed = []
    # print(num_learned)
    for i, (nl1, nl2) in enumerate(num_learned):
      outputs_processed.append(outputs[nl1:nl2])
      outputs = outputs[size[1]:]

    assert(outputs.shape[0]==0)    
    if any((x!=x).any() for x in outputs_processed):    # Check nans
      Tracer()()
    value = None
    return outputs_processed, value, clabels, aux_losses

  def get_allowed_actions(self, obs, **kwargs):
    pass

  def translate_action(self, action, obs, **kwargs):
    # print('Action is: {}'.format(action[:10]))
    return action

  def combine_actions(self, actions, **kwargs):    
    return actions
    # return torch.cat(actions)

  def select_action(self, obs_batch, **kwargs):
    logits, *_ = self.forward(obs_batch)
    assert(len(logits)==1)
    logits = logits[0]
    probs = F.softmax(logits,dim=1)
    ps = probs[:,0].detach().numpy()
    action = torch.from_numpy(np.random.binomial(1,p=ps)).unsqueeze(0)
    num_learned = obs_batch.ext_data[0]
    locked = obs_batch.clabels[0,num_learned[0]:num_learned[1],CLABEL_LOCKED].long().view(1,-1)
    final_action = torch.max(action,locked)

    return final_action

  def compute_loss(self, transition_data):
    collated_batch = collate_transitions(transition_data,self.settings)
    collated_batch.state = cudaize_obs(collated_batch.state, self.settings)
    returns = self.settings.cudaize_var(collated_batch.reward)
    batched_logits, values, _, aux_losses = self.forward(collated_batch.state, prev_obs=collated_batch.prev_obs)
    actions = collated_batch.action
    logprobs = []
    batched_clabels = collated_batch.state.clabels
    num_learned = collated_batch.state.ext_data
    for (action, logits, clabels, learned_idx) in zip(actions,batched_logits, batched_clabels, num_learned):      
      probs = F.softmax(logits,dim=1).clamp(min=0.001,max=0.999)
      locked = clabels[learned_idx[0]:learned_idx[1],CLABEL_LOCKED]
      pre_logprobs = probs.gather(1,action.view(-1,1)).log().view(-1)
      action_probs = ((1-locked)*pre_logprobs).sum()
      if (action_probs!=action_probs).any():
        Tracer()()
      logprobs.append(action_probs)
    adv_t = returns
    value_loss = 0.
    logprobs = torch.stack(logprobs)
    # entropies = (-probs*all_logprobs).sum(1)    
    adv_t = (adv_t - adv_t.mean())
    if self.settings['use_sum']:
      pg_loss = (-Variable(adv_t)*logprobs).sum()
    else:
      pg_loss = (-Variable(adv_t)*logprobs).mean()

    total_aux_loss = sum(aux_losses) if aux_losses else 0.    
    loss = pg_loss + self.lambda_value*value_loss + self.lambda_aux*total_aux_loss
    return loss, logits

class SatMiniLinearPolicy(PolicyBase):
  def __init__(self, encoder=None, **kwargs):
    super(SatMiniLinearPolicy, self).__init__(**kwargs)
    non_linearity = self.settings['policy_non_linearity']
    if self.policy_dim1:
      self.linear1 = nn.Linear(1, self.policy_dim1)

    if self.policy_dim2:
      self.linear2 = nn.Linear(self.policy_dim1,self.policy_dim2)
      self.action_score = nn.Linear(self.policy_dim2,1)
    elif self.policy_dim1:
      self.action_score = nn.Linear(self.policy_dim1,1) 
    else:
      self.action_score = nn.Linear(1,1) 

    if non_linearity is not None:
      self.activation = eval(non_linearity)
    else:
      self.activation = lambda x: x
    self.use_bn = self.settings['use_bn']
    if self.use_bn:
      self.cnorm_layer = nn.BatchNorm1d(self.clabel_dim)

  
  # state is just a (batched) vector of fixed size state_dim which should be expanded. 
  # vlabels are batch * max_vars * vlabel_dim

  # cmat is already "batched" into a single matrix

  def forward(self, obs, **kwargs):
    clabels = obs.clabels
    size = obs.clabels.shape

    aux_losses = []

    if self.settings['cuda'] and not self.settings['mp']:
      clabels = clabels.cuda()

    num_learned = obs.ext_data
    clabels = clabels.view(-1,self.clabel_dim)
    if self.use_bn:
      clabels = self.cnorm_layer(clabels)
    inputs = clabels[:,CLABEL_LBD].unsqueeze(1)

    # if size[0] > 1:
    #   break_every_tick(20)
    if self.policy_dim2:      
      outputs = self.action_score(self.activation(self.linear2(self.activation(self.linear1(inputs)))))
    elif self.policy_dim1:
      outputs = self.action_score(self.activation(self.linear1(inputs)))
    else:
      outputs = self.action_score(inputs)
    outputs_processed = []
    # print(num_learned)
    for i, (nl1, nl2) in enumerate(num_learned):
      outputs_processed.append(outputs[nl1:nl2])
      outputs = outputs[size[1]:]

    # Tracer()()
    assert(outputs.shape[0]==0)    
    if any((x!=x).any() for x in outputs_processed):    # Check nans
      Tracer()()
    value = None
    return outputs_processed, value, clabels, aux_losses

  def get_allowed_actions(self, obs, **kwargs):
    pass

  def translate_action(self, action, obs, **kwargs):
    # print('Action is: {}'.format(action[:10]))
    return action

  def combine_actions(self, actions, **kwargs):    
    return actions
    # return torch.cat(actions)

  def select_action(self, obs_batch, **kwargs):
    logits, *_ = self.forward(obs_batch)
    assert(len(logits)==1)
    logits = logits[0]
    probs = torch.sigmoid(logits)
    ps = probs[:,0].detach().cpu().numpy()    # cpu() just in case, if we're in SP+cuda
    action = torch.from_numpy(np.random.binomial(1,p=ps)).unsqueeze(0)
    num_learned = obs_batch.ext_data[0]
    locked = obs_batch.clabels[0,num_learned[0]:num_learned[1],CLABEL_LOCKED].long().view(1,-1)
    final_action = torch.max(action,locked)

    return final_action

  def compute_loss(self, transition_data):
    collated_batch = collate_transitions(transition_data,self.settings)
    collated_batch.state = cudaize_obs(collated_batch.state)
    returns = self.settings.cudaize_var(collated_batch.reward)
    batched_logits, values, _, aux_losses = self.forward(collated_batch.state, prev_obs=collated_batch.prev_obs)
    actions = collated_batch.action
    logprobs = []
    batched_clabels = collated_batch.state.clabels
    num_learned = collated_batch.state.ext_data
    for (action, logits, clabels, learned_idx) in zip(actions,batched_logits, batched_clabels, num_learned):      
      ps = torch.sigmoid(logits).clamp(min=0.001,max=0.999)
      locked = clabels[learned_idx[0]:learned_idx[1],CLABEL_LOCKED]
      action = self.settings.cudaize_var(action)
      all_action_probs = action.float().view_as(ps)*ps + (1-action.float().view_as(ps))*(1-ps)
      pre_logprobs = all_action_probs.log().view(-1)
      action_probs = ((1-locked)*pre_logprobs).sum()
      if (action_probs!=action_probs).any():
        Tracer()()
      logprobs.append(action_probs)
    adv_t = returns
    value_loss = 0.
    logprobs = torch.stack(logprobs)
    # entropies = (-probs*all_logprobs).sum(1)    
    adv_t = (adv_t - adv_t.mean())
    # Tracer()()
    if self.settings['use_sum']:
      pg_loss = (-Variable(adv_t)*logprobs).sum()
    else:
      pg_loss = (-Variable(adv_t)*logprobs).mean()

    total_aux_loss = sum(aux_losses) if aux_losses else 0.    
    loss = pg_loss + self.lambda_value*value_loss + self.lambda_aux*total_aux_loss
    return loss, logits

class SatRandomPolicy(PolicyBase):
  def __init__(self, encoder=None, **kwargs):
    super(SatRandomPolicy, self).__init__(**kwargs)    
    self.p = self.settings['sat_random_p']
    self.pval = nn.Parameter(torch.tensor(self.p), requires_grad=False)
  
  def forward(self, obs, **kwargs):
    pass

  def get_allowed_actions(self, obs, **kwargs):
    pass

  def translate_action(self, action, obs, **kwargs):
    # print('Action is: {}'.format(action[:10]))
    return action

  def combine_actions(self, actions, **kwargs):    
    return actions
    # return torch.cat(actions)

  def select_action(self, obs_batch, **kwargs):
    assert(obs_batch.clabels.shape[0]==1)
    num_learned = obs_batch.ext_data[0]
    action = torch.from_numpy(np.random.binomial(1,p=self.pval,size=num_learned[1]-num_learned[0])).unsqueeze(0)
    locked = obs_batch.clabels[0,num_learned[0]:num_learned[1],CLABEL_LOCKED].long().view(1,-1)
    final_action = torch.max(action,locked)

    # Tracer()()
    return final_action

  def compute_loss(self, transition_data):
    return None, None

class SatBernoulliPolicy(PolicyBase):
  def __init__(self, encoder=None, **kwargs):
    super(SatBernoulliPolicy, self).__init__(**kwargs)    
    self.p = self.settings['sat_random_p']
    self.pval = nn.Parameter(torch.tensor(self.p), requires_grad=True)
  
  # state is just a (batched) vector of fixed size state_dim which should be expanded. 
  # vlabels are batch * max_vars * vlabel_dim

  # cmat is already "batched" into a single matrix

  def forward(self, obs, **kwargs):
    clabels = obs.clabels
    size = obs.clabels.shape

    aux_losses = []

    if self.settings['cuda'] and not self.settings['mp']:
      clabels = clabels.cuda()

    num_learned = obs.ext_data
    inputs = clabels.view(-1,self.clabel_dim)[:,CLABEL_LBD].unsqueeze(1)
    outputs = torch.sigmoid(self.pval.view(1)).expand(inputs.shape[0]).view_as(inputs)
    # if size[0] > 1:
    #   break_every_tick(1)

    outputs_processed = []
    # print(num_learned)
    for i, (nl1, nl2) in enumerate(num_learned):
      outputs_processed.append(outputs[nl1:nl2])
      outputs = outputs[size[1]:]

    # Tracer()()
    assert(outputs.shape[0]==0)    
    if any((x!=x).any() for x in outputs_processed):    # Check nans
      Tracer()()
    value = None
    return outputs_processed, value, clabels, aux_losses


  def get_allowed_actions(self, obs, **kwargs):
    pass

  def translate_action(self, action, obs, **kwargs):
    # print('Action is: {}'.format(action[:10]))
    return action

  def combine_actions(self, actions, **kwargs):    
    return actions
    # return torch.cat(actions)

  def select_action(self, obs_batch, **kwargs):
    assert(obs_batch.clabels.shape[0]==1)
    logits, *_ = self.forward(obs_batch)
    assert(len(logits)==1)
    logits = logits[0]
    ps = logits[:,0].detach().cpu().numpy()
    action = torch.from_numpy(np.random.binomial(1,p=ps)).unsqueeze(0)
    num_learned = obs_batch.ext_data[0]
    locked = obs_batch.clabels[0,num_learned[0]:num_learned[1],CLABEL_LOCKED].long().view(1,-1)
    final_action = torch.max(action,locked)

    # Tracer()()
    return final_action

  def compute_loss(self, transition_data):    
    collated_batch = collate_transitions(transition_data,self.settings)
    collated_batch.state = cudaize_obs(collated_batch.state)
    returns = self.settings.cudaize_var(collated_batch.reward)
    batched_logits, values, _, aux_losses = self.forward(collated_batch.state, prev_obs=collated_batch.prev_obs)
    actions = collated_batch.action
    logprobs = []
    batched_clabels = collated_batch.state.clabels
    num_learned = collated_batch.state.ext_data
    for (action, logits, clabels, learned_idx) in zip(actions,batched_logits, batched_clabels, num_learned):      
      ps = logits.clamp(min=0.001,max=0.999)
      locked = clabels[learned_idx[0]:learned_idx[1],CLABEL_LOCKED]
      action = self.settings.cudaize_var(action)
      all_action_probs = action.float().view_as(ps)*ps + (1-action.float().view_as(ps))*(1-ps)
      pre_logprobs = all_action_probs.log().view(-1)
      action_probs = ((1-locked)*pre_logprobs).sum()
      if (action_probs!=action_probs).any():
        Tracer()()
      logprobs.append(action_probs)
    adv_t = returns
    value_loss = 0.
    logprobs = torch.stack(logprobs)
    # entropies = (-probs*all_logprobs).sum(1)    
    adv_t = (adv_t - adv_t.mean())
    # Tracer()()
    if self.settings['use_sum']:
      pg_loss = (-Variable(adv_t)*logprobs).sum()
    else:
      pg_loss = (-Variable(adv_t)*logprobs).mean()

    total_aux_loss = sum(aux_losses) if aux_losses else 0.    
    loss = pg_loss + self.lambda_value*value_loss + self.lambda_aux*total_aux_loss
    return loss, logits

class SatLBDPolicy(PolicyBase):
  def __init__(self, encoder=None, **kwargs):
    super(SatLBDPolicy, self).__init__(oracletype='glucose', **kwargs)
    self.action_score = nn.Linear(2,1)
  
  def forward(self, obs, **kwargs):
    return None

  def get_allowed_actions(self, obs, **kwargs):
    pass

  def translate_action(self, action, obs, **kwargs):
    # print('Action is: {}'.format(action[:10]))
    return action

  def combine_actions(self, actions, **kwargs):    
    return actions
    # return torch.cat(actions)

  def select_action(self, obs_batch, **kwargs):
    return [np.empty(shape=(0, 0), dtype=bool)], 0

  def compute_loss(self, transition_data, **kwargs):
    return None, None

class SatDeleteAllPolicy(PolicyBase):
  def __init__(self, encoder=None, **kwargs):
    super(SatDeleteAllPolicy, self).__init__(**kwargs)
    self.action_score = nn.Linear(2,1)
  
  def forward(self, obs, **kwargs):
    return None

  def get_allowed_actions(self, obs, **kwargs):
    pass

  def translate_action(self, action, obs, **kwargs):
    num_learned = obs.ext_data
    locked = obs.clabels[num_learned[0]:num_learned[1],CLABEL_LOCKED].long().view(-1)    
    return locked

  def combine_actions(self, actions, **kwargs):    
    return actions
    # return torch.cat(actions)

  def select_action(self, obs_batch, **kwargs):
    return [np.empty(shape=(0, 0), dtype=bool)]

  def compute_loss(self, transition_data, **kwargs):
    return None, None



class SatFixedThresholdPolicy(PolicyBase):
  def __init__(self, encoder=None, **kwargs):
    super(SatFixedThresholdPolicy, self).__init__(oracletype='lbd_threshold', **kwargs)
    self.t = self.settings['init_threshold']
    self.threshold = nn.Parameter(torch.tensor(self.t), requires_grad=False)
    
  # state is just a (batched) vector of fixed size state_dim which should be expanded. 
  # vlabels are batch * max_vars * vlabel_dim

  # cmat is already "batched" into a single matrix

  def forward(self, obs, **kwargs):    
    return self.threshold


  def get_allowed_actions(self, obs, **kwargs):
    pass

  def translate_action(self, action, obs, **kwargs):
    # Tracer()()
    return action

  def combine_actions(self, actions, **kwargs):    
    return actions
    # return torch.cat(actions)

  def select_action(self, obs_batch, **kwargs):
    
    return [self.t], 0.

  def compute_loss(self, transition_data):        
    return 0., self.threshold

class SatFreeThresholdPolicy(PolicyBase):
  def __init__(self, encoder=None, **kwargs):
    super(SatFreeThresholdPolicy, self).__init__(oracletype='lbd_threshold', **kwargs)
    self.t = float(self.settings['init_threshold'])
    self.threshold = nn.Parameter(torch.FloatTensor([self.t]), requires_grad=True)
    self.sigma = self.settings.FloatTensor(np.array(float(self.settings['threshold_sigma'])))
    self.logger.info('self.threshold is:')
    self.logger.info(self.threshold)

  def forward(self, obs, **kwargs):
    clabels = obs.clabels.view(-1,self.clabel_dim)
    return self.threshold, clabels


  def get_allowed_actions(self, obs, **kwargs):
    pass

  def translate_action(self, action, obs, **kwargs):
    threshold = action
    rc = obs.clabels[:,CLABEL_LBD] < threshold
    a = rc.detach()
    num_learned = obs.ext_data
    locked = obs.clabels[num_learned[0]:num_learned[1],CLABEL_LOCKED].long().view(1,-1)
    final_action = torch.max(a.long(),locked).view(-1)
    # break_every_tick(5)

    return final_action
  def combine_actions(self, actions, **kwargs):    
    return actions
    # return torch.cat(actions)

  def select_action(self, obs_batch, deterministic=False, log_threshold=False, **kwargs):
    obs_batch = collate_observations([obs_batch])
    assert(obs_batch.clabels.shape[0]==1)
    threshold, clabels = self.forward(obs_batch, **kwargs)
    if deterministic:
      if log_threshold:
        self.logger.info('Threshold is {}'.format(threshold))
      return (threshold, clabels)
    m = Normal(threshold,self.sigma)
    sampled_threshold = m.sample()

    return sampled_threshold, 0

  def get_logprobs(self, outputs, collated_batch):    
    actions = collated_batch.action
    actions = torch.cat([a[0] for a in actions]).view(-1)
    threshold = outputs.view(-1)
    return gaussian_logprobs(threshold.view(-1,1),self.sigma.view(1),actions.view(-1,1)).view(-1), 0

  def compute_loss(self, transition_data, **kwargs):    
    collated_batch = collate_transitions(transition_data,self.settings)
    collated_batch.state = cudaize_obs(collated_batch.state)
    returns = self.settings.cudaize_var(collated_batch.reward)
    outputs, _ = self.forward(collated_batch.state, prev_obs=collated_batch.prev_obs)    
    logprobs = self.get_logprobs(outputs, collated_batch)
    adv_t = returns
    value_loss = 0.
    adv_t = (adv_t - adv_t.mean())
    if self.settings['use_sum']:
      pg_loss = (-adv_t*logprobs).sum()
    else:
      pg_loss = (-adv_t*logprobs).mean()

    loss = pg_loss
    return loss, outputs


class SCPolicyBase(PolicyBase):
  def __init__(self, **kwargs):
    super(SCPolicyBase, self).__init__(**kwargs)
    self.snorm_window = self.settings['vbn_window']    
    # self.sigma = self.settings.FloatTensor(np.array(float(self.settings['threshold_sigma'])))
    if self.state_bn:
      # self.state_vbn = MovingAverageVBN((self.snorm_window,self.state_dim))
      self.state_vbn = NodeAverageAndStdVBN(self.state_dim, 'state_vbn', **kwargs)
    if self.use_bn:
      self.clabels_vbn = MovingAverageAndStdVBN((self.snorm_window,self.clabel_dim))      
    sublayers = []
    prev = self.input_dim()
    self.policy_layers = nn.Sequential()
    n = 0
    num_layers = len([x for x in self.settings['policy_layers'] if type(x) is int])
    for (i,x) in enumerate(self.settings['policy_layers']):
      if x == 'r':
        self.policy_layers.add_module('activation_{}'.format(i), nn.ReLU())
      elif x == 'lr':
        self.policy_layers.add_module('activation_{}'.format(i), nn.LeakyReLU())
      elif x == 'h':
        self.policy_layers.add_module('activation_{}'.format(i), nn.Tanh())        
      elif x == 's':
        self.policy_layers.add_module('activation_{}'.format(i), nn.Sigmoid())        
      else:
        n += 1
        layer = nn.Linear(prev,x)
        prev = x
        if self.settings['init_threshold'] is not None:          
          if n == num_layers:
            nn.init.constant_(layer.weight,0.)
            nn.init.constant_(layer.bias,self.settings['init_threshold'])
        self.policy_layers.add_module('linear_{}'.format(i), layer)

  # state is just a (batched) vector of fixed size state_dim which should be expanded. 
  # vlabels are batch * max_vars * vlabel_dim

  # cmat is already "batched" into a single matrix

  def forward(self, obs, **kwargs):
    state = obs.state    
    clabels = obs.clabels
    if self.state_bn:      
      state = self.state_vbn(state, **kwargs)
    if self.use_bn and clabels is not None:
      clabels = clabels.view(-1,self.clabel_dim)
      clabels = self.clabels_vbn(clabels, **kwargs)    
    return state, clabels


  def get_allowed_actions(self, obs, **kwargs):
    pass

  def combine_actions(self, actions, **kwargs):    
    return actions
    # return torch.cat(actions)

  def compute_loss(self, transition_data, **kwargs):    
    state = torch.cat([x.state.state for x in transition_data])
    collated_batch = EmptyState
    collated_batch.state = self.settings.cudaize_var(state)
    # collated_batch = collate_transitions(transition_data,self.settings)
    # collated_batch.state = cudaize_obs(collated_batch.state)
    returns = self.settings.FloatTensor([x.reward for x in transition_data])
    outputs = self.forward(collated_batch)
    actions = [x.action for x in transition_data]
    logprobs, entropy = self.get_logprobs(outputs, actions)
    adv_t = returns
    value_loss = 0.
    adv_t = (adv_t - adv_t.mean())
    pg_loss = (-adv_t*logprobs)
    loss = pg_loss - self.entropy_alpha*entropy
    if self.settings['use_sum']:
      total_loss = loss.sum()
    else:
      total_loss = loss.mean()

    # Recompute moving averages

    if self.state_bn:
      self.state_vbn.recompute_moments(collated_batch.state.detach())
    # if self.use_bn:
    #   z = collated_batch.state.clabels.detach().view(-1,6)
    #   self.clabels_vbn.recompute_moments(z.mean(dim=0).unsqueeze(0),z.std(dim=0).unsqueeze(0))

    
    return total_loss, outputs


class SatMiniHyperPlanePolicy(SCPolicyBase):
  def __init__(self, **kwargs):
    super(SatMiniHyperPlanePolicy, self).__init__(**kwargs)
    self.sigma = self.settings.FloatTensor(np.array(float(self.settings['threshold_sigma'])))
    self.threshold_shift = float(self.settings['threshold_shift'])

  def input_dim(self):
    return self.settings['state_dim']

  def forward(self, obs, **kwargs):    
    state, clabels = super(SatMiniHyperPlanePolicy, self).forward(obs)
    return self.policy_layers(state), clabels

    return threshold, clabels

  def translate_action(self, action, obs, **kwargs):
    sampled_output, clabels = action
    mini_clabels = clabels[:,CLABEL_LBD]
    # break_every_tick(5)
    plane = sampled_output[:,0] + self.threshold_shift
    shift = sampled_output[:,1]
    rc = ((mini_clabels * plane) - shift) < 0
    a = rc.detach()
    num_learned = obs.ext_data
    locked = obs.clabels[num_learned[0]:num_learned[1],CLABEL_LOCKED].long().view(1,-1)
    final_action = torch.max(a.long(),locked).view(-1)

    return final_action

  def select_action(self, obs_batch, training=True, **kwargs):
    assert(obs_batch.clabels.shape[0]==1)
    output, clabels = self.forward(obs_batch)
    if not training:
      return [(output, clabels)]
    m = MultivariateNormal(output,torch.eye(2)*self.sigma*self.sigma)
    sampled_output = m.sample()
    if self.settings['log_threshold']:      
      if self.shelf_key not in self.shelf_file.keys():        
        self.shelf_file[self.shelf_key] = []
      tmp = self.shelf_file[self.shelf_key]
      tmp.append((output.detach().numpy(),sampled_output.detach().numpy()))
      self.shelf_file[self.shelf_key] = tmp

    return [(sampled_output, clabels)]

  def get_logprobs(self, outputs, collated_batch):    
    actions = collated_batch.action
    actions = torch.cat([a[0] for a in actions])    
    return gaussian_logprobs(outputs,self.sigma,actions), 0

class SatMini2HyperPlanePolicy(SCPolicyBase):
  def __init__(self, **kwargs):
    super(SatMini2HyperPlanePolicy, self).__init__(**kwargs)
    self.sigma = self.settings.FloatTensor(np.array(float(self.settings['threshold_sigma'])))
    self.hp_normalize = self.settings['hp_normalize_shift']

  def input_dim(self):
    return self.settings['state_dim']

  def forward(self, obs, **kwargs):    
    state, clabels = super(SatMini2HyperPlanePolicy, self).forward(obs)
    rc = self.policy_layers(state)
    if self.hp_normalize:
      hp = torch.cat([rc[:,:2],rc[:,2].reshape(rc.shape[0],1)*2],dim=1)
      rc = hp
    return rc, clabels

  def translate_action(self, action, obs, **kwargs):
    sampled_output, clabels = action
    mini_clabels = clabels[:,[CLABEL_LBD,4]]
    plane = sampled_output[:,:2]
    shift = sampled_output[:,2]
    # break_every_tick(50)
    rc = ((mini_clabels * plane).sum(dim=1) - shift) < 0
    a = rc.detach()
    num_learned = obs.ext_data
    locked = obs.clabels[num_learned[0]:num_learned[1],CLABEL_LOCKED].long().view(1,-1)
    final_action = torch.max(a.long(),locked).view(-1)

    return final_action

  def select_action(self, obs_batch, training=True, **kwargs):
    assert(obs_batch.clabels.shape[0]==1)
    output, clabels = self.forward(obs_batch)
    if not training:
      return [(output, clabels)]
    m = MultivariateNormal(output,torch.eye(3)*self.sigma*self.sigma)
    sampled_output = m.sample()
    if self.settings['log_threshold']:      
      if self.shelf_key not in self.shelf_file.keys():        
        self.shelf_file[self.shelf_key] = []
      tmp = self.shelf_file[self.shelf_key]
      tmp.append((output.detach().numpy(),sampled_output.detach().numpy()))
      self.shelf_file[self.shelf_key] = tmp

    return [(sampled_output, clabels)]

  def get_logprobs(self, outputs, collated_batch):    
    actions = collated_batch.action
    actions = torch.cat([a[0] for a in actions])    
    return gaussian_logprobs(outputs,self.sigma,actions), 0


class SatMini3HyperPlanePolicy(SCPolicyBase):
  def __init__(self, **kwargs):
    super(SatMini3HyperPlanePolicy, self).__init__(**kwargs)
    self.sigma = self.settings.FloatTensor(np.array(float(self.settings['threshold_sigma'])))

  def input_dim(self):
    return self.settings['state_dim']

  def forward(self, obs, **kwargs):    
    state, clabels = super(SatMini3HyperPlanePolicy, self).forward(obs)
    return self.policy_layers(state), clabels

    return threshold, clabels

  def translate_action(self, action, obs, **kwargs):
    sampled_output, clabels = action
    mini_clabels = clabels[:,[CLABEL_LBD,4,1]]
    # break_every_tick(5)
    plane = sampled_output[:,:3]
    shift = sampled_output[:,3]
    rc = ((mini_clabels * plane).sum(dim=1) - shift) < 0
    a = rc.detach()
    num_learned = obs.ext_data
    locked = obs.clabels[num_learned[0]:num_learned[1],CLABEL_LOCKED].long().view(1,-1)
    final_action = torch.max(a.long(),locked).view(-1)

    return final_action

  def select_action(self, obs_batch, training=True, **kwargs):
    assert(obs_batch.clabels.shape[0]==1)
    output, clabels = self.forward(obs_batch)
    if not training:
      return [(output, clabels)]
    m = MultivariateNormal(output,torch.eye(4)*self.sigma*self.sigma)
    sampled_output = m.sample()
    if self.settings['log_threshold']:      
      if self.shelf_key not in self.shelf_file.keys():        
        self.shelf_file[self.shelf_key] = []
      tmp = self.shelf_file[self.shelf_key]
      tmp.append((output.detach().numpy(),sampled_output.detach().numpy()))
      self.shelf_file[self.shelf_key] = tmp

    return [(sampled_output, clabels)]

  def get_logprobs(self, outputs, collated_batch):    
    actions = collated_batch.action
    actions = torch.cat([a[0] for a in actions])    
    return gaussian_logprobs(outputs,self.sigma,actions), 0



class SatHyperPlanePolicy(SCPolicyBase):
  def __init__(self, **kwargs):
    super(SatHyperPlanePolicy, self).__init__(**kwargs)
    self.sigma = self.settings.FloatTensor(np.array(float(self.settings['threshold_sigma'])))

  def input_dim(self):
    return self.settings['state_dim']

  def forward(self, obs, **kwargs):    
    state, clabels = super(SatHyperPlanePolicy, self).forward(obs)
    return self.policy_layers(state), clabels

    return threshold, clabels

  def translate_action(self, action, obs, **kwargs):
    sampled_output, clabels = action
    plane = sampled_output[:,:6]
    shift = sampled_output[:,6]
    rc = ((clabels * plane).sum(dim=1) - shift) > 0
    a = rc.detach()
    num_learned = obs.ext_data
    locked = obs.clabels[num_learned[0]:num_learned[1],CLABEL_LOCKED].long().view(1,-1)
    final_action = torch.max(a.long(),locked).view(-1)
    # break_every_tick(5)

    return final_action

  def select_action(self, obs_batch, training=True, **kwargs):
    assert(obs_batch.clabels.shape[0]==1)
    output, clabels = self.forward(obs_batch)
    if not training:
      return [(output, clabels)]
    m = MultivariateNormal(output,torch.eye(self.clabel_dim+1)*self.sigma*self.sigma)
    sampled_output = m.sample()
    if self.settings['log_threshold']:      
      if self.shelf_key not in self.shelf_file.keys():        
        self.shelf_file[self.shelf_key] = []
      tmp = self.shelf_file[self.shelf_key]
      tmp.append((output.detach().numpy(),sampled_output.detach().numpy()))
      self.shelf_file[self.shelf_key] = tmp

    return [(sampled_output, clabels)]

  def get_logprobs(self, outputs, collated_batch):    
    actions = collated_batch.action
    actions = torch.cat([a[0] for a in actions])    
    return gaussian_logprobs(outputs,self.sigma,actions)




class SatThresholdSCPolicy(SCPolicyBase):
  def __init__(self, **kwargs):
    super(SatThresholdSCPolicy, self).__init__(**kwargs)
    self.sigma = self.settings.FloatTensor(np.array(float(self.settings['threshold_sigma'])))

  def input_dim(self):
    return self.settings['state_dim']+self.settings

  def forward(self, obs, **kwargs):
    state, clabels = super(SatThresholdSCPolicy, self).forward(obs, **kwargs)
    threshold = self.policy_layers(state) * self.threshold_scale    

    return threshold, clabels

  def translate_action(self, action, obs, **kwargs):
    threshold, clabels = action
    rc = clabels[:,CLABEL_LBD] < threshold
    a = rc.detach()
    num_learned = obs.ext_data
    locked = obs.clabels[num_learned[0]:num_learned[1],CLABEL_LOCKED].long().view(1,-1)
    final_action = torch.max(a.long(),locked).view(-1)
    # break_every_tick(5)

    return final_action

  def select_action(self, obs_batch, training=True, **kwargs):
    assert(obs_batch.clabels.shape[0]==1)
    threshold, clabels = self.forward(obs_batch)
    break_every_tick(50)
    if not training:
      return [(threshold, clabels)]
    m = Normal(threshold,self.sigma)
    sampled_threshold = m.sample()
    if self.settings['log_threshold']:      
      if self.shelf_key not in self.shelf_file.keys():        
        self.shelf_file[self.shelf_key] = []
      tmp = self.shelf_file[self.shelf_key]
      tmp.append((threshold.detach().numpy(),sampled_threshold.detach().numpy()))
      self.shelf_file[self.shelf_key] = tmp

    return [(sampled_threshold, clabels)]

  def get_logprobs(self, outputs, collated_batch):    
    actions = collated_batch.action
    actions = torch.cat([a[0] for a in actions]).view(-1)
    threshold = outputs.view(-1)
    return gaussian_logprobs(threshold,self.sigma,actions)

class SatThresholdStatePolicy(SCPolicyBase):
  def __init__(self, **kwargs):
    super(SatThresholdStatePolicy, self).__init__(oracletype='lbd_threshold', **kwargs)
    self.sigma = self.settings.FloatTensor(np.array(float(self.settings['threshold_sigma'])))
    self.threshold_scale = self.settings.FloatTensor(np.array(float(self.settings['threshold_scale'])))

  def input_dim(self):
    return self.settings['state_dim']

  def forward(self, obs, log_threshold=False, **kwargs):
    state, clabels = super(SatThresholdStatePolicy, self).forward(obs, **kwargs)
    threshold = self.policy_layers(state) * self.threshold_scale
    if every_tick(20) or log_threshold:
      self.logger.debug('Threshold before sampling: {}'.format(threshold))

    return threshold, clabels

  def translate_action(self, action, obs, **kwargs):
    threshold, clabels = action
    # print('Threshold is {}'.format(threshold))
    rc = clabels[:,CLABEL_LBD] < threshold
    a = rc.detach()
    num_learned = obs.ext_data
    locked = obs.clabels[num_learned[0]:num_learned[1],CLABEL_LOCKED].long().view(1,-1)
    final_action = torch.max(a.long(),locked).view(-1)
    # break_every_tick(5)

    return final_action

  def select_action(self, obs_batch, deterministic=False, **kwargs):
    obs_batch = collate_observations([obs_batch])
    assert(obs_batch.clabels.shape[0]==1)
    threshold, clabels = self.forward(obs_batch, **kwargs)
    if deterministic:
      return (threshold, clabels)
    m = Normal(threshold,self.sigma)
    sampled_threshold = m.sample()
    if self.settings['log_threshold']:      
      if self.shelf_key not in self.shelf_file.keys():        
        self.shelf_file[self.shelf_key] = []
      tmp = self.shelf_file[self.shelf_key]
      tmp.append((threshold.detach().numpy(),sampled_threshold.detach().numpy()))
      self.shelf_file[self.shelf_key] = tmp

    return (sampled_threshold, clabels), 0

  def get_logprobs(self, outputs, collated_batch):    
    actions = collated_batch.action
    actions = torch.cat([a[0] for a in actions]).view(-1)
    threshold = outputs.view(-1)
    return gaussian_logprobs(threshold.view(-1,1),self.sigma.view(1),actions.view(-1,1)).view(-1), 0

class SatPercentagePolicy(SCPolicyBase):
  def __init__(self, **kwargs):
    super(SatPercentagePolicy, self).__init__(oracletype='percentage', **kwargs)
    self.tsigma = self.settings.FloatTensor(np.array(float(self.settings['threshold_sigma'])))
    self.psigma = self.settings.FloatTensor(np.array(float(self.settings['percentage_sigma'])))
    self.threshold_shift = self.settings['threshold_shift']
    if self.settings['sat_init_percentage']:
      nn.init.constant_(self.policy_layers.linear_2.bias[0],0.)
      nn.init.constant_(self.policy_layers.linear_2.bias[1],self.settings['init_threshold'])

  def input_dim(self):
    return self.settings['state_dim']

  def forward(self, obs, **kwargs):    
    state, clabels = super(SatPercentagePolicy, self).forward(obs)
    rc = self.policy_layers(state)
    # Index 0 is for the percentage, Index 1 is for minimal drop
    rc = torch.stack([torch.sigmoid(rc[:,0]),self.threshold_shift+torch.relu(rc[:,1])],dim=1)
    return rc, clabels

  def translate_action(self, action, obs, **kwargs):
    sampled_output, clabels = action
    return sampled_output.detach().view(-1).tolist()

  def select_action(self, obs_batch, training=True, **kwargs):
    assert(obs_batch.clabels.shape[0]==1)
    output, clabels = self.forward(obs_batch)
    if not training:
      return [(output, clabels)]
    m = MultivariateNormal(output,torch.eye(2)*self.settings.FloatTensor([self.psigma*self.psigma, self.tsigma*self.tsigma]))
    sampled_output = m.sample()
    if self.settings['log_threshold']:      
      if self.shelf_key not in self.shelf_file.keys():        
        self.shelf_file[self.shelf_key] = []
      tmp = self.shelf_file[self.shelf_key]
      tmp.append((output.detach().numpy(),sampled_output.detach().numpy()))
      self.shelf_file[self.shelf_key] = tmp
    return [(sampled_output, clabels)]

  def get_logprobs(self, outputs, collated_batch):    
    actions = torch.cat([a[0] for a in collated_batch.action])
    return gaussian_logprobs(outputs,self.settings.FloatTensor([self.psigma,self.tsigma]),actions)

class SatDiscreteThresholdPolicy(SCPolicyBase):
  def __init__(self, **kwargs):
    super(SatDiscreteThresholdPolicy, self).__init__(oracletype='lbd_threshold', **kwargs)
    self.threshold_base = self.settings['sat_discrete_threshold_base']

  def input_dim(self):
    return self.settings['state_dim']

  def forward(self, obs, **kwargs):
    state, clabels = super(SatDiscreteThresholdPolicy, self).forward(obs, **kwargs)
    logits = self.policy_layers(state)

    return logits         # We do NOT use clabels vbn, so just return logits
    # return logits, clabels


  # This is going to MUTATE the obs object. Specifically, it deletes the reference to obs.clabels in order to save
  # memory before we save the observation in the batch trace.

  def translate_action(self, action, obs, **kwargs):        
    obs.clabels = None
    return [action + self.threshold_base]    # [2,3,...]
    # # print('Threshold is {}'.format(threshold))
    # rc = obs.clabels[:,CLABEL_LBD] < float(threshold)
    # a = rc.detach()
    # num_learned = obs.ext_data
    # locked = obs.clabels[num_learned[0]:num_learned[1],CLABEL_LOCKED].long().view(1,-1)
    # final_action = torch.max(a.long(),locked).view(-1)
    # obs.clabels = None
    # break_every_tick(5)

    # return final_action

  def select_action(self, obs_batch, deterministic=False, log_threshold=False, **kwargs):    
    logits = self.forward(obs_batch, **kwargs)
    if every_tick(20) or log_threshold:
      self.logger.info('State is:')
      self.logger.info(obs_batch.state)
      self.logger.info('Logits are: {}'.format(logits))    
      if log_threshold:
        if self.shelf_key not in self.shelf_file.keys():        
          self.shelf_file[self.shelf_key] = []
        tmp = self.shelf_file[self.shelf_key]
        tmp.append((obs_batch.state.detach().numpy(),logits.detach().numpy()))
        self.shelf_file[self.shelf_key] = tmp

    if deterministic:
      action = int(torch.argmax(logits.view(-1)))
      if every_tick(20) or log_threshold:
        self.logger.info('Threshold is {}'.format(action+2))
      return action, 0
    self.m = Categorical(logits=logits.view(-1))
    action=int(self.m.sample().detach())    
    if every_tick(20) or log_threshold:
      self.logger.info('Threshold is {}'.format(action+2))

    return action, float(self.m.entropy().detach())

  def get_logprobs(self, outputs, actions):    
    actions = self.settings.LongTensor(actions)
    logits = outputs
    probs = F.softmax(logits, dim=1)
    all_logprobs = safe_logprobs(probs)
    entropy = -(probs*all_logprobs).sum(dim=1)
    logprobs = all_logprobs.gather(1,actions.view(-1,1)).squeeze()
    return logprobs, entropy

class SatFreeDiscretePolicy(PolicyBase):
  def __init__(self, encoder=None, **kwargs):
    super(SatFreeDiscretePolicy, self).__init__(oracletype='lbd_threshold', **kwargs)
    self.t = float(self.settings['init_threshold'])
    self.threshold_base = self.settings['sat_discrete_threshold_base']
    self.num_actions = self.settings['sat_num_free_actions']
    self.threshold_logits = nn.Parameter(torch.FloatTensor(self.num_actions*[self.t]), requires_grad=True)

  def forward(self, obs, **kwargs):    
    return None


  def get_allowed_actions(self, obs, **kwargs):
    pass

  # This is going to MUTATE the obs object. Specifically, it deletes the reference to obs.clabels in order to save
  # memory before we save the observation in the batch trace.

  def translate_action(self, action, obs, **kwargs):
    threshold = action
    threshold += self.threshold_base    # [2,3,...]
    # print('Threshold is {}'.format(threshold))
    rc = obs.clabels[:,CLABEL_LBD] < float(threshold)
    a = rc.detach()
    num_learned = obs.ext_data
    locked = obs.clabels[num_learned[0]:num_learned[1],CLABEL_LOCKED].long().view(1,-1)
    final_action = torch.max(a.long(),locked).view(-1)
    obs.clabels = None

    return final_action

  def combine_actions(self, actions, **kwargs):    
    return actions

  def select_action(self, obs_batch, deterministic=False, log_threshold=False, **kwargs):    
    logits = self.threshold_logits
    if every_tick(20) or log_threshold:
      self.logger.info('State is:')
      self.logger.info(obs_batch.state)
      self.logger.info('Logits are: {}'.format(logits))    
    if deterministic:
      action = int(torch.argmax(logits.view(-1)))
      if every_tick(20) or log_threshold:
        self.logger.info('Threshold is {}'.format(action+self.threshold_base))
      return action, 0
    self.m = Categorical(logits=logits.view(-1))
    action=int(self.m.sample().detach())    
    if every_tick(20) or log_threshold:
      self.logger.info('Threshold is {}'.format(action+self.threshold_base))

    return action, float(self.m.entropy().detach())

  def get_logprobs(self, outputs, actions):    
    actions = self.settings.LongTensor(actions)
    logits = outputs
    probs = F.softmax(logits, dim=1)
    all_logprobs = safe_logprobs(probs)
    entropy = -(probs*all_logprobs).sum(dim=1)
    logprobs = all_logprobs.gather(1,actions.view(-1,1)).squeeze()
    return logprobs, entropy

  def compute_loss(self, transition_data, **kwargs):    
    returns = self.settings.FloatTensor([x.reward for x in transition_data])    
    actions = [x.action for x in transition_data]
    all_logits = self.threshold_logits.expand(len(transition_data),len(self.threshold_logits))
    logprobs, entropy = self.get_logprobs(all_logits, actions)
    adv_t = returns
    value_loss = 0.
    adv_t = (adv_t - adv_t.mean())
    pg_loss = (-adv_t*logprobs)
    loss = pg_loss - self.entropy_alpha*entropy
    if self.settings['use_sum']:
      total_loss = loss.sum()
    else:
      total_loss = loss.mean()

    return total_loss, all_logits
