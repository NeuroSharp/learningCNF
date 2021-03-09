import ipdb
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import init as nn_init
import torch.nn.functional as F
import torch.optim as optim
import utils
import numpy as np
from collections import namedtuple
from IPython.core.debugger import Tracer
import cadet_utils
from vbn import *
from qbf_data import *
from qbf_model import *
from settings import *
from policy_base import *
from rl_utils import *
from ray.rllib.policy.sample_batch import SampleBatch
  
class CadetModel(RLLibModel):
  def __init__(self, *args, **kwargs):
    super(CadetModel, self).__init__(*args, **kwargs)
    encoder = None
    self.final_embedding_dim = 2*self.max_iters*self.vemb_dim+self.vlabel_dim
    self.hidden_dim = 50
    self.snorm_window = 5000
    if self.settings['ac_baseline']:      
      self.value_attn = QbfFlattenedAttention(self.final_embedding_dim, n_heads=10, settings=self.settings)
      if self.settings['use_state_in_vn']:
        self.value_score1 = nn.Linear(self.state_dim+self.value_attn.n_heads*self.final_embedding_dim,self.hidden_dim)
      else:
        self.value_score1 = nn.Linear(self.value_attn.n_heads*self.final_embedding_dim,self.hidden_dim)
      self.value_score2 = nn.Linear(self.hidden_dim,1)        
    if encoder:
      print('Bootstraping Policy from existing encoder')
      self.encoder = encoder
    else:
      self.encoder = QbfNewEncoder(**kwargs)
      # self.encoder = QbfSimpleEncoder(**kwargs)
    if self.settings['use_global_state']:
      self.linear1 = nn.Linear(self.state_dim+self.final_embedding_dim, self.policy_dim1)
    else:
      self.linear1 = nn.Linear(self.final_embedding_dim, self.policy_dim1)

    if self.policy_dim2:
      self.linear2 = nn.Linear(self.policy_dim1,self.policy_dim2)
      self.action_score = nn.Linear(self.policy_dim2,1)
    else:
      self.action_score = nn.Linear(self.policy_dim1,1)
    if self.state_bn:
      self.state_vbn = MovingAverageVBN((self.snorm_window,self.state_dim))
    self.use_global_state = self.settings['use_global_state']
    self.activation = eval(self.settings['policy_non_linearity'])


  def from_batch(self, train_batch, is_training=True):
    """Convenience function that calls this model with a tensor batch.

    All this does is unpack the tensor batch to call this model with the
    right input dict, state, and seq len arguments.
    """
    def obs_from_input_dict(input_dict):
      z = list(input_dict.items())
      dense_obs = [undensify_obs(DenseState(*x)) for x in list(z[3][1])]
      return dense_obs

    collated_batch = collate_observations(obs_from_input_dict(train_batch),settings=self.settings)
    input_dict = {
      "collated_obs": collated_batch,
      "is_training": is_training,
    }
    if SampleBatch.PREV_ACTIONS in train_batch:
      input_dict["prev_actions"] = train_batch[SampleBatch.PREV_ACTIONS]
    if SampleBatch.PREV_REWARDS in train_batch:
      input_dict["prev_rewards"] = train_batch[SampleBatch.PREV_REWARDS]
    states = []
    i = 0
    while "state_in_{}".format(i) in train_batch:
      states.append(train_batch["state_in_{}".format(i)])
      i += 1
    return self.__call__(input_dict, states, train_batch.get("seq_lens"))


  
  # state is just a (batched) vector of fixed size state_dim which should be expanded. 
  # ground_embeddings are batch * max_vars * ground_embedding

  # cmat_net and cmat_pos are already "batched" into a single matrix
  def forward(self, input_dict, state, seq_lens, **kwargs):
    def obs_from_input_dict(input_dict):
      z = list(input_dict.items())
      z1 = list(z[0][1][0])
      return undensify_obs(DenseState(*z1))
    if 'collated_obs' in input_dict.keys():       # This is in batch/learning mode
      obs = input_dict['collated_obs']
    else:
      obs = obs_from_input_dict(input_dict)       # This is an experience rollout
    global_state = obs.state
    ground_embeddings = obs.ground
    clabels = obs.clabels
    self._value_out = torch.zeros(1).expand(len(ground_embeddings))
    cmat_pos, cmat_neg = split_sparse_adjacency(obs.cmat.float())
    
    aux_losses = []

    if self.settings['cuda']:
      cmat_pos, cmat_neg = cmat_pos.cuda(), cmat_neg.cuda()
      global_state, ground_embeddings = global_state.cuda(), ground_embeddings.cuda()     
      if clabels is not None:
        clabels = clabels.cuda()

    size = ground_embeddings.size()
    self.batch_size=size[0]
    if 'vs' in kwargs.keys():
      vs = kwargs['vs']   
    else:
      pos_vars, neg_vars = self.encoder(ground_embeddings.view(-1,self.vlabel_dim), clabels.view(-1,self.clabel_dim), cmat_pos=cmat_pos, cmat_neg=cmat_neg, **kwargs)
      vs_pos = pos_vars.view(self.batch_size,-1,self.final_embedding_dim)
      vs_neg = neg_vars.view(self.batch_size,-1,self.final_embedding_dim)
      vs = torch.cat([vs_pos,vs_neg])
      if 'do_debug' in kwargs:
        Tracer()()
    

    if self.use_global_state:
      if self.state_bn:
        global_state = self.state_vbn(global_state)
      a = global_state.unsqueeze(0).expand(2,*global_state.size()).contiguous().view(2*self.batch_size,1,self.state_dim)
      reshaped_state = a.expand(2*self.batch_size,size[1],self.state_dim) # add the maxvars dimention
      inputs = torch.cat([reshaped_state, vs],dim=2).view(-1,self.state_dim+self.final_embedding_dim)
    else:
      inputs = vs.view(-1,self.final_embedding_dim)

    if self.policy_dim2:      
      outputs = self.action_score(self.activation(self.linear2(self.activation(self.linear1(inputs)))))
    else:
      outputs = self.action_score(self.activation(self.linear1(inputs)))
    # Tracer()()
    outputs = outputs.view(2,self.batch_size,-1)
    outputs = outputs.transpose(2,0).transpose(1,0)     # batch x numvars x pos-neg
    logits = outputs.contiguous().view(self.batch_size,-1)
    if self.settings['ac_baseline'] and self.batch_size > 1:
      embs = vs.view(2,self.batch_size,-1,self.final_embedding_dim).transpose(0,1).contiguous().view(self.batch_size,-1,self.final_embedding_dim)
      mask = torch.cat([obs.vmask]*2,dim=1)
      graph_embedding, value_aux_loss = self.value_attn(global_state,embs,attn_mask=mask)
      aux_losses.append(value_aux_loss)
      if self.settings['use_state_in_vn']:
        val_inp = torch.cat([global_state,graph_embedding.view(self.batch_size,-1)],dim=1)
      else:
        val_inp = graph_embedding.view(self.batch_size,-1)
      value = self.value_score2(self.activation(self.value_score1(val_inp)))
    else:
      value = None
    allowed_actions = self.get_allowed_actions(obs).int().float()
    inf_mask = torch.max(allowed_actions.log(),torch.Tensor([torch.finfo().min]))
    # print('logits shape is {}'.format(logits.shape))
    # print(allowed_actions)
    return logits + inf_mask, []

  def get_allowed_actions(self, obs, **kwargs):
    rc = cadet_utils.get_allowed_actions(obs,**kwargs)
    s = rc.shape
    rc = rc.unsqueeze(2).expand(*s,2).contiguous()
    rc = rc.view(s[0],-1)
    return rc

  def combine_actions(self, actions, **kwargs):
    return self.settings.LongTensor(actions)

  def select_action(self, obs_batch, **kwargs):
    [logits], *_ = self.forward(collate_observations([obs_batch]))
    allowed_actions = self.get_allowed_actions(obs_batch)[0]
    allowed_idx = self.settings.cudaize_var(torch.from_numpy(np.where(allowed_actions.numpy())[0]))
    l = logits[allowed_idx]
    probs = F.softmax(l.contiguous().view(1,-1),dim=1)
    dist = probs.data.cpu().numpy()[0]
    choices = range(len(dist))
    aux_action = np.random.choice(choices, p=dist)
    action = allowed_idx[aux_action]
    return action, 0

  def compute_loss(self, transition_data, **kwargs):
    _, _, _, rewards, *_ = zip(*transition_data)
    collated_batch = collate_transitions(transition_data,settings=self.settings)
    collated_batch.state = cudaize_obs(collated_batch.state)
    logits, values, _, aux_losses = self.forward(collated_batch.state, prev_obs=collated_batch.prev_obs)
    allowed_actions = Variable(self.get_allowed_actions(collated_batch.state))
    if self.settings['cuda']:
      allowed_actions = allowed_actions.cuda()
    # unpacked_logits = unpack_logits(logits, collated_batch.state.pack_indices[1])
    effective_bs = len(logits)    

    if self.settings['masked_softmax']:
      allowed_mask = allowed_actions.float()      
      probs, debug_probs = masked_softmax2d(logits,allowed_mask)
    else:
      probs = F.softmax(logits, dim=1)
    all_logprobs = safe_logprobs(probs)
    if self.settings['disallowed_aux']:        # Disallowed actions are possible, so we add auxilliary loss
      aux_probs = F.softmax(logits,dim=1)
      disallowed_actions = Variable(allowed_actions.data^1).float()      
      disallowed_mass = (aux_probs*disallowed_actions).sum(1)
      disallowed_loss = disallowed_mass.mean()
      # print('Disallowed loss is {}'.format(disallowed_loss))

    returns = self.settings.FloatTensor(rewards)
    if self.settings['ac_baseline']:
      adv_t = returns - values.squeeze().data      
      value_loss = mse_loss(values.squeeze(), Variable(returns))    
      print('Value loss is {}'.format(value_loss.data.numpy()))
      print('Value Auxilliary loss is {}'.format(sum(aux_losses).data.numpy()))
      if i>0 and i % 60 == 0:
        vecs = {'returns': returns.numpy(), 'values': values.squeeze().data.numpy()}        
        pprint_vectors(vecs)
    else:
      adv_t = returns
      value_loss = 0.
    actions = collated_batch.action    
    try:
      logprobs = all_logprobs.gather(1,Variable(actions).view(-1,1)).squeeze()
    except:
      Tracer()()
    entropies = (-probs*all_logprobs).sum(1)    
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + float(np.finfo(np.float32).eps))
    # Tracer()()
    if self.settings['normalize_episodes']:
      episodes_weights = normalize_weights(collated_batch.formula.cpu().numpy())
      adv_t = adv_t*self.settings.FloatTensor(episodes_weights)    
    if self.settings['use_sum']:
      pg_loss = (-Variable(adv_t)*logprobs).sum()
    else:
      pg_loss = (-Variable(adv_t)*logprobs).mean()
    total_aux_loss = sum(aux_losses) if aux_losses else 0.    
    loss = pg_loss + self.lambda_value*value_loss + self.lambda_aux*total_aux_loss

    if self.use_global_state and self.state_bn:
      self.state_vbn.recompute_moments(collated_batch.state.state.detach())

    return loss, logits

  def value_function(self):
    return self._value_out.view(-1)
