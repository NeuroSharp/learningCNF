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
from qbf_data import *
# from batch_model import FactoredInnerIteration, GraphEmbedder
from qbf_model import QbfEncoder, QbfNewEncoder, QbfAttention
from settings import *

INVALID_BIAS = -1000
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


class Policy(nn.Module):
	def __init__(self, encoder=None, **kwargs):
		super(Policy, self).__init__()
		self.settings = kwargs['settings'] if 'settings' in kwargs.keys() else CnfSettings()				
		self.state_dim = self.settings['state_dim']
		self.embedding_dim = self.settings['embedding_dim']
		self.ground_dim = self.settings['ground_dim']
		self.policy_dim1 = self.settings['policy_dim1']
		self.policy_dim2 = self.settings['policy_dim2']		
		if self.settings['ac_baseline']:
			self.graph_embedder = GraphEmbedder(settings=self.settings)
			self.value_score = nn.Linear(self.state_dim+self.embedding_dim,1)
		if encoder:
			print('Bootstraping Policy from existing encoder')
			self.encoder = encoder
		else:
			self.encoder = QbfNewEncoder(**self.settings.hyperparameters)
			# self.encoder = QbfEncoder(**self.settings.hyperparameters)
		if self.settings['use_global_state']:
			self.linear1 = nn.Linear(self.state_dim+self.embedding_dim+self.ground_dim, self.policy_dim1)
		else:
			self.linear1 = nn.Linear(self.embedding_dim+self.ground_dim, self.policy_dim1)

		if self.settings['use_bn']:
			self.bn = nn.BatchNorm1d(self.embedding_dim)


		self.linear2 = nn.Linear(self.policy_dim1,self.policy_dim2)
		self.invalid_bias = nn.Parameter(self.settings.FloatTensor([self.settings['invalid_bias']]))
		self.action_score = nn.Linear(self.policy_dim2,1)
		self.activation = eval(self.settings['non_linearity'])		
		self.saved_log_probs = []
	
	# state is just a (batched) vector of fixed size state_dim which should be expanded. 
	# ground_embeddings are batch * max_vars * ground_embedding

	# cmat_net and cmat_pos are already "batched" into a single matrix

	def forward(self, obs, **kwargs):
		state = obs.state
		ground_embeddings = obs.ground
		cmat_pos = obs.cmat_pos		
		cmat_neg = obs.cmat_neg
		clabels = obs.clabels
		if self.settings['cuda']:
			cmat_pos, cmat_neg = cmat_pos.cuda(), cmat_neg.cuda()
			state, ground_embeddings = state.cuda(), ground_embeddings.cuda()
			if clabels:
				clabels = clabels.cuda()
		size = ground_embeddings.size()
		self.batch_size=size[0]
		if 'vs' in kwargs.keys():
			vs = kwargs['vs']		
		else:						
			rc = self.encoder(ground_embeddings, clabels, cmat_pos=cmat_pos, cmat_neg=cmat_neg, **kwargs)
			if self.settings['use_bn']:
				rc = self.bn(rc.view(-1,self.embedding_dim))
			vs = rc.view(self.batch_size,-1,self.embedding_dim)
		if self.settings['use_global_state']:
			reshaped_state = state.view(self.batch_size,1,self.state_dim).expand(self.batch_size,size[1],self.state_dim)
			inputs = torch.cat([reshaped_state, vs,ground_embeddings],dim=2).view(-1,self.state_dim+self.embedding_dim+self.ground_dim)
		else:
			inputs = torch.cat([vs,ground_embeddings],dim=2).view(-1,self.embedding_dim+self.ground_dim)			
		# inputs = torch.cat([reshaped_state, ground_embeddings],dim=2).view(-1,self.state_dim+self.ground_dim)
		# inputs = ground_embeddings.view(-1,self.ground_dim)
		outputs = self.action_score(self.activation(self.linear2(self.activation(self.linear1(inputs))))).view(self.batch_size,-1)
		# outputs = outputs-value		# Advantage
		# outputs = self.action_score(self.activation(self.linear1(inputs))).view(self.batch_size,-1)		

		if self.settings['pre_bias']:
			missing = (1-ground_embeddings[:,:,IDX_VAR_UNIVERSAL])*(1-ground_embeddings[:,:,IDX_VAR_EXISTENTIAL])
			valid = (1-(1-missing)*(1-ground_embeddings[:,:,IDX_VAR_DETERMINIZED]))*self.invalid_bias
			outputs = outputs + valid
		if self.settings['ac_baseline']:
			graph_embedding = self.graph_embedder(vs,batch_size=len(vs))
			value = self.value_score(torch.cat([state,graph_embedding],dim=1))
		else:
			value = None
		return outputs, value

		# rc = F.softmax(valid_outputs)
		# return rc



class DoublePolicy(nn.Module):
	def __init__(self, encoder=None, **kwargs):
		super(DoublePolicy, self).__init__()
		self.settings = kwargs['settings'] if 'settings' in kwargs.keys() else CnfSettings()				
		self.state_dim = self.settings['state_dim']
		self.embedding_dim = self.settings['embedding_dim']
		self.ground_dim = self.settings['ground_dim']
		self.policy_dim1 = self.settings['policy_dim1']
		self.policy_dim2 = self.settings['policy_dim2']		
		if self.settings['ac_baseline']:
			self.graph_embedder = GraphEmbedder(settings=self.settings)
			self.value_score = nn.Linear(self.state_dim+self.embedding_dim,1)
		if encoder:
			print('Bootstraping Policy from existing encoder')
			self.encoder = encoder
		else:
			self.encoder = QbfEncoder(**self.settings.hyperparameters)
		if self.settings['use_global_state']:
			self.linear1 = nn.Linear(self.state_dim+self.embedding_dim+self.ground_dim, self.policy_dim1)
		else:
			self.linear1 = nn.Linear(self.embedding_dim+self.ground_dim, self.policy_dim1)

		self.linear2 = nn.Linear(self.policy_dim1,self.policy_dim2)
		self.invalid_bias = nn.Parameter(self.settings.FloatTensor([self.settings['invalid_bias']]))
		self.action_score = nn.Linear(self.policy_dim2,2)
		if self.settings['leaky']:
			self.activation = F.leaky_relu
		else:
			self.activation = F.relu
		self.saved_log_probs = []
	
	# state is just a (batched) vector of fixed size state_dim which should be expanded. 
	# ground_embeddings are batch * max_vars * ground_embedding

	# cmat_net and cmat_pos are already "batched" into a single matrix

	def forward(self, obs, **kwargs):
		state = obs.state
		ground_embeddings = obs.ground
		cmat_pos = obs.cmat_pos		
		cmat_neg = obs.cmat_neg

		if self.settings['cuda']:
			cmat_pos, cmat_neg = cmat_pos.cuda(), cmat_neg.cuda()
			state, ground_embeddings = state.cuda(), ground_embeddings.cuda()			
		size = ground_embeddings.size()
		self.batch_size=size[0]
		if 'vs' in kwargs.keys():
			vs = kwargs['vs']		
		else:						
			rc = self.encoder(ground_embeddings, None, cmat_pos=cmat_pos, cmat_neg=cmat_neg, **kwargs)
			vs = rc.view(self.batch_size,-1,self.embedding_dim)
		if self.settings['use_global_state']:
			reshaped_state = state.view(self.batch_size,1,self.state_dim).expand(self.batch_size,size[1],self.state_dim)
			inputs = torch.cat([reshaped_state, vs,ground_embeddings],dim=2).view(-1,self.state_dim+self.embedding_dim+self.ground_dim)
		else:
			inputs = torch.cat([vs,ground_embeddings],dim=2).view(-1,self.embedding_dim+self.ground_dim)			

		outputs = self.action_score(self.activation(self.linear2(self.activation(self.linear1(inputs))))).view(self.batch_size,-1,2)
		Tracer()()
		
		
		if self.settings['pre_bias']:
			missing = (1-ground_embeddings[:,:,IDX_VAR_UNIVERSAL])*(1-ground_embeddings[:,:,IDX_VAR_EXISTENTIAL])
			valid = (1-(1-missing)*(1-ground_embeddings[:,:,IDX_VAR_DETERMINIZED]))*self.invalid_bias
			outputs = outputs + valid.unsqueeze(2).expand_as(outputs)
		if self.settings['ac_baseline']:
			graph_embedding = self.graph_embedder(vs,batch_size=len(vs))
			value = self.value_score(torch.cat([state,graph_embedding],dim=1))
		else:
			value = None
		return outputs, value

		# rc = F.softmax(valid_outputs)
		# return rc



class NewDoublePolicy(nn.Module):
	def __init__(self, encoder=None, **kwargs):
		super(NewDoublePolicy, self).__init__()
		self.settings = kwargs['settings'] if 'settings' in kwargs.keys() else CnfSettings()				
		self.state_dim = self.settings['state_dim']
		self.embedding_dim = self.settings['embedding_dim']
		self.vemb_dim = self.settings['vemb_dim']
		self.cemb_dim = self.settings['cemb_dim']
		self.vlabel_dim = self.settings['vlabel_dim']
		self.clabel_dim = self.settings['clabel_dim']
		self.final_embedding_dim = 2*self.settings['max_iters']*self.vemb_dim+self.vlabel_dim
		self.policy_dim1 = self.settings['policy_dim1']
		self.policy_dim2 = self.settings['policy_dim2']		
		if self.settings['ac_baseline']:
			self.graph_embedder = GraphEmbedder(settings=self.settings)
			self.value_score = nn.Linear(self.state_dim+self.vemb_dim,1)
		if encoder:
			print('Bootstraping Policy from existing encoder')
			self.encoder = encoder
		else:
			self.encoder = QbfNewEncoder(**kwargs)
		if self.settings['use_global_state']:
			self.linear1 = nn.Linear(self.state_dim+self.final_embedding_dim, self.policy_dim1)
		else:
			self.linear1 = nn.Linear(self.final_embedding_dim, self.policy_dim1)

		self.linear2 = nn.Linear(self.policy_dim1,self.policy_dim2)
		self.invalid_bias = nn.Parameter(self.settings.FloatTensor([self.settings['invalid_bias']]))
		self.action_score = nn.Linear(self.policy_dim2,1)
		if self.settings['leaky']:
			self.activation = F.leaky_relu
		else:
			self.activation = eval(self.settings['non_linearity'])
		self.saved_log_probs = []
	
	# state is just a (batched) vector of fixed size state_dim which should be expanded. 
	# ground_embeddings are batch * max_vars * ground_embedding

	# cmat_net and cmat_pos are already "batched" into a single matrix

	def forward2(self, obs, **kwargs):
		state = obs.state
		ground_embeddings = obs.ground
		clabels = obs.clabels
		cmat_pos = obs.cmat_pos		
		cmat_neg = obs.cmat_neg

		if self.settings['cuda']:
			cmat_pos, cmat_neg = cmat_pos.cuda(), cmat_neg.cuda()
			state, ground_embeddings = state.cuda(), ground_embeddings.cuda()			
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

		if self.settings['use_global_state']:
			# if self.batch_size > 1:
			# 	Tracer()()
			a = state.unsqueeze(0).expand(2,*state.size()).contiguous().view(2*self.batch_size,1,self.state_dim)
			reshaped_state = a.expand(2*self.batch_size,size[1],self.state_dim) # add the maxvars dimention
			inputs = torch.cat([reshaped_state, vs],dim=2).view(-1,self.state_dim+self.final_embedding_dim)
		else:
			inputs = vs.view(-1,self.final_embedding_dim)

		outputs = self.action_score(self.activation(self.linear2(self.activation(self.linear1(inputs)))))
		outputs = outputs.view(2,self.batch_size,-1)
		outputs = outputs.transpose(2,0).transpose(1,0)			# batch x numvars x pos-neg
		# Tracer()()

		if self.settings['pre_bias']:
			missing = (1-ground_embeddings[:,:,IDX_VAR_UNIVERSAL])*(1-ground_embeddings[:,:,IDX_VAR_EXISTENTIAL])
			valid = (1-(1-missing)*(1-ground_embeddings[:,:,IDX_VAR_DETERMINIZED]))*self.invalid_bias
			outputs = outputs + valid.unsqueeze(2).expand_as(outputs)
		if self.settings['ac_baseline']:
			graph_embedding = self.graph_embedder(vs,batch_size=len(vs))
			value = self.value_score(torch.cat([state,graph_embedding],dim=1))
		else:
			value = None
		return outputs, value

	def forward(self, obs, packed=False, **kwargs):
		if not packed:
			return self.forward2(obs,**kwargs)
		state = obs.state
		ground_embeddings = obs.ground
		clabels = obs.clabels
		cmat_pos = obs.cmat_pos		
		cmat_neg = obs.cmat_neg

		if self.settings['cuda']:
			cmat_pos, cmat_neg = cmat_pos.cuda(), cmat_neg.cuda()
			state, ground_embeddings = state.cuda(), ground_embeddings.cuda()			
			if clabels is not None:
				clabels = clabels.cuda()

		size = ground_embeddings.size()
		self.batch_size=size[0]
		if 'vs' in kwargs.keys():
			vs = kwargs['vs']		
		else:						
			vs_pos, vs_neg = self.encoder(ground_embeddings, clabels, cmat_pos=cmat_pos, cmat_neg=cmat_neg, **kwargs)
			vs = torch.cat([vs_pos,vs_neg])
			if 'do_debug' in kwargs:
				Tracer()()
				

		if self.settings['use_global_state']:
			reshaped_state = []
			for i in range(len(obs.pack_indices[1])-1):
				reshaped_state.append(state[i].expand(obs.pack_indices[1][i+1]-obs.pack_indices[1][i],self.state_dim))
			reshaped_state = torch.cat(reshaped_state)
			reshaped_state = reshaped_state.unsqueeze(0).expand(2,*reshaped_state.size()).contiguous().view(-1,self.state_dim)
			inputs = torch.cat([reshaped_state, vs],dim=1)
		else:
			inputs = vs

		outputs = self.action_score(self.activation(self.linear2(self.activation(self.linear1(inputs)))))
		outputs = outputs.view(2,self.batch_size).transpose(1,0)	# batch x pos-neg

		value = None
		return outputs, value


class AttnPolicy(nn.Module):
	def __init__(self, encoder=None, **kwargs):
		super(AttnPolicy, self).__init__()
		self.settings = kwargs['settings'] if 'settings' in kwargs.keys() else CnfSettings()				
		self.state_dim = self.settings['state_dim']
		self.embedding_dim = self.settings['embedding_dim']
		self.vemb_dim = self.settings['vemb_dim']
		self.cemb_dim = self.settings['cemb_dim']
		self.vlabel_dim = self.settings['vlabel_dim']
		self.clabel_dim = self.settings['clabel_dim']
		self.final_embedding_dim = 2*self.settings['max_iters']*self.vemb_dim+self.vlabel_dim
		self.policy_dim1 = self.settings['policy_dim1']
		self.policy_dim2 = self.settings['policy_dim2']		
		self.state_bn = self.settings['state_bn']		
		self.hidden_dim = 50
		if self.settings['ac_baseline']:			
			self.value_attn = QbfAttention(self.final_embedding_dim, n_heads=20, settings=self.settings)
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
		if self.settings['use_global_state']:
			self.linear1 = nn.Linear(self.state_dim+self.final_embedding_dim, self.policy_dim1)
		else:
			self.linear1 = nn.Linear(self.final_embedding_dim, self.policy_dim1)

		if self.policy_dim2:
			self.linear2 = nn.Linear(self.policy_dim1,self.policy_dim2)
			self.action_score = nn.Linear(self.policy_dim2,1)
		else:
			self.action_score = nn.Linear(self.policy_dim1,1)
		self.invalid_bias = nn.Parameter(self.settings.FloatTensor([self.settings['invalid_bias']]))
		if self.state_bn:
			self.state_bn = nn.BatchNorm1d(self.state_dim)
		if self.settings['leaky']:
			self.activation = F.leaky_relu
		else:
			self.activation = eval(self.settings['non_linearity'])
		self.saved_log_probs = []
	
	# state is just a (batched) vector of fixed size state_dim which should be expanded. 
	# ground_embeddings are batch * max_vars * ground_embedding

	# cmat_net and cmat_pos are already "batched" into a single matrix

	def forward(self, obs, **kwargs):
		state = obs.state
		ground_embeddings = obs.ground
		clabels = obs.clabels
		cmat_pos = obs.cmat_pos		
		cmat_neg = obs.cmat_neg
		aux_losses = []

		if self.settings['cuda']:
			cmat_pos, cmat_neg = cmat_pos.cuda(), cmat_neg.cuda()
			state, ground_embeddings = state.cuda(), ground_embeddings.cuda()			
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
		
		if self.state_bn:
			state = self.state_bn(state)

		if self.settings['use_global_state']:
			# if self.batch_size > 1:
			# 	Tracer()()
			a = state.unsqueeze(0).expand(2,*state.size()).contiguous().view(2*self.batch_size,1,self.state_dim)
			reshaped_state = a.expand(2*self.batch_size,size[1],self.state_dim) # add the maxvars dimention
			inputs = torch.cat([reshaped_state, vs],dim=2).view(-1,self.state_dim+self.final_embedding_dim)
		else:
			inputs = vs.view(-1,self.final_embedding_dim)

		if self.policy_dim2:			
			outputs = self.action_score(self.activation(self.linear2(self.activation(self.linear1(inputs)))))
		else:
			outputs = self.action_score(self.activation(self.linear1(inputs)))
		outputs = outputs.view(2,self.batch_size,-1)
		outputs = outputs.transpose(2,0).transpose(1,0)			# batch x numvars x pos-neg
		# Tracer()()

		if self.settings['pre_bias']:
			missing = (1-ground_embeddings[:,:,IDX_VAR_UNIVERSAL])*(1-ground_embeddings[:,:,IDX_VAR_EXISTENTIAL])
			valid = (1-(1-missing)*(1-ground_embeddings[:,:,IDX_VAR_DETERMINIZED]))*self.invalid_bias
			outputs = outputs + valid.unsqueeze(2).expand_as(outputs)
		if self.settings['ac_baseline'] and self.batch_size > 1:
			embs = vs.view(2,self.batch_size,-1,self.final_embedding_dim).transpose(0,1).contiguous().view(self.batch_size,-1,self.final_embedding_dim)
			graph_embedding, value_aux_loss = self.value_attn(state,embs,attn_mask=obs.vmask)
			aux_losses.append(value_aux_loss)
			if self.settings['use_state_in_vn']:
				val_inp = torch.cat([state,graph_embedding.view(self.batch_size,-1)],dim=1)
			else:
				val_inp = graph_embedding.view(self.batch_size,-1)
			value = self.value_score2(self.activation(self.value_score1(val_inp)))
		else:
			value = None
		return outputs, value, vs, aux_losses
