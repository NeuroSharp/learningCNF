import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import init as nn_init
import torch.nn.functional as F
import torch.optim as optim
import utils
import numpy as np
from IPython.core.debugger import Tracer
import pdb
from settings import *

class GroundCombinator(nn.Module):
	def __init__(self, ground_dim, embedding_dim, hidden_dim=20):
		super(GroundCombinator, self).__init__()
		self.layer1 = nn.Linear(embedding_dim+ground_dim,hidden_dim)
		self.layer2 = nn.Linear(hidden_dim,embedding_dim)

	def forward(self, ground,state):
		return self.layer2(F.relu(self.layer1(torch.cat([ground.float(),state],dim=1))))
		
class DummyGroundCombinator(nn.Module):
	def __init__(self, *args, **kwargs):
		super(DummyGroundCombinator, self).__init__()        		

	def forward(self, ground,state):
		return state
		

class SymmetricSumCombine(nn.Module):
	def __init__(self, embedding_dim):
		super(SymmetricSumCombine, self).__init__()        
		self.layer1 = nn.Linear(embedding_dim,embedding_dim)
		self.layer2 = nn.Linear(embedding_dim,embedding_dim)

	def forward(self, inputs):		
		out = [utils.normalize(F.sigmoid(self.layer1(x)) + self.layer2(x)) for x in inputs]
		return torch.stack(out,dim=2).sum(dim=2).view(out[0].size())


# class VariableIteration(nn.Module):
# 	def __init__(self, embedding_dim, max_clauses, max_variables, num_ground_variables, max_iters):


class BatchInnerIteration(nn.Module):
	def __init__(self, get_ground_embeddings, embedding_dim, max_variables, num_ground_variables, permute=True, **kwargs):
		super(BatchInnerIteration, self).__init__()        
		self.settings = kwargs['settings'] if 'settings' in kwargs.keys() else CnfSettings()
		self.comb_type = eval(self.settings['combinator_type'])
		self.ground_comb_type = eval(self.settings['ground_combinator_type'])
		self.get_ground_embeddings = get_ground_embeddings		
		self.embedding_dim = embedding_dim
		self.max_variables = max_variables		
		self.permute = permute
		self.num_ground_variables = num_ground_variables	
		self.extra_embedding = nn.Embedding(1, embedding_dim, max_norm=1.)				
		self.ground_combiner = self.ground_comb_type(self.settings['ground_dim'],embedding_dim)
		self.cuda = kwargs['cuda']		
		self.vb = nn.Parameter(self.settings.FloatTensor(self.embedding_dim,1))
		self.cb = nn.Parameter(self.settings.FloatTensor(self.embedding_dim,1))
		nn_init.normal(self.vb)
		nn_init.normal(self.cb)
		# self.var_bias = torch.stack([torch.cat([vb]*self.settings['max_variables'])]*self.settings['batch_size'])
		# self.clause_bias = torch.stack([torch.cat([cb]*self.settings['max_clauses'])]*self.settings['batch_size'])
				
		self.W_z = nn.Linear(self.embedding_dim,self.embedding_dim,bias=False)
		self.U_z = nn.Linear(self.embedding_dim,self.embedding_dim,bias=self.settings['gru_bias'])
		self.W_r = nn.Linear(self.embedding_dim,self.embedding_dim,bias=False)
		self.U_r = nn.Linear(self.embedding_dim,self.embedding_dim,bias=self.settings['gru_bias'])
		self.W = nn.Linear(self.embedding_dim,self.embedding_dim,bias=False)
		self.U = nn.Linear(self.embedding_dim,self.embedding_dim,bias=self.settings['gru_bias'])
		self.non_linearity = eval(self.settings['non_linearity'])
		self.re_init()

	def re_init(self):
		self.var_bias = torch.cat([self.vb]*self.settings['max_variables'])
		self.clause_bias = torch.cat([self.cb]*self.settings['max_clauses'])
		

	def gru(self, av, prev_emb):					
		z = F.sigmoid(self.W_z(av) + self.U_z(prev_emb))
		r = F.sigmoid(self.W_r(av) + self.U_r(prev_emb))
		h_tilda = F.tanh(self.W(av) + self.U(r*prev_emb))
		h = (1-z) * prev_emb + z*h_tilda
		# g = h.view(2,-1,self.embedding_dim)
		# if (g[0]==g[1]).data.all():
		# 	print('Same embedding in gru. wtf?')
		# 	# pdb.set_trace()
		return h

	def forward(self, variables, v_mat, c_mat, ground_vars=None, **kwargs):
		try:			
			c_emb = F.tanh(torch.bmm(c_mat,variables) + self.clause_bias)
			return c_emb
			if (c_emb[0] == c_emb[1]).data.all():
				print('c_emb identical!')
			# c_emb = utils.normalize(c_emb.view(-1,self.embedding_dim)).view(self.settings['batch_size'],-1,1)
		except:
			pdb.set_trace()
		v_emb = F.tanh(torch.bmm(v_mat,c_emb) + self.var_bias)
		# v_emb = utils.normalize(v_emb.view(-1,self.embedding_dim)).view(self.settings['batch_size'],-1,1)
		v_emb = self.ground_combiner(ground_vars,v_emb.view(-1,self.embedding_dim))
		new_vars = self.gru(v_emb, variables.view(-1,self.embedding_dim))
		rc = new_vars.view(-1,self.max_variables*self.embedding_dim,1)
		if (rc != rc).data.any():			# We won't stand for NaN
			print('NaN in our tensors!!')
			pdb.set_trace()
		if (rc[0] == rc[1]).data.all():
			print('Same embedding. wtf?')
			# pdb.set_trace()
		return rc


class FactoredInnerIteration(nn.Module):
	def __init__(self, **kwargs):
		super(FactoredInnerIteration, self).__init__()        
		self.settings = kwargs['settings'] if 'settings' in kwargs.keys() else CnfSettings()
		self.ground_comb_type = eval(self.settings['ground_combinator_type'])
		self.non_linearity = eval(self.settings['non_linearity'])
		self.ground_dim = self.settings['ground_dim']
		self.embedding_dim = self.settings['embedding_dim']		
		self.ground_combiner = self.ground_comb_type(self.settings['ground_dim'],self.embedding_dim)
		self.cuda = self.settings['cuda']		
		self.vb = nn.Parameter(self.settings.FloatTensor(self.embedding_dim,1))
		self.cb = nn.Parameter(self.settings.FloatTensor(self.embedding_dim,1))
		nn_init.normal(self.vb)
		nn_init.normal(self.cb)

				
		self.W_z = nn.Linear(self.embedding_dim,self.embedding_dim,bias=False)
		self.U_z = nn.Linear(self.embedding_dim,self.embedding_dim,bias=self.settings['gru_bias'])
		self.W_r = nn.Linear(self.embedding_dim,self.embedding_dim,bias=False)
		self.U_r = nn.Linear(self.embedding_dim,self.embedding_dim,bias=self.settings['gru_bias'])
		self.W = nn.Linear(self.embedding_dim,self.embedding_dim,bias=False)
		self.U = nn.Linear(self.embedding_dim,self.embedding_dim,bias=self.settings['gru_bias'])

		self.re_init()

	def re_init(self):
		self.var_bias = torch.cat([self.vb]*self.settings['max_variables'])
		self.clause_bias = torch.cat([self.cb]*self.settings['max_clauses'])
		

	def gru(self, av, prev_emb):
		z = F.sigmoid(self.W_z(av) + self.U_z(prev_emb))
		r = F.sigmoid(self.W_r(av) + self.U_r(prev_emb))
		h_tilda = F.tanh(self.W(av) + self.U(r*prev_emb))
		h = (1-z) * prev_emb + z*h_tilda
		# g = h.view(2,-1,self.embedding_dim)
		# if (g[0]==g[1]).data.all():
		# 	print('Same embedding in gru. wtf?')
		# 	# pdb.set_trace()
		return h




	def forward(self, variables, v_mat, c_mat, ground_vars=None, v_block=None, c_block=None, **kwargs):		
		if 'old_forward' in kwargs and kwargs['old_forward']:
			return self.forward2(variables,v_mat,c_mat,ground_vars=ground_vars, **kwargs)
		assert(v_block is not None and c_block is not None)
		bsize = kwargs['batch_size'] if 'batch_size' in kwargs else self.settings['batch_size']
		self.max_variables = kwargs['max_variables'] if 'max_variables' in kwargs else self.settings['max_variables']
		org_size = variables.size()
		v = variables.view(-1,self.embedding_dim).t()
		size = v.size(1)	# batch x num_vars
		use_neg = self.settings['negate_type'] != 'minus'
		if use_neg:
			# ipdb.set_trace()
			pos_vars, neg_vars = torch.bmm(c_block,v.expand(2,self.embedding_dim,size)).transpose(1,2)			
			if self.settings['sparse'] and 'cmat_pos' in kwargs and 'cmat_neg' in kwargs:
				pos_cmat = kwargs['cmat_pos']
				neg_cmat = kwargs['cmat_neg']
				# c = torch.mm(pos_cmat,pos_vars) + torch.mm(neg_cmat,neg_vars)
				c = torch.mm(pos_cmat,pos_vars) + torch.matmul(neg_cmat,neg_vars)
				c = c.view(bsize,-1,self.embedding_dim)				
			else:				
				pos_cmat = c_mat.clamp(0,1).float()
				neg_cmat = -c_mat.clamp(-1,0).float()
				y1 = pos_vars.contiguous().view(org_size[0],-1,self.embedding_dim)
				y2 = neg_vars.contiguous().view(org_size[0],-1,self.embedding_dim)	
				c = torch.bmm(pos_cmat,y1) + torch.bmm(neg_cmat,y2)									
		else:
			vars_all = torch.mm(c_block[0],v).t().contiguous().view(org_size[0],-1,self.embedding_dim)
			c = torch.bmm(c_mat.float(),vars_all)	

		c = self.non_linearity(c + self.cb.squeeze())		
		cv = c.view(-1,self.embedding_dim).t()		
		size = cv.size(1)
		if use_neg:
			pos_cvars, neg_cvars = torch.bmm(v_block,cv.expand(2,self.embedding_dim,size)).transpose(1,2)
			if self.settings['sparse'] and 'cmat_pos' in kwargs and 'cmat_neg' in kwargs:
				pos_vmat = kwargs['cmat_pos'].t()
				neg_vmat = kwargs['cmat_neg'].t()
				nv = torch.mm(pos_vmat,pos_cvars) + torch.mm(neg_vmat,neg_cvars)
				nv = nv.view(bsize,-1,self.embedding_dim)
			else:	
				pos_vmat = v_mat.clamp(0,1).float()
				neg_vmat = -v_mat.clamp(-1,0).float()
				y1 = pos_cvars.contiguous().view(org_size[0],-1,self.embedding_dim)
				y2 = neg_cvars.contiguous().view(org_size[0],-1,self.embedding_dim)
				nv = torch.bmm(pos_vmat,y1) + torch.bmm(neg_vmat,y2)
		else:
			vars_all = torch.mm(v_block[0],cv).t().contiguous().view(org_size[0],-1,self.embedding_dim)
			nv = torch.bmm(v_mat.float(),vars_all)	
			
		v_emb = self.non_linearity(nv + self.vb.squeeze())		
		v_emb = self.ground_combiner(ground_vars.view(-1,self.ground_dim),v_emb.view(-1,self.embedding_dim))		
		if self.settings['use_gru']:
			new_vars = self.gru(v_emb, variables.view(-1,self.embedding_dim))	
		else:
			new_vars = v_emb
		rc = new_vars.view(-1,self.max_variables*self.embedding_dim,1)
		if (rc != rc).data.any():			# We won't stand for NaN
			print('NaN in our tensors!!')
			pdb.set_trace()
		# if (rc[0] == rc[1]).data.all():
		# 	print('Same embedding. wtf?')
			# pdb.set_trace()
		return rc

	def forward2(self, variables, v_mat, c_mat, ground_vars=None, **kwargs):
		try:			
			c_emb = F.tanh(torch.bmm(c_mat,variables) + self.clause_bias)
			if (c_emb[0] == c_emb[1]).data.all():
				print('c_emb identical!')
			# c_emb = utils.normalize(c_emb.view(-1,self.embedding_dim)).view(self.settings['batch_size'],-1,1)
		except:
			pdb.set_trace()
		v_emb = F.tanh(torch.bmm(v_mat,c_emb) + self.var_bias)
		# v_emb = utils.normalize(v_emb.view(-1,self.embedding_dim)).view(self.settings['batch_size'],-1,1)
		v_emb = self.ground_combiner(ground_vars,v_emb.view(-1,self.embedding_dim))
		new_vars = self.gru(v_emb, variables.view(-1,self.embedding_dim))
		rc = new_vars.view(-1,self.max_variables*self.embedding_dim,1)
		if (rc != rc).data.any():			# We won't stand for NaN
			print('NaN in our tensors!!')
			pdb.set_trace()
		if (rc[0] == rc[1]).data.all():
			print('Same embedding. wtf?')
			# pdb.set_trace()
		return rc



class BatchEncoder(nn.Module):
	def __init__(self, embedding_dim, num_ground_variables, max_iters, **kwargs):
		super(BatchEncoder, self).__init__() 
		self.settings = kwargs['settings'] if 'settings' in kwargs.keys() else CnfSettings()
		self.debug = False
		self.ground_dim = self.settings['ground_dim']
		self.batch_size = self.settings['batch_size']
		self.max_variables = self.settings['max_variables']
		self.embedding_dim = embedding_dim		
		self.expand_dim_const = Variable(self.settings.zeros([self.max_variables,self.embedding_dim - self.ground_dim - 1 ]), requires_grad=False)
		self.max_iters = max_iters		
		self.num_ground_variables = num_ground_variables		
		self.use_ground = self.settings['use_ground']
		self.moving_ground = self.settings['moving_ground']

		# 'moving_ground' will make num_ground_variables embeddings, plus only one for all the non-ground variables.

		if self.moving_ground:
			self.embedding = nn.Embedding(num_ground_variables+1, self.ground_dim, max_norm=1., norm_type=2)
		else:

		# If not moving ground, then just a one-hot representation for different ground variables plus last one for non-ground vars.

			base_annotations = Variable(torch.eye(num_ground_variables+1),requires_grad=False)
			if self.settings['cuda']:
				base_annotations = base_annotations.cuda()
			if self.use_ground:
				dup_annotations = torch.cat([base_annotations[-1].unsqueeze(0)] * (self.max_variables - num_ground_variables -1))
				exp_annotations = torch.cat([base_annotations,dup_annotations])				
			else:
				pdb.set_trace()
				exp_annotations = torch.cat([base_annotations[-1].unsqueeze(0)]*self.max_variables)				
			self.ground_annotations = self.expand_ground_to_state(exp_annotations)

		self.inner_iteration = FactoredInnerIteration(**kwargs)
		# self.inner_iteration2 = BatchInnerIteration(self.get_ground_embeddings, embedding_dim, num_ground_variables=num_ground_variables, **kwargs)
	
		self.zero_block = Variable(self.settings.zeros([1,self.embedding_dim,self.embedding_dim]), requires_grad=False)
		self.forward_pos_neg = nn.Parameter(self.settings.FloatTensor(2,self.embedding_dim,self.embedding_dim))
		nn_init.normal(self.forward_pos_neg)		
		self.backwards_pos_neg = nn.Parameter(self.settings.FloatTensor(2,self.embedding_dim,self.embedding_dim))		
		nn_init.normal(self.backwards_pos_neg)

		if self.use_ground:
			self.var_indices = Variable(torch.arange(0,self.max_variables).long().clamp(0,self.num_ground_variables),requires_grad=False)
		else:
			self.var_indices = Variable(self.settings.LongTensor([self.num_ground_variables]*self.max_variables),requires_grad=False)


		
# input is one training sample (a formula), we'll permute it a bit at every iteration and possibly split to create a batch

	def fix_annotations(self):
		if self.settings['cuda']:
			self.ground_annotations = self.ground_annotations.cuda()
			self.zero_block = self.zero_block.cuda()
			self.forward_block = self.forward_block.cuda()
			self.backwards_block = self.backwards_block.cuda()
		else:
			self.ground_annotations = self.ground_annotations.cpu()
			self.zero_block = self.zero_block.cpu()
			self.forward_block = self.forward_block.cpu()
			self.backwards_block = self.backwards_block.cpu()


	def expand_ground_to_state(self,v):
		return torch.cat([v,self.expand_dim_const],dim=1)

	def get_ground_embeddings(self):
		if self.moving_ground:
			embs = self.expand_ground_to_state(self.embedding(self.var_indices))
		else:
			embs = self.ground_annotations
		return embs.view(1,-1).transpose(0,1)

	def get_block_matrix(self, blocks, indices):		
		rc = []
		for a in indices:
			rc.append(torch.cat([torch.cat([blocks[x] for x in i],dim=1) for i in a.long()]))

		return torch.stack(rc)

	def get_block_matrix2(self, blocks, indices):		
		def get_block(blocks, t):
			if t == 0:
				return self.zero_block.squeeze()
			elif t == 1:
				return blocks[0]
			elif t == -1:
				return blocks[1]
			else:
				pdb.set_trace()

		rc = []
		for a in indices:
			# rc.append(torch.cat([torch.cat([blocks[x-1] if x != 0 else self.zero_block.squeeze() for x in i],dim=1) for i in a.long()]))
			rc.append(torch.cat([torch.cat([get_block(blocks,x) for x in i],dim=1) for i in a.long()]))

		return torch.stack(rc)

	def forward(self, input, **kwargs):
		variables = []
		clauses = []
		if self.settings['sparse']:			
			f_vars = None
			f_clauses = None
		else:
			f_vars = input
			f_clauses = f_vars.transpose(1,2)				
		v = self.get_ground_embeddings()		
		variables = v.expand(len(input),v.size(0),1).contiguous()
		ground_variables = variables.view(-1,self.embedding_dim)[:,:self.ground_dim]

		if self.debug:
			print('Variables:')
			print(variables)
			pdb.set_trace()
		# Start propagation

		self.inner_iteration.re_init()
		
		for i in range(self.max_iters):
			# print('Starting iteration %d' % i)
			if (variables != variables).data.any():
				print('Variables have nan!')
				pdb.set_trace()
			variables = self.inner_iteration(variables, f_vars, f_clauses, ground_vars=ground_variables, v_block = self.forward_pos_neg, c_block=self.backwards_pos_neg, **kwargs)
			# variables = self.inner_iteration(variables, v_mat, c_mat, ground_vars=ground_variables, v_block = self.forward_pos_neg, c_block=self.backwards_pos_neg, old_forward=True)
			if self.debug:
				print('Variables:')
				print(variables)
				pdb.set_trace()
			# if (variables[0]==variables[1]).data.all():
			# 	print('Variables identical on (inner) iteration %d' % i)
		aux_losses = Variable(torch.zeros(len(variables)))
		return torch.squeeze(variables), aux_losses

class TopVarEmbedder(nn.Module):
	def __init__(self, **kwargs):
		super(TopVarEmbedder, self).__init__()        
		self.settings = kwargs['settings'] if 'settings' in kwargs.keys() else CnfSettings()
		self.embedding_dim = self.settings['embedding_dim']
		# self.max_variables = self.settings['max_variables']		

	def forward(self, embeddings, output_ind, **kwargs):		
		b = ((torch.abs(output_ind)-1)*self.embedding_dim).view(-1,1)
		ind = b.clone()
		for i in range(self.embedding_dim-1):
			ind = torch.cat([ind,b+1+i],dim=1)
		out = torch.gather(embeddings,1,ind)
		return out

class GraphEmbedder(nn.Module):
	def __init__(self, **kwargs):
		super(GraphEmbedder, self).__init__()        
		self.settings = kwargs['settings'] if 'settings' in kwargs.keys() else CnfSettings()
		self.embedding_dim = self.settings['embedding_dim']
		self.i_mat = nn.Linear(self.embedding_dim,self.embedding_dim)
		self.j_mat = nn.Linear(self.embedding_dim,self.embedding_dim)
	def forward(self, embeddings, batch_size=None, **kwargs):
		if not batch_size:
			batch_size = self.settings['batch_size']
		embeddings = embeddings.view(-1,self.embedding_dim)
		per_var = F.sigmoid(self.i_mat(embeddings)) * F.tanh(self.j_mat(embeddings))
		out = F.tanh(torch.sum(per_var.view(batch_size,-1,self.embedding_dim),dim=1))
		return out

class TopLevelClassifier(nn.Module):
	def __init__(self, num_classes, encoder=None, embedder=None, **kwargs):
		super(TopLevelClassifier, self).__init__()        
		self.num_classes = num_classes
		self.settings = kwargs['settings'] if 'settings' in kwargs.keys() else CnfSettings()
		# self.embedding_dim = self.settings['embedding_dim']
		# self.max_variables = self.settings['max_variables']
		if encoder:
			self.encoder = encoder
			self.encoder.fix_annotations()
		else:
			self.encoder_type = eval(self.settings['encoder_type'])
			self.encoder = self.encoder_type(**kwargs)
		if embedder:
			self.embedder = embedder
		else:
			self.embedder_type = eval(self.settings['embedder_type'])
			self.embedder = self.embedder_type(**kwargs)
		self.softmax_layer = nn.Linear(self.encoder.embedding_dim,num_classes)

	def forward(self, input, **kwargs):
		embeddings, aux_losses = self.encoder(input, **kwargs)
		enc = self.embedder(embeddings,**kwargs)
		return self.softmax_layer(enc), aux_losses     # variables are 1-based


class TopLevelSiamese(nn.Module):
	def __init__(self, **kwargs):
		super(TopLevelSiamese, self).__init__()        
		self.settings = kwargs['settings'] if 'settings' in kwargs.keys() else CnfSettings()
		self.embedding_dim = self.settings['embedding_dim']
		self.max_variables = self.settings['max_variables']
		self.encoder_type = eval(self.settings['encoder_type'])
		self.encoder = self.encoder_type(**kwargs)
		self.embedder_type = eval(self.settings['embedder_type'])
		self.embedder = self.embedder_type(**kwargs)

	def forward(self, input, output_ind, **kwargs):
		left, right = input
		left_idx, right_idx = output_ind
		l_embeddings, _ = self.encoder(left, **kwargs)
		r_embeddings, _ = self.encoder(right, **kwargs)
		l_enc = self.embedder(l_embeddings, output_ind=left_idx, **kwargs)
		r_enc = self.embedder(r_embeddings, output_ind=right_idx, **kwargs)
		return (l_enc, r_enc)



class BatchEqClassifier(nn.Module):
	def __init__(self, num_classes, **kwargs):
		super(BatchEqClassifier, self).__init__()        
		self.num_classes = num_classes
		self.settings = kwargs['settings'] if 'settings' in kwargs.keys() else CnfSettings()
		self.embedding_dim = self.settings['embedding_dim']
		self.max_variables = self.settings['max_variables']
		self.encoder = BatchEncoder(**kwargs)
		self.softmax_layer = nn.Linear(self.encoder.embedding_dim,num_classes)

	def forward(self, input, output_ind):
		embeddings, aux_losses = self.encoder(input)
		b = ((torch.abs(output_ind)-1)*self.embedding_dim).view(-1,1)
		ind = b.clone()
		for i in range(self.embedding_dim-1):
			ind = torch.cat([ind,b+1+i],dim=1)
		out = torch.gather(embeddings,1,ind)
		return self.softmax_layer(out), aux_losses     # variables are 1-based
		
		

class BatchGraphLevelClassifier(nn.Module):
	def __init__(self, num_classes, **kwargs):
		super(BatchGraphLevelClassifier, self).__init__()
		self.encoder = BatchEncoder(**kwargs)
		self.i_mat = nn.Linear(self.encoder.embedding_dim,self.encoder.embedding_dim)
		self.j_mat = nn.Linear(self.encoder.embedding_dim,self.encoder.embedding_dim)
		self.num_classes = num_classes
		self.settings = kwargs['settings'] if 'settings' in kwargs.keys() else CnfSettings()
		self.softmax_layer = nn.Linear(self.encoder.embedding_dim,num_classes)

	def forward(self, input, output_ind):
		embeddings, aux_losses = self.encoder(input)
		embeddings = embeddings.view(-1,self.settings['embedding_dim'])
		per_var = F.sigmoid(self.i_mat(embeddings)) * F.tanh(self.j_mat(embeddings))
		out = F.tanh(torch.sum(per_var.view(len(input[0]),-1,self.settings['embedding_dim']),dim=1)).squeeze()
		return self.softmax_layer(out), aux_losses

class SiameseClassifier(nn.Module):
	def __init__(self, **kwargs):
		super(SiameseClassifier, self).__init__()        
		self.settings = kwargs['settings'] if 'settings' in kwargs.keys() else CnfSettings()
		self.encoder = Encoder(**kwargs)

	def forward(self, inputs, output_ind):
		left, right = inputs
		left_idx, right_idx = output_ind
		l_embeddings, _ = self.encoder(left)
		r_embeddings, _ = self.encoder(right)
		embeddings = [l_embeddings, r_embeddings]

		for i,x in zip([0,1],output_ind):
			neg = (x.data<0).all()
			idx = torch.abs(x).data[0]-1
			out = embeddings[i][idx]
			if neg:
				out = self.encoder.inner_iteration.negation(out)
			embs.append(out)

		return tuple(embs)     

