import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils
import numpy as np
from IPython.core.debugger import Tracer
import pdb
from settings import *


class TestDup(nn.Module):
	def __init__(self, param1):
		super(TestDup, self).__init__()
		self.extra_embedding = nn.Embedding(1, param1, max_norm=1.)				
		self.layer1 = nn.Linear(param1*3,param1)

	@property
	def false(self):
		return self.extra_embedding(Variable(torch.LongTensor([0]), requires_grad=False))

	def forward(self, inputs):
		return self.layer1(torch.cat([self.false,self.false,inputs],dim=1))
		

class TestAdd(nn.Module):
	def __init__(self, param1):
		super(TestAdd, self).__init__()
		self.coefficient = nn.Parameter(torch.Tensor([param1]))

	def forward(self, input):
		return self.coefficient * input[0] + input[1]

class ResidualCombine(nn.Module):
	def __init__(self, input_size, embedding_dim):
		super(ResidualCombine, self).__init__()        
		self.layer1 = nn.Linear(input_size*embedding_dim,embedding_dim)
		self.layer2 = nn.Linear(input_size*embedding_dim,embedding_dim)

	def forward(self, input):
		try:
			out = utils.normalize(F.sigmoid(self.layer1(input)) + self.layer2(input))
		except Exception as e:
			print(e)
			Tracer()()
		return out

class SimpleCombinator(nn.Module):
	def __init__(self, embedding_dim, hidden_dim=12):
		super(SimpleCombinator, self).__init__()        
		self.layer1 = nn.Linear(embedding_dim,hidden_dim)
		self.layer2 = nn.Linear(hidden_dim,embedding_dim)

	def forward(self, inputs):
		out = [self.layer2(F.sigmoid(self.layer1(x))) for x in inputs]
		return torch.stack(out,dim=2).sum(dim=2).view(out[0].size())

class GroundCombinator(nn.Module):
	def __init__(self, ground_dim, embedding_dim, hidden_dim=12):
		super(GroundCombinator, self).__init__()        
		self.layer1 = nn.Linear(embedding_dim+ground_dim,hidden_dim)
		self.layer2 = nn.Linear(hidden_dim,embedding_dim)

	def forward(self, ground,state):
		return self.layer2(F.sigmoid(self.layer1(torch.cat([ground,state],dim=1))))
		
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


class InnerIteration(nn.Module):
	def __init__(self, get_ground_embeddings, embedding_dim, max_variables, num_ground_variables, split=True, permute=True, **kwargs):
		super(InnerIteration, self).__init__()        
		self.settings = kwargs['settings'] if 'settings' in kwargs.keys() else CnfSettings()
		self.comb_type = eval(self.settings['combinator_type'])
		self.ground_comb_type = eval(self.settings['ground_combinator_type'])
		self.get_ground_embeddings = get_ground_embeddings
		self.embedding_dim = embedding_dim
		self.max_variables = max_variables
		self.split = split
		self.permute = permute
		self.num_ground_variables = num_ground_variables
		self.negation = nn.Linear(embedding_dim, embedding_dim)   # add non-linearity?		
		self.extra_embedding = nn.Embedding(1, embedding_dim, max_norm=1.)				
		self.clause_combiner = self.comb_type(embedding_dim)
		self.variable_combiner = self.comb_type(embedding_dim)
		self.ground_combiner = self.ground_comb_type(self.settings['ground_dim'],embedding_dim)
		self.cuda = kwargs['cuda']		

		self.W_z = nn.Linear(self.embedding_dim,self.embedding_dim,bias=False)
		self.U_z = nn.Linear(self.embedding_dim,self.embedding_dim,bias=self.settings['gru_bias'])
		self.W_r = nn.Linear(self.embedding_dim,self.embedding_dim,bias=False)
		self.U_r = nn.Linear(self.embedding_dim,self.embedding_dim,bias=self.settings['gru_bias'])
		self.W = nn.Linear(self.embedding_dim,self.embedding_dim,bias=False)
		self.U = nn.Linear(self.embedding_dim,self.embedding_dim,bias=self.settings['gru_bias'])


	@property
	def false(self):
		return self.extra_embedding(Variable(self.settings.LongTensor([0]), requires_grad=False))

	@property
	def true(self):
		return self.negation(self.false)

	def prepare_clauses(self, clauses):
		if self.permute and False:  	
			rc = torch.cat(utils.permute_seq(clauses),dim=1)
			if not self.split:
				return rc
			else:
				org = torch.cat(clauses,1)			# split
				return torch.cat([org,rc])	
		else:
			# return torch.cat(clauses,dim=1)
			return clauses

 # i is the index of the special variable (the current one)
	def prepare_variables(self, variables, curr_variable):
		tmp = variables.pop(curr_variable)
		if self.permute and False:    		
			rc = [tmp] + utils.permute_seq(variables)	    		    	
			try:
				perm = torch.cat(rc,1)
			except RuntimeError:
				Tracer()()
			if not self.split:
				return perm
			else:
				org = torch.cat([tmp] + variables,1)        # splitting batch
				return torch.cat([org,perm])
		else:
			rc = [tmp] + variables
			# return torch.cat(rc,1)
			return rc



	def _forward_clause(self, variables, clause, i):
		c_vars = []
		for j in range(self.max_variables):
			if j<len(clause):								# clause is a list of tensors
				l=clause[j]									# l is a tensored floaty integer
				ind = torch.abs(l)-1       					# variables in clauses are 1-based and negative if negated
				v = torch.stack(variables)[ind.data][0] 	# tensored variables (to be indexed by tensor which is inside a torch variable..gah)
				if (ind==i).data.all():
					ind_in_clause = j
				if (l < 0).data.all():
					v = self.negation(v)
			else:
				continue
			c_vars.append(v)

		return self.variable_combiner(self.prepare_variables(c_vars,ind_in_clause))

	def gru(self, av, prev_emb):
		z = F.sigmoid(self.W_z(av) + self.U_z(prev_emb))
		r = F.sigmoid(self.W_r(av) + self.U_r(prev_emb))
		h_tilda = F.tanh(self.W(av) + self.U(r*prev_emb))
		h = (1-z) * prev_emb + z*h_tilda
		return h


	def forward(self, variables, formula):
		out_embeddings = []
		for i,clauses in enumerate(formula):
			# print('Clauses for variable %d: %d' % (i+1, len(clauses)))			
			if clauses:
				clause_embeddings = [self._forward_clause(variables,c, i) for c in clauses]
				new_var_embedding = self.ground_combiner(self.get_ground_embeddings(i),self.clause_combiner(self.prepare_clauses(clause_embeddings)))
				out_embeddings.append(new_var_embedding)
			else:
				out_embeddings.append(variables[i])

		new_vars = self.gru(torch.cat(out_embeddings,dim=0), torch.cat(variables,dim=0))
		return torch.chunk(new_vars,len(new_vars))


class Encoder(nn.Module):
	def __init__(self, embedding_dim, num_ground_variables, max_iters, **kwargs):
		super(Encoder, self).__init__() 
		self.settings = kwargs['settings'] if 'settings' in kwargs.keys() else CnfSettings()
		self.ground_dim = self.settings['ground_dim']
		self.embedding_dim = embedding_dim		
		self.expand_dim_const = Variable(torch.zeros(1,self.embedding_dim - self.ground_dim))
		self.max_iters = max_iters		
		self.num_ground_variables = num_ground_variables		
		self.embedding = nn.Embedding(num_ground_variables, self.ground_dim, max_norm=1.)				
		self.tseitin_embedding = nn.Embedding(1, self.ground_dim, max_norm=1.)		
		self.inner_iteration = InnerIteration(self.get_ground_embeddings, embedding_dim, num_ground_variables=num_ground_variables, **kwargs)
		self.use_ground = self.settings['use_ground']
	
# input is one training sample (a formula), we'll permute it a bit at every iteration and possibly split to create a batch

	def expand_ground_to_state(self,v):
		return torch.cat([v,self.expand_dim_const],dim=1)

	@property
	def tseitin(self):
		return self.tseitin_embedding(Variable(self.settings.LongTensor([0])))

	def get_ground_embeddings(self,i):
		if i<self.num_ground_variables and self.use_ground:
			return self.embedding(Variable(self.settings.LongTensor([i])))
		else:
			return self.tseitin
		

	def forward(self, input):
		variables = []        
		for i in range(len(input)):
			v = self.expand_ground_to_state(self.get_ground_embeddings(i))
			variables.append(v)
		for i in range(self.max_iters):
			# print('Starting iteration %d' % i)
			variables = self.inner_iteration(variables, input)

		# We add loss on each variable embedding to encourage different elements in the batch to stay close. 
	
		aux_losses = Variable(torch.zeros(len(variables)))
		return variables, aux_losses


class EqClassifier(nn.Module):
	def __init__(self, num_classes, **kwargs):
		super(EqClassifier, self).__init__()        
		self.num_classes = num_classes
		self.settings = kwargs['settings'] if 'settings' in kwargs.keys() else CnfSettings()
		self.encoder = Encoder(**kwargs)
		self.softmax_layer = nn.Linear(self.encoder.embedding_dim,num_classes)

	def forward(self, input, output_ind):
		embeddings, aux_losses = self.encoder(input)

		neg = (output_ind.data<0).all()
		idx = torch.abs(output_ind).data[0]-1
		out = embeddings[idx]
		if neg:
			out = self.encoder.inner_iteration.negation(out)
		return self.softmax_layer(out), aux_losses     # variables are 1-based
		# return F.relu(self.softmax_layer(embeddings[output_ind.data[0]-1])), aux_losses     # variables are 1-based
		

class GraphLevelClassifier(nn.Module):
	def __init__(self, num_classes, **kwargs):
		super(GraphLevelClassifier, self).__init__()
		self.encoder = Encoder(**kwargs)
		self.i_mat = nn.Linear(self.encoder.embedding_dim,self.encoder.embedding_dim)
		self.j_mat = nn.Linear(self.encoder.embedding_dim,self.encoder.embedding_dim)
		self.num_classes = num_classes
		self.settings = kwargs['settings'] if 'settings' in kwargs.keys() else CnfSettings()
		self.softmax_layer = nn.Linear(self.encoder.embedding_dim,num_classes)

	def forward(self, input, output_ind):
		embeddings, aux_losses = self.encoder(input)

		out = F.tanh(sum([F.sigmoid(self.i_mat(x)) * F.tanh(self.j_mat(x)) for x in embeddings]))

		return self.softmax_layer(out), aux_losses     # variables are 1-based
		# return F.relu(self.softmax_layer(embeddings[output_ind.data[0]-1])), aux_losses     # variables are 1-based
		

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

