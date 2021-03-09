import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import init as nn_init
import torch.nn.functional as F
import numpy as np
from IPython.core.debugger import Tracer


import utils
from qbf_data import *
from batch_model import GraphEmbedder, GroundCombinator, DummyGroundCombinator
from settings import *


class SatEncoder(nn.Module):
	def __init__(self, **kwargs):
		super(SatEncoder, self).__init__() 
		self.settings = kwargs['settings'] if 'settings' in kwargs.keys() else CnfSettings()
		self.debug = False
		self.vlabel_dim = self.settings['vlabel_dim']
		self.clabel_dim = self.settings['clabel_dim']
		self.vemb_dim = self.settings['vemb_dim']
		self.cemb_dim = self.settings['cemb_dim']		
		self.max_iters = self.settings['max_iters']
		self.final_vdim = 2*self.max_iters*self.vemb_dim+self.vlabel_dim
		self.non_linearity = eval(self.settings['non_linearity'])
		W_L_params = []
		B_L_params = []
		W_C_params = []
		B_C_params = []
		# if self.settings['use_bn']:
		self.vnorm_layers = nn.ModuleList([])
		for i in range(self.max_iters):
			W_L_params.append(nn.Parameter(self.settings.FloatTensor(self.cemb_dim,self.vlabel_dim+2*i*self.vemb_dim)))
			B_L_params.append(nn.Parameter(self.settings.FloatTensor(self.cemb_dim)))
			W_C_params.append(nn.Parameter(self.settings.FloatTensor(self.vemb_dim,self.clabel_dim+self.cemb_dim)))
			B_C_params.append(nn.Parameter(self.settings.FloatTensor(self.vemb_dim)))
			nn_init.normal_(W_L_params[i])
			nn_init.normal_(B_L_params[i])		
			nn_init.normal_(W_C_params[i])				
			nn_init.normal_(B_C_params[i])
			if self.settings['use_bn']:
				self.vnorm_layers.append(nn.BatchNorm1d(self.vemb_dim))
			if self.settings['use_ln']:
				self.vnorm_layers.append(nn.LayerNorm(self.vemb_dim))

		self.W_L_params = nn.ParameterList(W_L_params)
		self.B_L_params = nn.ParameterList(B_L_params)
		self.W_C_params = nn.ParameterList(W_C_params)
		self.B_C_params = nn.ParameterList(B_C_params)
		self.final_v2c = nn.Parameter(self.settings.FloatTensor(self.cemb_dim, self.final_vdim))
		self.final_bias = nn.Parameter(self.settings.FloatTensor(self.cemb_dim))
		nn_init.normal_(self.final_v2c)
		nn_init.normal_(self.final_bias)
		
					
	def copy_from_encoder(self, other, freeze=False):
		for i in range(len(other.W_L_params)):
			self.W_L_params[i] = other.W_L_params[i]
			self.B_L_params[i] = other.B_L_params[i]
			self.W_C_params[i] = other.W_C_params[i]
			self.B_C_params[i] = other.B_C_params[i]
			if freeze:
				self.W_L_params[i].requires_grad=False
				self.B_L_params[i].requires_grad=False
				self.W_C_params[i].requires_grad=False
				self.B_C_params[i].requires_grad=False
			if self.settings['use_bn']:
				for i, layer in enumerate(other.vnorm_layers):
					self.vnorm_layers[i].load_state_dict(layer.state_dict())


# vlabels are (batch,maxvars,vlabel_dim)
# clabels are sparse (batch,maxvars,maxvars,label_dim)
# cmat_pos and cmat_neg is the bs*v -> bs*c block-diagonal adjacency matrix 

	def forward(self, vlabels, clabels, cmat_pos, cmat_neg, do_timing=False, **kwargs):
		pos_vars = vlabels
		neg_vars = vlabels
		vmat_pos = cmat_pos.t()
		vmat_neg = cmat_neg.t()
		
		for t, p in enumerate(self.W_L_params):
			# results is everything we computed so far, its precisely the correct input to W_L_t
			av = (torch.mm(cmat_pos,pos_vars)+torch.mm(cmat_neg,neg_vars)).t()
			c_t_pre = self.non_linearity(torch.mm(self.W_L_params[t],av).t() + self.B_L_params[t])
			c_t = torch.cat([clabels,c_t_pre],dim=1)
			pv = torch.mm(vmat_pos,c_t).t()
			nv = torch.mm(vmat_neg,c_t).t()
			pv_t_pre = self.non_linearity(torch.mm(self.W_C_params[t],pv).t() + self.B_C_params[t])
			nv_t_pre = self.non_linearity(torch.mm(self.W_C_params[t],nv).t() + self.B_C_params[t])
			if self.settings['use_bn'] or self.settings['use_ln']:
				pv_t_pre = self.vnorm_layers[t](pv_t_pre.contiguous())
				nv_t_pre = self.vnorm_layers[t](nv_t_pre.contiguous())			
			pos_vars = torch.cat([pos_vars,pv_t_pre,nv_t_pre],dim=1)
			neg_vars = torch.cat([neg_vars,nv_t_pre,pv_t_pre],dim=1)

		# Final half-iteration

		av = (torch.mm(cmat_pos,pos_vars)+torch.mm(cmat_neg,neg_vars)).t()
		c_t_pre = self.non_linearity(torch.mm(self.final_v2c,av).t() + self.final_bias)
		rc = torch.cat([clabels,c_t_pre],dim=1)		
		

		return rc