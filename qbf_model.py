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

def expand_ground_to_state(v, settings=None):
  if not settings:
    settings = CnfSettings()  
  dconst = settings.expand_dim_const.expand(len(v),settings['embedding_dim'] - settings['ground_dim'])
  return torch.cat([v,dconst],dim=1)



class GruOperator(nn.Module):
    def __init__(self, embedding_dim=None, **kwargs):
        super(GruOperator, self).__init__()        
        self.settings = kwargs['settings'] if 'settings' in kwargs.keys() else CnfSettings()
        if embedding_dim:
            self.embedding_dim = embedding_dim
        else:
            self.embedding_dim = self.settings['embedding_dim']        
        self.W_z = nn.Linear(self.embedding_dim,self.embedding_dim,bias=False)
        self.U_z = nn.Linear(self.embedding_dim,self.embedding_dim,bias=self.settings['gru_bias'])
        self.W_r = nn.Linear(self.embedding_dim,self.embedding_dim,bias=False)
        self.U_r = nn.Linear(self.embedding_dim,self.embedding_dim,bias=self.settings['gru_bias'])
        self.W = nn.Linear(self.embedding_dim,self.embedding_dim,bias=False)
        self.U = nn.Linear(self.embedding_dim,self.embedding_dim,bias=self.settings['gru_bias'])
        
    def forward(self, av, prev_emb):
        z = torch.sigmoid(self.W_z(av) + self.U_z(prev_emb))
        r = torch.sigmoid(self.W_r(av) + self.U_r(prev_emb))
        h_tilda = torch.tanh(self.W(av) + self.U(r*prev_emb))
        h = (1-z) * prev_emb + z*h_tilda
        return h


class FactoredInnerIteration(nn.Module):
    def __init__(self, **kwargs):
        super(FactoredInnerIteration, self).__init__()        
        self.settings = kwargs['settings'] if 'settings' in kwargs.keys() else CnfSettings()
        self.ground_comb_type = eval(self.settings['ground_combinator_type'])
        self.non_linearity = eval(self.settings['non_linearity'])
        self.ground_dim = self.settings['ground_dim']
        self.embedding_dim = self.settings['embedding_dim']        
        self.ground_combiner = self.ground_comb_type(self.settings['ground_dim'],self.embedding_dim)
        if self.settings['use_gru']:
            self.gru = GruOperator(settings=self.settings)
        self.cuda = self.settings['cuda']        
        self.vb = nn.Parameter(self.settings.FloatTensor(self.embedding_dim,1))
        self.cb = nn.Parameter(self.settings.FloatTensor(self.embedding_dim,1))
        nn_init.normal_(self.vb)
        nn_init.normal_(self.cb)
        
    def forward(self, variables, v_mat, c_mat, ground_vars=None, v_block=None, c_block=None, **kwargs):        
        if 'old_forward' in kwargs and kwargs['old_forward']:
            return self.forward2(variables,v_mat,c_mat,ground_vars=ground_vars, **kwargs)
        assert(v_block is not None and c_block is not None)
        bsize = kwargs['batch_size'] if 'batch_size' in kwargs else self.settings['batch_size']
        self.max_variables = kwargs['max_variables'] if 'max_variables' in kwargs else self.settings['max_variables']
        org_size = variables.size()
        v = variables.view(-1,self.embedding_dim).t()
        size = v.size(1)    # batch x num_vars
        use_neg = self.settings['negate_type'] != 'minus'
        if use_neg:
            # Tracer()()
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
        if (rc != rc).data.any():            # We won't stand for NaN
            print('NaN in our tensors!!')
            pdb.set_trace()
        # if (rc[0] == rc[1]).data.all():
        #     print('Same embedding. wtf?')
            # pdb.set_trace()
        return rc

class WeightedNegInnerIteration(nn.Module):
    def __init__(self, **kwargs):
        super(WeightedNegInnerIteration, self).__init__()        
        self.settings = kwargs['settings'] if 'settings' in kwargs.keys() else CnfSettings()
        self.ground_comb_type = eval(self.settings['ground_combinator_type'])
        self.non_linearity = eval(self.settings['non_linearity'])
        self.ground_dim = self.settings['ground_dim']
        self.embedding_dim = self.settings['embedding_dim']        
        self.ground_combiner = self.ground_comb_type(self.settings['ground_dim'],self.embedding_dim)
        self.cuda = self.settings['cuda']        
        self.forward_backwards_block = nn.Parameter(self.settings.FloatTensor(2,self.embedding_dim,self.embedding_dim))
        nn_init.normal_(self.forward_backwards_block)        
        if self.settings['use_gru']:
            self.gru = GruOperator(settings=self.settings)
        self.vb = nn.Parameter(self.settings.FloatTensor(self.embedding_dim,1))
        self.cb = nn.Parameter(self.settings.FloatTensor(self.embedding_dim,1))
        nn_init.normal_(self.vb)
        nn_init.normal_(self.cb)
                
    def forward(self, variables, v_mat, c_mat, ground_vars=None, cmat_pos=None, cmat_neg=None, **kwargs):                
        bsize = kwargs['batch_size'] if 'batch_size' in kwargs else self.settings['batch_size']
        self.max_variables = kwargs['max_variables'] if 'max_variables' in kwargs else self.settings['max_variables']
        org_size = variables.size()
        v = variables.view(-1,self.embedding_dim).t()
        c_mat = cmat_pos - cmat_neg
        v_mat = c_mat.t()
        vars_all = torch.mm(self.forward_backwards_block[0],v).t()
        # Tracer()()
        c = torch.mm(c_mat.float(),vars_all)    

        c = self.non_linearity(c + self.cb.squeeze())        
        cv = c.view(-1,self.embedding_dim).t()        
        vars_all = torch.mm(self.forward_backwards_block[1],cv).t()
        try:
            nv = torch.mm(v_mat.float(),vars_all)    
        except:
            Tracer()()
            
        v_emb = self.non_linearity(nv + self.vb.squeeze())        
        v_emb = self.ground_combiner(ground_vars.view(-1,self.ground_dim),v_emb.view(-1,self.embedding_dim))        
        if self.settings['use_gru']:
            new_vars = self.gru(v_emb, variables.view(-1,self.embedding_dim))    
        else:
            new_vars = v_emb
        rc = new_vars.view(-1,self.max_variables*self.embedding_dim,1)
        if (rc != rc).data.any():            # We won't stand for NaN
            print('NaN in our tensors!!')
            pdb.set_trace()
        # if (rc[0] == rc[1]).data.all():
        #     print('Same embedding. wtf?')
            # pdb.set_trace()
        return rc


class QbfClassifier(nn.Module):
    def __init__(self, encoder=None, embedder=None, **kwargs):
        super(QbfClassifier, self).__init__()        
        self.settings = kwargs['settings'] if 'settings' in kwargs.keys() else CnfSettings()
        # self.embedding_dim = self.settings['embedding_dim']
        # self.max_variables = self.settings['max_variables']
        if encoder:
            self.encoder = encoder
            # self.encoder.fix_annotations()
        else:
            self.encoder = QbfEncoder(**kwargs)
        if embedder:
            self.embedder = embedder
        else:
            self.embedder = GraphEmbedder(**kwargs)
        self.softmax_layer = nn.Linear(self.encoder.embedding_dim,2)

    def forward(self, input, **kwargs):
        embeddings = self.encoder(input, **kwargs)
        enc = self.embedder(embeddings, batch_size=len(input), **kwargs)
        return self.softmax_layer(enc)     # variables are 1-based


class QbfEncoder(nn.Module):
    def __init__(self, **kwargs):
        super(QbfEncoder, self).__init__() 
        self.settings = kwargs['settings'] if 'settings' in kwargs.keys() else CnfSettings()
        self.debug = False
        self.ground_dim = self.settings['ground_dim']
        # self.batch_size = 1
        self.batch_size = self.settings['batch_size']
        self.embedding_dim = self.settings['embedding_dim']        
        self.expand_dim_const = Variable(self.settings.zeros(1), requires_grad=False)
        self.max_iters = self.settings['max_iters']
        self.ggn_core = eval(self.settings['ggn_core'])
        self.inner_iteration = self.ggn_core(**kwargs)            
        self.forward_pos_neg = nn.Parameter(self.settings.FloatTensor(2,self.embedding_dim,self.embedding_dim))
        nn_init.normal_(self.forward_pos_neg)        
        self.backwards_pos_neg = nn.Parameter(self.settings.FloatTensor(2,self.embedding_dim,self.embedding_dim))        
        nn_init.normal_(self.backwards_pos_neg)        
                    
    # def expand_ground_to_state(self,v):
    #     # Tracer()()
    #     dconst = self.expand_dim_const.expand(len(v),self.embedding_dim - self.ground_dim)
    #     return torch.cat([v,dconst],dim=1)
    
# This should probably be factored into a base class, its basically the same as for BatchEncoder

# ground_embeddings are (batch,maxvars,ground_dim)

    
    def forward(self, ground_embeddings, clabels, **kwargs):
        variables = []
        clauses = []        
        size = ground_embeddings.size()
        if 'batch_size' in kwargs:
            self.batch_size=kwargs['batch_size']
            assert(self.batch_size==size[0])        
        f_vars = None
        f_clauses = None
        # Tracer()()
        # v = self.expand_ground_to_state(ground_embeddings.view(-1,self.ground_dim)).view(1,-1).transpose(0,1)
        v = expand_ground_to_state(ground_embeddings.view(-1,self.ground_dim)).view(1,-1).transpose(0,1)
        variables = v.view(-1,self.embedding_dim*size[1],1)
        
        # Inner iteration expects (batch*,embedding_dim*maxvar,1) for variables

        if self.debug:
            print('Variables:')
            print(variables)
            pdb.set_trace()
        # Start propagation
        
        for i in range(self.max_iters):
            # print('Starting iteration %d' % i)
            if (variables != variables).data.any():
                print('Variables have nan!')
                pdb.set_trace()
            variables = self.inner_iteration(variables, f_vars, f_clauses, batch_size=size[0], ground_vars=ground_embeddings, 
                        v_block = self.forward_pos_neg, c_block=self.backwards_pos_neg, max_variables=size[1], **kwargs)
            # variables = self.inner_iteration(variables, v_mat, c_mat, ground_vars=ground_variables, v_block = self.forward_pos_neg, c_block=self.backwards_pos_neg, old_forward=True)
            if self.debug:
                print('Variables:')
                print(variables)
                pdb.set_trace()
            # if (variables[0]==variables[1]).data.all():
            #     print('Variables identical on (inner) iteration %d' % i)
        # aux_losses = Variable(torch.zeros(len(variables)))
        return torch.squeeze(variables)

class QbfNewEncoder(nn.Module):
    def __init__(self, **kwargs):
        super(QbfNewEncoder, self).__init__() 
        self.settings = kwargs['settings'] if 'settings' in kwargs.keys() else CnfSettings()
        self.debug = False
        self.ground_dim = self.settings['ground_dim']
        self.vlabel_dim = self.settings['vlabel_dim']
        self.clabel_dim = self.settings['clabel_dim']
        self.vemb_dim = self.settings['vemb_dim']
        self.cemb_dim = self.settings['cemb_dim']
        self.batch_size = self.settings['batch_size']
        # self.embedding_dim = self.settings['embedding_dim']                
        self.max_iters = self.settings['max_iters']        
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
            elif self.settings['use_ln']:
                self.vnorm_layers.append(nn.LayerNorm(self.vemb_dim))


        if self.settings['use_gru']:
            self.gru = GruOperator(embedding_dim=self.vemb_dim, settings=self.settings)

        self.W_L_params = nn.ParameterList(W_L_params)
        self.B_L_params = nn.ParameterList(B_L_params)
        self.W_C_params = nn.ParameterList(W_C_params)
        self.B_C_params = nn.ParameterList(B_C_params)
        
                    
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


# vlabels are (vars,vlabel_dim)
# clabels are sparse (clauses,clabel_dim)
# cmat_pos and cmat_neg is the bs*v -> bs*c block-diagonal adjacency matrix 

    def forward(self, vlabels, clabels, cmat_pos, cmat_neg, **kwargs):
        # size = vlabels.size()
        # bs = size[0]
        # maxvars = size[1]
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
            # if bs>1:
            #     Tracer()()            
            pos_vars = torch.cat([pos_vars,pv_t_pre,nv_t_pre],dim=1)
            neg_vars = torch.cat([neg_vars,nv_t_pre,pv_t_pre],dim=1)


        return pos_vars, neg_vars   


class QbfEncoderAag(nn.Module):
    """
    Encoder for the aag graph.
    First Attempt.
    """
    def __init__(self, **kwargs):
        super(QbfNewEncoder, self).__init__() 
        self.settings = kwargs['settings'] if 'settings' in kwargs.keys() else CnfSettings()
        self.debug = False
        
        # Dimensions:
        self.num_literals = self.settings['num_literals']
        
        self.batch_size = self.settings['batch_size']
        self.non_linearity = eval(self.settings['non_linearity'])
        
        # Parameters:
        self.W_A = nn_init.normal_(nn.Parameter(self.settings.FloatTensor(self.num_literals, self.num_literals)))
        self.B_A = nn_init.normal_(nn.Parameter(self.settings.FloatTensor(self.num_literals)))


    def forward(self, llabels, amat, **kwargs):
        """
        2n = num_literals
        llabels: shape (2n x 8) , initial embeddings for each literal
        amat: shape (2n x 2n) , adjacency matrix of `aag and gates`
        """

#        pos_vars = vlabels
#        neg_vars = vlabels
#        vmat_pos = cmat_pos.t()
#        vmat_neg = cmat_neg.t()
        
#        import ipdb
#        ipdb.set_trace()

        L = (torch.mm(self.W_A, amat) + self.B_A)
        NL = self.non_linearity(L)
        return NL        
     
        

class QbfFixedEncoder(nn.Module):
    def __init__(self, **kwargs):
        super(QbfFixedEncoder, self).__init__() 
        self.settings = kwargs['settings'] if 'settings' in kwargs.keys() else CnfSettings()
        self.debug = False
        self.ground_dim = self.settings['ground_dim']
        self.vlabel_dim = self.settings['vlabel_dim']
        self.clabel_dim = self.settings['clabel_dim']
        self.vemb_dim = self.settings['vemb_dim']
        self.cemb_dim = self.settings['cemb_dim']
        self.batch_size = self.settings['batch_size']
        # self.embedding_dim = self.settings['embedding_dim']                
        self.max_iters = self.settings['max_iters']        
        self.non_linearity = eval(self.settings['non_linearity'])
        self.use_gru = self.settings['use_gru']
        W_L_params = []
        B_L_params = []
        W_C_params = []
        B_C_params = []
        # if self.settings['use_bn']:
        self.vnorm_layers = nn.ModuleList([])
        self.W_L_params = nn.Parameter(self.settings.FloatTensor(self.cemb_dim,self.vlabel_dim+2*self.vemb_dim))
        self.B_L_params = nn.Parameter(self.settings.FloatTensor(self.cemb_dim))
        self.W_C_params = nn.Parameter(self.settings.FloatTensor(self.vemb_dim,self.clabel_dim+self.cemb_dim))
        self.B_C_params = nn.Parameter(self.settings.FloatTensor(self.vemb_dim))
        nn_init.normal_(self.W_L_params)
        nn_init.normal_(self.B_L_params)        
        nn_init.normal_(self.W_C_params)                
        nn_init.normal_(self.B_C_params)
        for i in range(self.max_iters):
            if self.settings['use_bn']:
                self.vnorm_layers.append(nn.BatchNorm1d(self.vemb_dim))
            elif self.settings['use_ln']:
                self.vnorm_layers.append(nn.LayerNorm(self.vemb_dim))
        if self.use_gru:
            self.gru = GruOperator(embedding_dim=2*self.vemb_dim, settings=self.settings)
        # self.vlabel_base = self.settings.expand_dim_const.expand(1,self.vemb_dim)
                        
    def copy_from_encoder(self, other, freeze=False):
        for i in range(len(other.W_L_params)):
            self.W_L_params = other.W_L_params
            self.B_L_params = other.B_L_params
            self.W_C_params = other.W_C_params
            self.B_C_params = other.B_C_params
            if freeze:
                self.W_L_params.requires_grad=False
                self.B_L_params.requires_grad=False
                self.W_C_params.requires_grad=False
                self.B_C_params.requires_grad=False
            if self.settings['use_bn']:
                for i, layer in enumerate(other.vnorm_layers):
                    self.vnorm_layers[i].load_state_dict(layer.state_dict())
            if self.settings['use_gru']:
                self.gru.load_state_dict(other.gru.state_dict())

# vlabels are (batch,maxvars,vlabel_dim)
# clabels are sparse (batch,maxvars,maxvars,label_dim)
# cmat_pos and cmat_neg is the bs*v -> bs*c block-diagonal adjacency matrix 

    def forward(self, vlabels, clabels, cmat_pos , cmat_neg, **kwargs):
        size = vlabels.size()
        bs = size[0]
        maxvars = size[1]
        # pos_vars = vlabels
        # neg_vars = vlabels
        pos_vars = self.settings.expand_dim_const.expand(bs,self.vemb_dim*2)
        neg_vars = self.settings.expand_dim_const.expand(bs,self.vemb_dim*2)

        vmat_pos = cmat_pos.t()
        vmat_neg = cmat_neg.t()
        for i in range(self.max_iters):
            if self.use_gru and i>0:
                pos_vars = self.gru(prev_pos_vars,pos_vars)
                neg_vars = self.gru(prev_neg_vars,neg_vars)
            prev_pos_vars = pos_vars
            prev_neg_vars = neg_vars
            pos_vars = torch.cat([vlabels,pos_vars],dim=1)
            neg_vars = torch.cat([vlabels,neg_vars],dim=1)
            # results is everything we computed so far, its precisely the correct input to W_L_t
            av = (torch.mm(cmat_pos,pos_vars)+torch.mm(cmat_neg,neg_vars)).t()
            c_t_pre = self.non_linearity(torch.mm(self.W_L_params,av).t() + self.B_L_params)
            c_t = torch.cat([clabels,c_t_pre],dim=1)
            pv = torch.mm(vmat_pos,c_t).t()
            nv = torch.mm(vmat_neg,c_t).t()
            pv_t_pre = self.non_linearity(torch.mm(self.W_C_params,pv).t() + self.B_C_params)
            nv_t_pre = self.non_linearity(torch.mm(self.W_C_params,nv).t() + self.B_C_params)
            if self.settings['use_bn'] or self.settings['use_ln']:
                pv_t_pre = self.vnorm_layers(pv_t_pre.contiguous())
                nv_t_pre = self.vnorm_layers(nv_t_pre.contiguous())            
            # if bs>1:
            #     Tracer()()            
            pos_vars = torch.cat([pv_t_pre,nv_t_pre],dim=1)
            neg_vars = torch.cat([nv_t_pre,pv_t_pre],dim=1)


        pos_vars = torch.cat([vlabels,pos_vars],dim=1)
        neg_vars = torch.cat([vlabels,neg_vars],dim=1)
        return pos_vars, neg_vars        


class QbfSimpleEncoder(nn.Module):
    def __init__(self, **kwargs):
        super(QbfSimpleEncoder, self).__init__() 
        self.settings = kwargs['settings'] if 'settings' in kwargs.keys() else CnfSettings()
        self.debug = False
        self.vlabel_dim = self.settings['ground_dim']
        self.clabel_dim = self.settings['clabel_dim']
        self.vemb_dim = self.settings['vemb_dim']
        self.cemb_dim = self.settings['cemb_dim']
        self.max_iters = self.settings['max_iters']        
        self.non_linearity = eval(self.settings['non_linearity'])
        self.W_V_pos_params = nn.Parameter(torch.FloatTensor(self.vlabel_dim+self.vemb_dim,self.cemb_dim))
        self.W_V_neg_params = nn.Parameter(torch.FloatTensor(self.vlabel_dim+self.vemb_dim,self.cemb_dim))
        self.B_V_params = nn.Parameter(torch.FloatTensor(self.cemb_dim))
        self.W_C_pos_params = nn.Parameter(torch.FloatTensor(self.clabel_dim+self.cemb_dim,self.vemb_dim))
        self.W_C_neg_params = nn.Parameter(torch.FloatTensor(self.clabel_dim+self.cemb_dim,self.vemb_dim))
        self.B_C_params = nn.Parameter(torch.FloatTensor(self.vemb_dim))
        nn_init.normal_(self.W_V_pos_params)
        nn_init.normal_(self.W_V_neg_params)
        nn_init.normal_(self.B_V_params)        
        nn_init.normal_(self.W_C_pos_params)
        nn_init.normal_(self.W_C_neg_params)
        nn_init.normal_(self.B_C_params)        
        if self.settings['use_bn']:
            self.vnorm_layer = nn.BatchNorm1d(self.vemb_dim)
        elif self.settings['use_ln']:
            self.vnorm_layer = nn.LayerNorm(self.vemb_dim)
        self.const_var = self.settings.zeros(1)


        if self.settings['use_gru']:
            self.gru = GruOperator(embeddding_dim=self.vemb_dim, settings=self.settings)
        
                    
    def copy_from_encoder(self, other, freeze=False):
        self.W_V_pos_params = other.W_V_pos_params
        self.W_V_neg_params = other.W_V_neg_params
        self.B_V_params = other.B_V_params
        self.W_C_pos_params = other.W_C_pos_params
        self.W_C_neg_params = other.W_C_neg_params
        self.B_C_params = other.B_C_params
        if freeze:
            self.W_V_pos_params.requires_grad=False
            self.W_V_neg_params.requires_grad=False
            self.B_V_params.requires_grad=False
            self.W_C_pos_params.requires_grad=False
            self.W_C_neg_params.requires_grad=False
            self.B_C_params.requires_grad=False
        if self.settings['use_bn']:            
            self.vnorm_layer.load_state_dict(other.vnorm_layer.state_dict())
        elif self.settings['use_gru']:
            self.gru.load_state_dict(other.gru.state_dict())


# vlabels are (batch,maxvars,vlabel_dim)
# clabels are sparse (batch,maxvars,maxvars,label_dim)
# cmat_pos and cmat_neg is the bs*v -> bs*c block-diagonal adjacency matrix 

    def forward(self, vlabels, clabels, cmat_pos, cmat_neg, **kwargs):
        size = vlabels.size()
        bs = size[0]
        pos_vars = vlabels
        neg_vars = vlabels
        vmat_pos = cmat_pos.t()
        vmat_neg = cmat_neg.t()        
        vembs = self.const_var.expand(bs,self.vemb_dim)

        for _ in range(self.max_iters):
            prev_vembs = vembs
            vs = torch.cat([vlabels,vembs],dim=1)
            pv = torch.mm(vs,self.W_V_pos_params)
            nv = torch.mm(vs,self.W_V_neg_params)
            cembs = self.non_linearity(torch.mm(cmat_pos,pv) + torch.mm(cmat_neg,nv) + self.B_V_params)
            cs = torch.cat([clabels,cembs],dim=1)
            pc = torch.mm(cs,self.W_C_pos_params)
            nc = torch.mm(cs,self.W_C_neg_params)
            vembs = self.non_linearity(torch.mm(vmat_pos,pc) + torch.mm(vmat_neg,nc) + self.B_C_params)
            if self.settings['use_bn'] or self.settings['use_ln']:
                vembs = self.vnorm_layer(vembs)
            elif self.settings['use_gru']:
                vembs = self.gru(prev_vembs,vembs)

        return vembs        




class QbfAttention(nn.Module):
    def __init__(self, emb_dim, n_heads=10, seed_dim=None, **kwargs):
        super(QbfAttention, self).__init__()        
        self.settings = kwargs['settings'] if 'settings' in kwargs.keys() else CnfSettings()
        self.emb_dim = emb_dim
        self.seed_dim = seed_dim if seed_dim else self.settings['state_dim']
        self.non_linearity = eval(self.settings['attn_non_linearity'])
        self.n_heads = n_heads
        self.w_qs = nn.Parameter(self.settings.FloatTensor(n_heads, self.emb_dim, self.seed_dim))
        nn_init.xavier_normal_(self.w_qs)
        
        # embeddings are used as both keys and values, and are bs*(numvar pos+neg)*vemb_dim
        # query seed is bs x seed_dim
        # attn_mask is bs x num_vars
    def forward(self, query_seed, embeddings, attn_mask=None, values=None, **kwargs):
        def normalize_tensor(x):
            # return x
            return x / x.norm(2,2).unsqueeze(2).expand_as(x)
        bs, varnum, _ = embeddings.size()
        if not values:
            values = embeddings
        norm_embs = normalize_tensor(embeddings)
        qs = self.non_linearity(torch.mm(self.w_qs.view(-1,self.seed_dim),query_seed.t()).t().contiguous().view(bs,self.n_heads,-1))            # bs*n_heads*emb_dim
        norm_qs = normalize_tensor(qs)

        # We'd like the attention vectors in norm_qs to be mostly orthogonal, so we penalize norm(A^t*A-I)
        diff = torch.bmm(norm_qs,norm_qs.transpose(1,2)) - Variable(torch.eye(self.n_heads))
        loss = (diff*diff).sum() / bs

        attn = torch.bmm(norm_qs,norm_embs.transpose(1,2))
        # Translate attention mask from bs*numvars to basically double that
        softmax_mask = 1-torch.cat([attn_mask]*2,dim=1).unsqueeze(1).expand_as(attn)
        attn.data.masked_fill_(softmax_mask.byte(), -float('inf'))
        final_attn = F.softmax(attn.view(-1,varnum),dim=1).view(attn.size())
        rc = torch.bmm(final_attn,values)

        return rc, loss

class QbfFlattenedAttention(nn.Module):
    def __init__(self, emb_dim, n_heads=10, seed_dim=None, **kwargs):
        super(QbfFlattenedAttention, self).__init__()        
        self.settings = kwargs['settings'] if 'settings' in kwargs.keys() else CnfSettings()
        self.emb_dim = emb_dim
        self.seed_dim = seed_dim if seed_dim else self.settings['state_dim']
        self.non_linearity = eval(self.settings['attn_non_linearity'])
        self.n_heads = n_heads
        self.w_qs = nn.Parameter(self.settings.FloatTensor(n_heads, self.emb_dim, self.seed_dim))
        nn_init.xavier_normal_(self.w_qs)
                
        # embeddings are used as both keys and values, and are bs*(numvar pos+neg)*vemb_dim
        # query seed is bs x seed_dim
        # attn_mask is bs x (num_vars pos+neg)
    def forward(self, query_seed, embeddings, attn_mask=None, values=None, **kwargs):
        def normalize_tensor(x):
            # return x
            return x / x.norm(2,2).unsqueeze(2).expand_as(x)
        bs, varnum, _ = embeddings.size()
        if not values:
            values = embeddings
        norm_embs = normalize_tensor(embeddings)
        qs = self.non_linearity(torch.mm(self.w_qs.view(-1,self.seed_dim),query_seed.t()).t().contiguous().view(bs,self.n_heads,-1))            # bs*n_heads*emb_dim
        norm_qs = normalize_tensor(qs)

        # We'd like the attention vectors in norm_qs to be mostly orthogonal, so we penalize norm(A^t*A-I)
        diff = torch.bmm(norm_qs,norm_qs.transpose(1,2)) - Variable(torch.eye(self.n_heads))
        loss = (diff*diff).sum() / bs

        attn = torch.bmm(norm_qs,norm_embs.transpose(1,2))
        # Translate attention mask from bs*numvars to basically double that
        softmax_mask = 1-attn_mask.unsqueeze(1).expand_as(attn)
        attn.data.masked_fill_(softmax_mask.byte(), -float('inf'))
        final_attn = F.softmax(attn.view(-1,varnum),dim=1).view(attn.size())
        rc = torch.bmm(final_attn,values)

        return rc, loss

        

