import os
import sys
import ipdb
import torch
import pickle
import dgl
import numpy as np
import functools
import operator
# from lru import LRU
from torchvision import transforms
from torch.utils.data import Dataset
from utils_dir.utils import load_dir, load_files
from sat_code.capture import get_graph

CACHE_SIZE = 200


def sat_collate_fn(batch):
  rc = {}
  rc['gss'] = torch.cat([x['gss'] for x in batch],dim=0)
  rc['graph'] = dgl.batch_hetero([x['graph'] for x in batch])
  rc['vnum'] = [x['vnum'] for x in batch]
  rc['cnum'] = [x['cnum'] for x in batch]

  labels = rc['graph'].nodes['clause'].data['clause_effective_targets']
  pred_idx = torch.where(rc['graph'].nodes['clause'].data['predicted_clauses'])[0]
  # print("collate says, pred_idx (out of {}) is:".format(labels.size(0)))
  # print(pred_idx)

  labels_learnt = labels[pred_idx]

  return rc, labels_learnt

def load_formula(fname):
  with open(fname,'rb') as f:
    formula = pickle.load(f)
  g_dgl = get_graph(formula['adjacency'], torch.Tensor(formula['clause_feat_and_labels']), torch.Tensor(formula['lit_feat']))
  rc = {}
  if formula['lit_feat'].max() == np.inf:
    ipdb.set_trace()
  rc['graph'] = g_dgl
  rc['vnum'] = g_dgl.number_of_nodes('literal')
  rc['cnum'] = g_dgl.number_of_nodes('clause')
  rc['gss'] = torch.Tensor(formula['gss']).expand(rc['cnum'], len(formula['gss']))
  return rc

def cudaize_sample(sample):
  sample['gss'] = sample['gss'].cuda(non_blocking=True)
  sample['graph'] = sample['graph'].to(torch.device('cuda:0'))

  return sample

class MakeLbdTarget(object):

  def translate_lbd(self, inp):
    rc = torch.zeros_like(inp)
    rc[inp>3] += 1
    rc[inp>6] += 1
    rc[inp>9] += 1
    rc[inp>12] += 1
    return rc.long()

  def __call__(self, sample):
    G = sample['graph']
    G.nodes['clause'].data['clause_effective_targets'] = self.translate_lbd(G.nodes['clause'].data['clause_labels'][:,2])

    return sample

class MakeEverUsedTarget(object):
  def __call__(self, sample):
    G = sample['graph']

    effective_val = G.nodes['clause'].data['clause_targets'][:,2] - G.nodes['clause'].data['clause_labels'][:,0]
    G.nodes['clause'].data['clause_effective_targets'] = (effective_val > 0).long()
    return sample

class CapGraph(object):
  def __init__(self, n):
    self.n = n    

  def get_partition_neighbors(self, adjmat, seed_indices):
    all_indices = adjmat.indices()
    xs = [(all_indices[0] == x) for x in seed_indices]   # The indices we care for
    fbase = torch.zeros(1).expand_as(xs[0]).bool()
    connected_literals_ind = functools.reduce(operator.or_,xs,fbase).nonzero().view(-1)
    _, connected_literals = all_indices[:,connected_literals_ind].numpy()
    return list(set(connected_literals))

  def add_other_polarity(self, indices):
    indices = np.array(indices)
    pos = np.where(1-indices%2)[0]
    neg = np.where(indices%2)[0]
    add_pos = indices[pos] + 1
    add_neg = indices[neg] - 1
    return np.concatenate([add_pos,add_neg],axis=0).tolist()

  def __call__(self, sample):
    etypes = ['l2c', 'c2l']
    G = sample['graph']

    predicted_clauses = G.nodes['clause'].data['predicted_clauses']
    target_indices = predicted_clauses.nonzero().view(-1).tolist()
    # literals are computed at the env of even i, clauses at odd
    results = ([],[target_indices])
    # Do n half-rounds beginning with the predicted clauses
    for i in range(self.n):
      mat = G.adjacency_matrix(etype=etypes[i%2]).coalesce()
      target_indices = self.get_partition_neighbors(mat,target_indices)
      if i%2==0:    # For literals, we need to take care of opposite polarity
        extra_indices = self.add_other_polarity(target_indices)
        target_indices = list(set(target_indices+extra_indices))
      results[i%2].append(target_indices)    
    sub_literals = sorted(np.unique(np.concatenate(results[0])))
    sub_clauses = sorted(np.unique(np.concatenate(results[1])))

    new_G = G.subgraph({'literal': sub_literals, 'clause': sub_clauses})
    sample['graph'] = new_G
    sample['vnum'] = new_G.number_of_nodes('literal')
    sample['cnum'] = new_G.number_of_nodes('clause')

    # Re-compute predicted clauses
    z = torch.LongTensor([sub_clauses.index(x) for x in predicted_clauses.nonzero().view(-1).tolist()])
    new_predicted_clauses = torch.zeros(predicted_clauses.shape)
    new_predicted_clauses[z]=1
    sample['predicted_clauses'] = new_predicted_clauses

    # Shring gss
    gss = sample['gss'][0].view(-1)    # Edge case - one edge?
    sample['gss'] = gss.expand(sample['cnum'], len(gss))
    
    return sample

class SampleLearntClauses(object):
  def __init__(self,num, num_categories=2):
    self.num = num
    self.num_categories = num_categories
  def __call__(self, sample):
    G = sample['graph']
    tagged = (G.nodes['clause'].data['clause_targets'][:,0]).int()
    learnt = (G.nodes['clause'].data['clause_labels'][:,-1]).int()
    learnt_idx = torch.where(learnt)[0]
    tagged_idx = torch.where(tagged)[0]
    if len(tagged_idx) > 0:           # We use tagging
      assert len(tagged_idx) > self.num
      relevant_idx = torch.Tensor(list(set(tagged_idx.tolist()).intersection(learnt_idx.tolist()))).long()
    else:
      relevant_idx = learnt_idx
    labels = G.nodes['clause'].data['clause_effective_targets']
    relevant_labels = labels[relevant_idx]
    predicted_idx = []
    for i in range(self.num_categories):
      cat_idx = torch.where(relevant_labels==i)[0]    
      predicted_idx.append(relevant_idx[cat_idx[torch.torch.randperm(cat_idx.size(0))[:self.num]]])
    
    predicted_idx = torch.cat(predicted_idx,dim=0)
    predicted_arr = torch.zeros(labels.size(0))
    predicted_arr[predicted_idx] = 1
    G.nodes['clause'].data['predicted_clauses'] = predicted_arr
    return sample

class CapActivity(object):
  def __call__(self, sample):
    G = sample['graph']
    G.nodes['literal'].data['lit_labels'][:,2].tanh_()
    G.nodes['clause'].data['clause_labels'][:,3].tanh_()
    return sample
        
class ZeroClauseIndex(object):
  def __init__(self, index):
    self.index = index
  def __call__(self, sample):
    G = sample['graph']    
    G.nodes['clause'].data['clause_labels'][:,self.index] = 0
    return sample

class ZeroLiteralIndex(object):
  def __init__(self, index):
    self.index = index
  def __call__(self, sample):
    G = sample['graph']    
    G.nodes['literal'].data['lit_labels'][:,self.index] = 0
    return sample


class CnfGNNDataset(Dataset):
  def __init__(self, fname, transform=lambda x: x, cap_size=sys.maxsize):
    self.items = load_files(fname)    
    # self.cache = LRU(cache_size)
    self.transform = transform

  def __len__(self):
    return len(self.items)

  def __getitem__(self, idx):
    fname = self.items[idx]
    return self.transform(load_formula(fname))
    # if fname not in self.cache:     
    #   self.cache[fname] = self.transform(load_formula(fname))
    # return self.cache[fname]
