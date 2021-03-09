from eqnet_parser import *
from cnf_parser import *
import numpy as np
from functools import partial
from torch.utils.data import Dataset
from enum import Enum
import time
import torch
import re
import collections
import os
import random
import pdb
from IPython.core.debugger import Tracer

_use_shared_memory = False

class DataMode(Enum):
    NORMAL = 1
    SAT=2
    TRENERY=3
    TF=4

# We make the batch of sparse matrices into one huge sparse matrix

def cnf_collate(batch):
    rc = {}
    null_fields = ['sparse', 'num_vars', 'num_clauses']
    sample = batch[0]
    v_size = sample['num_vars']
    c_size = sample['num_clauses']
    for k in batch[0].keys():

        # We have a batch of sparse matrices, which we want to make into a big sparse matrix (along the 2nd dimension, no 3rd)

        # remap indices
        if k == 'sp_indices':
            rc_i = np.concatenate([b[k] + np.asarray([i*c_size,i*v_size]) for i,b in enumerate(batch)], 0)
        # just concatenate vals
        elif k == 'sp_vals':
            rc_v = np.concatenate([b[k] for b in batch], 0)
        elif k == 'samples':
            rc[k] = [b[k] for b in batch]
        elif k in null_fields:
            continue
        elif isinstance(sample[k],np.ndarray):
            rc[k] = torch.stack([torch.from_numpy(b[k]) for b in batch], 0)
        else:
            rc[k] = torch.from_numpy(np.asarray([b[k] for b in batch]))            


    if sample['sparse']:
        sp_ind_pos = torch.from_numpy(rc_i[np.where(rc_v>0)])
        sp_ind_neg = torch.from_numpy(rc_i[np.where(rc_v<0)])
        sp_val_pos = torch.ones(len(sp_ind_pos))
        sp_val_neg = torch.ones(len(sp_ind_neg))
        rc['sp_v2c_pos'] = torch.sparse.FloatTensor(sp_ind_pos.t(),sp_val_pos,torch.Size([c_size*len(batch),v_size*len(batch)]))
        rc['sp_v2c_neg'] = torch.sparse.FloatTensor(sp_ind_neg.t(),sp_val_neg,torch.Size([c_size*len(batch),v_size*len(batch)]))
    else:
        rc['sp_v2c_pos'] = None
        rc['sp_v2c_neg'] = None
    return rc


def filter_classes_by_ref(ref_dataset, classes, threshold, **kwargs):
    return {k: v for k,v in classes.items() if k in ref_dataset.labels}

def filter_classes(classes, threshold):
    a = {k: v for k,v in classes.items() if len(v) > threshold}
    m = np.mean([len(x) for x in a.values()])
    rc = a
    # rc = {k: v for k,v in a.items() if len(v) < 3*m}
    rc1 = {}
    for k,v in rc.items():
        v1 = [x for x in v if x['clauses_per_variable']]
        if len(v1) < len(v):
            print('removed empty %d formulas from key %s' % (len(v)-len(v1),k))
        rc[k] = v1
    return rc

def trenery_filter_classes(classes, threshold, **kwargs):
    a = {k: v for k,v in classes.items() if len(v) > threshold}
    m = np.mean([len(x) for x in a.values()])
    rc = {'Other': []}
    for k,v in a.items():
        v1 = [x for x in v if x['clauses_per_variable']]
        if len(v1) < len(v):
            print('removed empty %d formulas from key %s' % (len(v)-len(v1),k))
        if k in ['True', 'False']:
            rc[k] = v1
        else:
            rc['Other'] += v1
    return rc

def tf_filter_classes(classes, threshold, **kwargs):
    a = {k: v for k,v in classes.items() if len(v) > threshold}
    m = np.mean([len(x) for x in a.values()])
    rc = {}
    for k,v in a.items():
        v1 = [x for x in v if x['clauses_per_variable']]
        if len(v1) < len(v):
            print('removed empty %d formulas from key %s' % (len(v)-len(v1),k))
        if k in ['True', 'False']:
            rc[k] = v1            
    return rc

def sat_filter_classes(classes, threshold, max_size=100, **kwargs):    
    a = {k: v for k,v in classes.items() if len(v) > threshold}
    m = np.mean([len(x) for x in a.values()])
    rc = {'SAT': []}
    for k,v in a.items():
        # v1 = [x for x in v if len(x['clauses_per_variable']) < max_size and len(x['clauses']) < max_size]
        v1 = [x for x in v if len(x['clauses_per_variable']) < max_size]
        if len(v1) < len(v):
            print('removed %d formulas from key %s' % (len(v)-len(v1),k))
        if k in ['False']:
            rc[k] = v1
        else:
            rc['SAT'] += v1
    return rc


class CnfDataset(Dataset):    
    def __init__(self, classes, filter_fn = filter_classes, threshold=10, **kwargs):
        self.CLASS_THRESHOLD = threshold
        if 'num_max_clauses' in kwargs:
            self.num_max_clauses = kwargs['num_max_clauses']

        self.sparse = kwargs['sparse'] if 'sparse' in kwargs else False
        time1 = time.time()
        self.eq_classes = filter_fn(classes, threshold, **kwargs)
        time2 = time.time()
        print('After filtering, time difference was %f' % (time2-time1))

    
        # self.eq_classes = self.dummy_filter(to_cnf(load_bool_data(json_file)))
        self.labels = list(self.eq_classes.keys())
        self.samples = list(self.eq_classes.values())        
        self.class_size = [len(x) for x in self.samples]
        self.class_cumsize = np.cumsum(self.class_size) 
        self.cache = np.empty(sum(self.class_size),dtype=object)


    @classmethod
    def from_eqparser(cls, json_file, mode: DataMode=DataMode.NORMAL, ref_dataset=None, **kwargs):
        classes = to_cnf(load_bool_data(json_file))

        if mode == DataMode.TRENERY:
            func = trenery_filter_classes
        elif mode == DataMode.SAT:
            func = sat_filter_classes
        elif mode == DataMode.TF:
            func = tf_filter_classes
        elif mode == DataMode.NORMAL:
            if not ref_dataset:
                func = filter_classes
            else:
                func = partial(filter_classes_by_ref, ref_dataset)
        else:
            print('OK, thats not a known DataMode..')
            pdb.set_trace()

        return cls(classes, filter_fn=func, **kwargs)

    @classmethod
    def from_dimacs(cls, sat_dname, unsat_dname, **kwargs):
        sat = load_class(sat_dname)
        unsat = load_class(unsat_dname)
        classes = {'SAT': sat, 'False': unsat}

        return cls(classes, filter_fn = sat_filter_classes, **kwargs)

    def __len__(self):
        return sum(self.class_size)

    def __getitem__(self, idx):
        if self.cache[idx]:
            return self.cache[idx]
        i = np.where(self.class_cumsize > idx)[0][0]            # This is the equivalence class
        j = idx if i==0 else idx-self.class_cumsize[i-1]        # index inside equivalence class
        orig_sample = self.samples[i][j]        
        variables, clauses, topvar, sp_indices, sp_vals = self.transform_sample(orig_sample)
        self.cache[idx] = {'variables': variables, 'clauses': clauses, 'sp_indices': sp_indices, 'sp_vals': sp_vals, 'label': i, 'topvar': topvar, 
                            'idx_in_dataset': idx, 'samples': orig_sample, 'sparse': self.sparse, 'num_vars': self.max_variables,
                            'num_clauses': self.max_clauses}
        # self.cache[idx] = {'topvar': topvar}
        return self.cache[idx]
        

    @property
    def weights_vector(self):
        try:
            return self.__weights_vector
        except:
            pass

        rc = []
        a =[[1/x]*x for x in self.class_size]
        a = np.concatenate(a) / len(self.class_size)     # a now holds the relative size of classes
        self.__weights_vector = a
        return a


    def get_class_indices(self,c):
        if c==0:
            return range(self.class_cumsize[0])
        else:
            return range(self.class_cumsize[c-1],self.class_cumsize[c])


    def transform_sample(self,sample):
        clauses = sample['clauses_per_variable']
        auxvars = sample['auxvars']
        if sample['topvar'] is None:
            sample['topvar'] = 1
        topvar = sample['topvar']

        
        origvars = list(sample['origvars'].values())
        num_total_vars = len(origvars) + len(auxvars)
        

        def convert_var(v):
            j = abs(v)
            if j<=self.ground_vars: return v            
            if j in auxvars:
                newvar = self.ground_vars+auxvars.index(j)+1
                rc = newvar if v>0 else -newvar 
                if topvar < 0 and abs(v) == abs(topvar):                  # We invert the topvar variable if its nega
                    return -rc
                else:
                    return rc
                    
            else:
                print('What the heck?')
                import ipdb; Tracer()()

        
        all_clauses = [list(map(convert_var,x)) for x in sample['clauses']]

        if self.sparse:        
            indices = []
            values = []        
            for i,c in enumerate(all_clauses):
                for v in c:
                    val = 1 if v>0 else -1
                    v = abs(v)-1
                    indices.append(np.array([i,v]))
                    values.append(val)

            sp_indices = np.vstack(indices)
            sp_vals = np.stack(values)
            c2v = np.zeros(1)
            v2c = np.zeros(1)
        
        else:
            new_all_clauses = []        
            for i in range(self.max_clauses):
                new_clause = np.zeros(self.max_variables)
                if i<len(all_clauses):
                    x = all_clauses[i]
                    for j in range(self.max_variables):
                        t = j+1
                        if t in x:
                            new_clause[j]=1
                        elif -t in x:
                            # new_clause[j]=2
                            new_clause[j]=-1
                    new_all_clauses.append(new_clause)
                else:                
                    new_all_clauses.append(new_clause)
            if len(new_all_clauses) != self.max_clauses:
                import ipdb; Tracer()()

            sp_indices = np.zeros(1)
            sp_vals = np.zeros(1)
            v2c = np.stack(new_all_clauses)
            c2v = v2c.transpose()        
        return c2v, v2c, convert_var(sample['topvar']), sp_indices, sp_vals


    def dummy_filter(self, classes):
        return {'b': classes['b'], 'a': classes['a']}
    
    @property
    def ground_vars(self):
        try:
            return self.num_ground_vars
        except:
            pass
        rc = 0
        for x in self.samples:
            for sample in x:
                rc = max(rc,max(sample['origvars'].values()))
        self.num_ground_vars = rc
        return rc
    @property
    def max_clauses(self):
        try:
            return self.num_max_clauses
        except:
            pass
        rc = 0
        for x in self.samples:
            for sample in x:
                # rc = max(rc,max([len(x) for x in sample['clauses_per_variable'].values()]))
                rc = max(rc,len(sample['clauses']))
        self.num_max_clauses = rc
        return rc

    @property
    def max_variables(self):
        try:
            return self.num_max_variables
        except:
            pass
        rc = 0
        for x in self.samples:
            for sample in x:
                rc = max(rc,len(sample['auxvars']))                
        self.num_max_variables = rc + self.ground_vars
        return self.num_max_variables
        
    @property
    def num_classes(self):
        return len(self.labels)



class SiameseDataset(Dataset):    
    def __init__(self, json_file, pairwise_epochs=1, **kwargs):
        self.cnf_ds = CnfDataset(json_file, **kwargs)        
        self.pairs = []
        self.build_pairs(epochs=pairwise_epochs)

    ''' 
        For each epoch we sample N positive and N negative pairs from the internal CnfDataset
    '''

    def build_pairs(self, epochs):
        while epochs > 0:
            epochs -= 1
            for i in range(self.cnf_ds.num_classes):
                x = 0
                y = 0

                # sample two indices of same class
                while x == y:
                    x = random.choice(self.cnf_ds.get_class_indices(i))
                    y = random.choice(self.cnf_ds.get_class_indices(i))
                self.pairs.append({'left': self.cnf_ds[x], 'right': self.cnf_ds[y], 'label': 1})

                # sample two indices of different class
                x = random.choice(self.cnf_ds.get_class_indices(i))
                other_class = i
                while i == other_class:
                    other_class = random.choice(range(self.cnf_ds.num_classes))
                y = random.choice(self.cnf_ds.get_class_indices(other_class))
                self.pairs.append({'left': self.cnf_ds[x], 'right': self.cnf_ds[y], 'label': -1})

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]

    @property
    def num_classes(self):
        return 2


    @property
    def max_clauses(self):
        return self.cnf_ds.max_clauses

    @property
    def max_variables(self):
        return self.cnf_ds.max_variables

