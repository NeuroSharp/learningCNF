import torch
import time
import itertools
import logging
import numpy as np
import random
import string
from torch.autograd import Variable
import pandas as pd
import scipy.signal
import getopt
import sys
from pprint import pprint
import functools
import weakref
import signal
import os
import pdb

from enum import IntEnum


class BaseMode(IntEnum):
    ALL = 1
    EMBEDDING = 2
    ITERS = 3


def pprint_vectors(vecs_dict):
    x = pd.DataFrame(vecs_dict)
    pd.options.display.max_rows = 2000
    print(x)

def where(cond, x_1, x_2):
    cond = cond.float()
    return (cond * x_1) + ((1-cond) * x_2)

def copy_model_weights(model1, model2):
    model2.load_state_dict(model1.state_dict())

def memoized_method(*lru_args, **lru_kwargs):
    def decorator(func):
        @functools.wraps(func)
        def wrapped_func(self, *args, **kwargs):
            # We're storing the wrapped method inside the instance. If we had
            # a strong reference to self the instance would never die.
            self_weak = weakref.ref(self)
            @functools.wraps(func)
            @functools.lru_cache(*lru_args, **lru_kwargs)
            def cached_method(*args, **kwargs):
                return func(self_weak(), *args, **kwargs)
            setattr(self, func.__name__, cached_method)
            return cached_method(*args, **kwargs)
        return wrapped_func
    return decorator

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


def convert_var(v, reverse=False):
    s = sign(v)
    val = abs(v)

def log_name(settings):
    return 'exp_{}'.format(settings['name'])

# Get a distribution p, return (1-eps)*p + eps*U
def epsilonize(p, eps):
    n = len(p)
    U = np.ones(n) / n
    return (1-eps)*p + eps*U


def flatten(l):
    return [i for x in l for i in x]

def normalize(input, p=2, dim=1, eps=1e-20):
    return input / input.norm(p, dim).clamp(min=eps).expand_as(input)

def formula_to_input(formula):
    try:
        return [[Variable(x, requires_grad=False) for x in y] for y in formula]
    except:
        return [[Variable(torch.LongTensor([x]), requires_grad=False) for x in y] for y in formula]


def permute_seq(inp):
    # inp is a sequence of tensors, we return a random permutation

    p = list(itertools.permutations(inp))
    i = np.random.randint(len(p))
    return list(p[i])


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=7, decay_rate=0.1):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * ((1-decay_rate)**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0 and epoch > 0:
        print('LR is set to {}'.format(lr))
        set_lr(optimizer,lr)

    return optimizer

def discount(x, gamma):
    """
    Compute discounted sum of future values
    out[i] = in[i] + gamma * in[i+1] + gamma^2 * in[i+2] + ...
    """
    return scipy.signal.lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1]


def normalize_weights(x, make_unit=False):
    rc = x.copy()
    for i in set(x):
        l = len(x[x==i])
        rc[x==i]=l
    rc = 1/rc
    if make_unit:
        rc = rc/sum(rc)

    return rc


def cleanup_process(pid):
    try:
        os.kill(pid,signal.SIGKILL)
    except:
        pass
    os.waitpid(pid, 0)

# cross-product

def dict_product(dicts):
    """
    >>> list(dict_product(dict(number=[1,2], character='ab')))
    [{'character': 'a', 'number': 1},
     {'character': 'a', 'number': 2},
     {'character': 'b', 'number': 1},
     {'character': 'b', 'number': 2}]
    """
    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))


class EnvIdGen(metaclass=Singleton):
    def __init__(self, initial_id=10000):
        self.initial_id = initial_id

    def get_id(self):
        rc = self.initial_id
        self.initial_id += 1
        return rc

class ForkablePdb(pdb.Pdb):

    _original_stdin_fd = sys.stdin.fileno()
    _original_stdin = None

    def __init__(self):
        pdb.Pdb.__init__(self, nosigint=True)

    def _cmdloop(self):
        current_stdin = sys.stdin
        try:
            if not self._original_stdin:
                self._original_stdin = os.fdopen(self._original_stdin_fd)
            sys.stdin = self._original_stdin
            self.cmdloop()
        finally:
            sys.stdin = current_stdin


def set_proc_name(newname):
    try:
        import setproctitle
        setproctitle.setproctitle(str(newname))
        return
    except:
        pass
    from ctypes import cdll, byref, create_string_buffer
    libc = cdll.LoadLibrary('libc.so.6')
    buff = create_string_buffer(len(newname)+1)
    buff.value = newname
    libc.prctl(15, byref(buff), 0, 0, 0)

def get_proc_name():
    from ctypes import cdll, byref, create_string_buffer
    libc = cdll.LoadLibrary('libc.so.6')
    buff = create_string_buffer(128)
    # 16 == PR_GET_NAME from <linux/prctl.h>
    libc.prctl(16, byref(buff), 0, 0, 0)
    return buff.value

# We return either one sparse matrix (P+N) or (|P|,|N|) if split=True
def csr_to_pytorch(m, split=False, size=None):
    if not size:
        size = m.shape
    if not split:
        np_ind = m.nonzero()
        indices = torch.from_numpy(np.stack(np_ind)).long()
        vals = torch.from_numpy(m[np_ind]).float().squeeze()
        return torch.sparse.FloatTensor(indices,vals,size)
    else:
        indices = torch.from_numpy(np.stack((m==1).nonzero())).long()
        vals = torch.ones(indices.shape[1])
        pos_rc = torch.sparse.FloatTensor(indices,vals,size)
        indices = torch.from_numpy(np.stack((m==-1).nonzero())).long()
        vals = torch.ones(indices.shape[1])
        neg_rc = torch.sparse.FloatTensor(indices,vals,size)
        return pos_rc, neg_rc


# Indices and vals are numpy arrays. Indices is Rx2
# Return value is pytorch sparse matrix

def create_sparse_adjacency(indices, vals, size, split=False):
    if not split:
        return torch.sparse.FloatTensor(torch.from_numpy(indices.transpose()),torch.from_numpy(vals),size)
    else:
        sp_ind_pos = torch.from_numpy(indices[np.where(vals>0)])
        sp_ind_neg = torch.from_numpy(indices[np.where(vals<0)])
        sp_val_pos = torch.ones(len(sp_ind_pos))
        sp_val_neg = torch.ones(len(sp_ind_neg))
        cmat_pos = torch.sparse.FloatTensor(sp_ind_pos.t(),sp_val_pos,size)
        cmat_neg = torch.sparse.FloatTensor(sp_ind_neg.t(),sp_val_neg,size)
        return cmat_pos, cmat_neg

# input is existing cmat in pytorch
# output is (cmat_pos, cmat_neg), such that cmat_pos-cmat_neg=cmat

def split_sparse_adjacency(cmat):
    ind = cmat._indices()
    vals = cmat._values()
    size = cmat.shape
    posind = torch.stack([ind[0].masked_select(vals>0),ind[1].masked_select(vals>0)])
    negind = torch.stack([ind[0].masked_select(vals<0),ind[1].masked_select(vals<0)])
    posvals = vals.masked_select(vals>0)
    negvals = -vals.masked_select(vals<0)
    cmat_pos = torch.sparse.FloatTensor(posind,posvals,size)
    cmat_neg = torch.sparse.FloatTensor(negind,negvals,size)
    return cmat_pos, cmat_neg

# This function takes two 2d sparse tensors and concatenates them along dim 0

def concat_sparse(t1,t2):
    dim0 = t1.shape[0]+t2.shape[0]
    dim1 = max(t1.shape[1],t2.shape[1])
    size = (dim0,dim1)
    ind1 = t1._indices()
    ind2 = t2._indices()
    vals1 = t1._values()
    vals2 = t2._values()
    vals = torch.cat([vals1,vals2])
    offset = torch.from_numpy(np.array([t1.shape[0],0])).unsqueeze(1)
    new_ind2 = ind2+offset
    new_ind = torch.cat([ind1,new_ind2],1)
    rc = torch.sparse.FloatTensor(new_ind,vals,size)

    return rc
def make_one_hot(labels, C=2):
    '''
    Converts an integer label torch.autograd.Variable to a one-hot Variable.

    Parameters
    ----------
    labels : Tensor sized N
    C : integer.
        number of classes in labels.

    Returns
    -------
    Tensor sizes NxC
    '''

    y = torch.eye(C)
    return y[labels]

def unsig(y):
    return -np.log(1/y-1)

def statedict_to_numpy(state_dict):
    z = {}
    for k in state_dict.keys():
        z[k] = state_dict[k].numpy()
    return z

def numpy_into_statedict(state_dict, np_dict):
    for k in state_dict.keys():
        state_dict[k] = torch.from_numpy(np_dict[k])

def get_logger(settings, logger_name, filename=None):
    logger = logging.getLogger(logger_name)
    logger.setLevel(eval(settings['loglevel']))
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s',
                                    '%Y-%m-%d %H:%M:%S')
    if filename:
        fh = logging.FileHandler(filename, mode='w')
        fh.setFormatter(formatter)
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)
    # ch = logging.StreamHandler()
    # ch.setFormatter(formatter)
    # logger.addHandler(ch)

    return logger

def seed_all(settings, name):
    if settings['use_seed'] is None:
      np.random.seed(int(time.time())+abs(hash(name)) % 1000)
      torch.manual_seed(int(time.time())+abs(hash(name)) % 1000)
    else:
      np.random.seed(settings['use_seed'])
      torch.manual_seed(settings['use_seed'])

def load_dir(directory):
  return load_files([os.path.join(directory, f) for f in os.listdir(directory)])

def load_files(files):
  if type(files) is not list:
    files = [files]
  only_files = [x for x in files if os.path.isfile(x)]
  only_dirs = [x for x in files if os.path.isdir(x)]
  return only_files if not only_dirs else only_files + list(itertools.chain.from_iterable([load_dir(x) for x in only_dirs]))

def repeat_end(val, n, k):
  return [val for i in range(n)] + [k]

class OneTimeSwitch(object):
  def __init__(self, val=True, default=False):
    self.val = val
    self.default = default

  def get_val(self):
    rc = self.val
    self.val = self.default
    return rc

def patch_grid(grid, indices, vals):
  np.put(grid,np.ravel_multi_index(indices.T,grid.shape),vals)


# indices is array of dimension 3 indices, grid is dimension 4, where 4th dimension is embedding
# return indices.shape[0] x grid.shape[-1]
def get_from_grid(grid, indices):
  embdim = grid.shape[-1]
  ind = np.ravel_multi_index(indices.T,grid.shape[:-1])
  a = grid.reshape(-1, embdim)
  return a[ind,:]

