from collections import namedtuple
from namedlist import namedlist
import numpy as np

import utils

DiscreteParam = namedtuple('DiscreteParam', ['name', 'values'])
IntervalParam = namedtuple('IntervalParam', ['name', 'start', 'stop', 'step'])


# Expand interval, basically arange

def interval_to_discrete(interval):
  return DiscreteParam(interval.name,list(np.arange(interval.start, interval.stop, interval.step)))

class GridParams(object):
  def __init__(self, fname=None):
    if fname is None:
      self.all_params = []
    else:
      self.load_file(fname)

  def load_file(self, fname):
    with open(fname,'r') as f:
      self.all_params = [eval(x.strip()) for x in f]

  def save_file(self, fname):
    with open(fname,'w') as f:
      for param in self.all_params:
        f.write("%s\n" % str(param))

  def grid_dict(self):
    rc = {}
    for param in self.all_params:
      if type(param) is IntervalParam:
        param = interval_to_discrete(param)
      rc[param.name] = param.values

    return rc

  def get_all_config(self):
    return list(utils.dict_product(self.grid_dict()))

