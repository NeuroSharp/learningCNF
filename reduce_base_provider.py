from IPython.core.debugger import Tracer
import random
import logging
import random

from settings import *
from dispatcher import *

class AbstractReduceBaseProvider(AbstractProvider):
  def __init__(self,settings=None):
    if settings is None:
      self.settings = CnfSettings()
    else:
      self.settings = settings
    self.logger = logging.getLogger('reduce_base_provider')
    self.logger.setLevel(eval(self.settings['loglevel']))

  def get_reduce_base(self):
    pass

class FixedReduceBaseProvider(AbstractReduceBaseProvider):
  def __init__(self, settings):
    super(FixedReduceBaseProvider, self).__init__(settings)
    self.reduce_base = self.settings['sat_reduce_base']

  def get_reduce_base(self):
    return self.reduce_base

class RandomReduceBaseProvider(AbstractReduceBaseProvider):
  def __init__(self, settings):
    super(UniformReduceBaseProvider, self).__init__(settings)
    self.reduce_base = self.settings['sat_reduce_base']
    ObserverDispatcher().register('new_batch',self)

  def sample_reduce_base(self):
    pass

  def notify(self, *args, **kwargs):
    self.sample_reduce_base()

  def get_reduce_base(self):
    return self.reduce_base
    
class UniformReduceBaseProvider(RandomReduceBaseProvider):
  def __init__(self, settings):
    super(UniformReduceBaseProvider, self).__init__(settings)
    self.rb_min = self.settings['sat_rb_min']
    self.rb_max = self.settings['sat_rb_max']

  def sample_reduce_base(self):
    self.reduce_base = random.randrange(self.rb_min,self.rb_max)
    
