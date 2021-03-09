from settings import *
from utils import Singleton

class AbstractProvider(object):
  def notify(self, *args, **kwargs):
    pass

class ObserverDispatcher(metaclass=Singleton):
  def __init__(self, settings=None):
    if settings != None:
      self.settings = settings
    else:
      self.settings = CnfSettings()
    self.event_dict = {}

  def register(self, event_name, provider):
    if not event_name in self.event_dict.keys():
      self.event_dict[event_name] = []
    self.event_dict[event_name].append(provider)

  def notify(self, event_name, *args, **kwargs):
    if not event_name in self.event_dict.keys():
      return
    for provider in self.event_dict[event_name]:
      provider.notify(*args, **kwargs)
