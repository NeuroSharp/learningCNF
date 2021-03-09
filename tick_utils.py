from IPython.core.debugger import Tracer
from tick import *
clock = GlobalTick()

def break_every_tick(n):
  t = clock.get_tick()
  if (t % n) == 0 and t > 0:
    Tracer()()

def every_tick(n):
  t = clock.get_tick()
  return ((t % n) == 0 and t > 0)


def get_tick():
	return clock.get_tick()