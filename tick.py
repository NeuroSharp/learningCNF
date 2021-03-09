from utils import Singleton

class GlobalTick(metaclass=Singleton):
	def __init__(self):
		self.__tick_counter = 0

	def reset(self):
		self.__tick_counter = 0

	def tick(self):
		self.__tick_counter += 1

	def get_tick(self):
		return self.__tick_counter
