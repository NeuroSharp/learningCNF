from tensorboard_logger import configure, log_value


class BaseLogger(object):
	def __init__(self):
		pass

	def log(tag,val,step):
		pass 

class TensorBoardLogger(BaseLogger):
	def __init__(self, fname):
		super(TensorBoardLogger, self).__init__()
		configure(fname, flush_secs=5)

	def log(tag,val,step):
		log_value(tag,val,step)

class SacredLogger(BaseLogger):
	def __init__(self, run):
		super(SacredLogger, self).__init__()
		self._run = run

	def log(tag,val,step):
		self._run.log_scalar(tag,val,step)

class ComposedLogger(BaseLogger):
	def __init__(self, loggers):
		super(ComposedLogger, self).__init__()
		self.loggers = loggers

	def log(tag,val,step):
		for l in self.logger:
			l.log(tag,val,step)

