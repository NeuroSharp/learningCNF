from pysat.formula import CNF
from pysat._fileio import FileObject
import logging
from settings import *
logging.basicConfig(level=logging.DEBUG)

class FormulaCache(object):
	def __init__(self, settings=None):
		if settings:
			self.settings = settings
		else:
			self.settings = CnfSettings()
		self.preload_cnf = self.settings['preload_cnf']
		self.formulas_dict = {}
		self.logger = logging.getLogger('formula_cache')
		self.logger.setLevel(eval(self.settings['loglevel']))

	def load_files(self,flist):
		for fname in flist:
			with FileObject(fname, mode='r', compression='use_ext') as fobj:
				formula_str = fobj.fp.read()				
				self.formulas_dict[fname] = CNF(from_string=formula_str) if self.preload_cnf else formula_str
				self.logger.debug('loaded {}'.format(fname))

	def load_formula(self,fname):
		if fname not in self.formulas_dict.keys():
			self.logger.debug('Loading {} in runtime!'.format(fname))
			self.load_files([fname])
		formula = self.formulas_dict[fname]
		rc = formula if self.preload_cnf else CNF(from_string=formula)
		return rc

	def delete_key(self, fname):
		self.logger.debug('Asked to delete {} from cache'.format(fname))
		try:
			del self.formulas_dict[fname]
		except KeyError:
			self.logger.warning('delete_key: formula {} is not in cache!'.format(fname))
