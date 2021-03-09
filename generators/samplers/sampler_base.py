import logging

import aiger as A
import aiger_cnf as ACNF

from gen_types import FileName
from cnf_tools import write_to_file

class SamplerBase:
    def __init__(self, dir=".", **kwargs):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s')
        self.log = logging.getLogger(__name__)
        self.dir = dir

    def write_expression(self, e, fname, is_cnf=False):
        if is_cnf:
            c = e
        else:
            c = ACNF.aig2cnf(e)             
        maxvar = max([max([abs(y) for y in x]) for x in c.clauses])
        write_to_file(maxvar, c.clauses, fname)

    # This returns a filename!
    def sample(self, stats_dict: dict) -> (FileName, FileName):
        raise NotImplementedError
