import unittest

import sys
import random
import logging
import numpy as np
from scipy.sparse import csr_matrix

from pysat.solvers import SharpSAT
from pysat.formula import CNF

from samplers.sudoku_sampler import *


class SudokuUnit(unittest.TestCase):
    def test_program_units(self):
        print("Testing test_program_units")
        cnf_file = "/tmp/sudoku_sanity_check.cnf"

        puzzle = SudokuSampler(newspaper=True).sample_board()
        puzzle.encode().print_dimacs(cnf_file)

        sharp = SharpSAT()
        count = sharp.solve(cnf_file)
        implications = sharp.get_problem_units()
        sharp.delete()

        cnf = CNF()
        cnf.from_file(cnf_file)
        units = [unit[0] for unit in cnf.clauses if len(unit)==1]

        solved_puzzle  = puzzle.solve()
        current_puzzle = puzzle.fit(implications).solve()

        self.assertGreater(len(implications), len(units), "SharpSAT should produce more implication units from the problem units.")
        self.assertGreater(len(implications), 0, "The set of initial unit clauses is empty.")
        self.assertEqual(count, 1, "Something's wrong!")
        self.assertNotEqual(current_puzzle, None, "The problem units and their implications caused conflict!")
        self.assertEqual(solved_puzzle, current_puzzle, "there are more than one solutions.")

        print("done!")

###############################################################################
    def units_cb(self, row, col, data, label, lit_stack):
        print(".", end="", flush=True)

        units = self.solver.get_problem_units()
        partial = np.concatenate([units,lit_stack])
        empty_board = np.full((9, 9), -1, dtype=int)
        current_puzzle = SudokuCNF(3, empty_board)
        solved = current_puzzle.fit_partial(partial).solve()

        self.assertNotEqual(solved, None, "trail caused conflict!")
        return -1

    def test_lit_stack(self):
        print("Testing test_lit_stack")
        cnf_file = "sat-rl/pysat-master/datasets/sharp/sudoku/sudoku-9x9-25-A383KWeH.cnf"
        cnf = CNF()
        cnf.from_file(cnf_file)

        self.solver = SharpSAT(branching_oracle = {"branching_cb": self.units_cb})
        self.solver.solve(cnf_file)
        self.solver.delete()

        print("done!")

if __name__ == '__main__':
    unittest.main()
