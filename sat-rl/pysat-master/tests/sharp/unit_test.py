import unittest

import sys
import random
import numpy as np

from pysat.solvers import SharpSAT
from pysat.formula import CNF

random.seed(42)
np.set_printoptions(suppress=True, threshold=sys.maxsize)

class SharpSATUnit(unittest.TestCase):
    def toInt(self, val):
        sign = 1 if (val & 0x01) else -1
        return (val >> 1) * sign

    def sharpSAT_cb(self, row, col, data, labels, lit_stack):
        ind = labels['var_score'].idxmax()
        pick = ind + 1 if (labels['activity'][ind] < labels['activity'][ind + 1]) else ind
        return pick

    def test_mimic_sharpSAT(self):
        self.solver = SharpSAT(branching_oracle = {"branching_cb": self.sharpSAT_cb})
        self.solver.solve("datasets/sharp/uf20-91/uf20-0208.cnf")
        observed_seq = self.solver.get_branching_seq()
        observed_reward = self.solver.reward()
        self.solver.delete()

        self.solver = SharpSAT()
        self.solver.solve("datasets/sharp/uf20-91/uf20-0208.cnf")
        expected_seq = self.solver.get_branching_seq()
        expected_reward = self.solver.reward()
        self.solver.delete()

        self.assertTrue(np.array_equal(expected_seq, observed_seq),
            f"""Mismatch between the branching literals:
            Expected: {expected_seq}
            Got     : {observed_seq}""")

        self.assertTrue(np.array_equal(expected_reward, observed_reward),
            f"""Mismatch between the rewards:
            Expected: {expected_reward}
            Got     : {observed_reward}""")

###############################################################################
    def termination_cb(self, *args):
        self.counter += 1
        self.reward = self.solver.reward()
        if (self.counter == self.counter_max): self.solver.terminate()

        return 0

    def test_termination(self):
        self.counter = 0
        self.reward = 0
        self.counter_max = 3

        self.solver = SharpSAT(branching_oracle= {"branching_cb": self.termination_cb})
        self.solver.solve("datasets/sharp/uf20-91/uf20-0208.cnf")

        self.assertEqual(self.solver.reward(), self.reward,
            "Asynchronous termination! The Solver did not quit right after terminate() call.")

        self.assertEqual(self.counter, self.counter_max,
            """Asynchronous termination! The Solver did not quit right after terminate() call.
            More callback calls than counter_max""")


###############################################################################
    def cnf_to_cvig_cb(self, row, col, data):
        a = {}
        adj =list(zip(row, [self.toInt(c_i + 2) for c_i in col]))
        for item in adj:
            a[item[0]] = a.get(item[0], "") + f"{item[1]} "

        lines_sharp = [''.join(sorted(cla)).strip() for cla in a.values()]
        lines_sharp.sort()

        self.assertEqual(lines_sharp, self.lines_cnf,
            "There's a mismatch between the clauses in the cnf file and what callback has received.")

        self.solver.terminate()
        return -1

    def test_cvig(self):
        filename = "datasets/sharp/uf20-91/uf20-0208.cnf"
        self.lines_cnf = []
        with open(filename) as file_in:
            for line in file_in:
                if (line.startswith("c") or line.startswith("p")): continue

                self.lines_cnf.append(''.join(sorted(line.rstrip("0\n"))).strip())

            self.lines_cnf.sort()

        self.solver = SharpSAT(branching_oracle= {"branching_cb": self.cnf_to_cvig_cb, "test_mode": True})
        self.solver.solve(filename)
        self.lines_cnf = []


###############################################################################
    def random_cb(self, row, col, data, labels, lit_stack):
        self.counter += 1
        self.assertLessEqual(self.counter, 1000,
            "Callback is called too many times. Seems that the solver is stuck in an infinite loop.")
        return random.randint(0, len(labels) - 1)

    def test_valid_actions(self):
        """
        This test is to make sure that all actions (branching vars) result in the
        eventual termination of the solver. Solver should never get stuck in infinite
        loop, which could happen if a variable is selected that does not belong to the
        current component. This test makes sure that all the variables that are passed to
        the callback can be branched on.

        This test runs 100 times with random branchings to test this.
        """
        for i in range(100):
            self.counter = 0
            self.solver = SharpSAT(branching_oracle = {"branching_cb": self.random_cb})
            self.solver.solve("datasets/sharp/uf20-91/uf20-0208.cnf")


###############################################################################
    def actions_complement_cb(self, row, col, data, labels, lit_stack):
        self.counter += 1

        cvig_lits = np.unique(col)

        r = random.randint(0, len(labels) - 1)
        # finding the litreal whose complement does not appear in the cvig_lits (unitary literal)
        for i in range(0, len(cvig_lits)):
            if (int(cvig_lits[i] % 2) == 0):
                if (i == len(cvig_lits) - 1): # list should not end with an even literal
                    r = i
                    self.found = True
                    break
                if (cvig_lits[i] + 1 != cvig_lits[i + 1]):
                    r = i
                    self.found = True
                    break
            else:
                if (i == 0): # list should not start with odd literal
                    r = i
                    self.found = True
                    break
                if (cvig_lits[i] - 1 != cvig_lits[i - 1]):
                    r = i
                    self.found = True
                    break

        self.assertLessEqual(self.counter, 1000,
            "Callback is called too many times. Seems that the solver is stuck in an infinite loop.")
        return r

    def test_valid_action_complement(self):
        """
        This test is to make sure that all actions (branching vars) result in the
        eventual termination of the solver. Solver should never get stuck in infinite
        loop, which could happen if a variable is selected that does not belong to the
        current component. This test makes sure that all the variables that are passed to
        the callback can be branched on (including the complement of literals, that the complement
        is not in the CVIG).

        This test runs 100 times with random branchings to test this.
        """
        for i in range(100):
            self.found = False
            self.counter = 0
            self.solver = SharpSAT(branching_oracle = {"branching_cb": self.actions_complement_cb,
                "include_orig_lit_id": True})
            self.solver.solve("datasets/sharp/uf20-91/uf20-0208-2.cnf")
            self.assertTrue(self.found, "This test should find at least one unitary literal in this cnf.")

###############################################################################
    def var_label_cb(self, row, col, data, labels, lit_stack):
        self.assertLessEqual(len(labels) % 2, 0,
            "Literal Labels array should have an even number of rows.")
        match = True
        # print("lit_stack", lit_stack)
        # print("id_dimacs", labels['id_dimacs'])
        for i in range(0,len(labels), 2):
            if (labels['id_dimacs'][i] != -labels['id_dimacs'][i + 1]):
                match = False
                # print(labels['id_dimacs'][i], -labels['id_dimacs'][i + 1])
            if (labels['id'][i] + 1 != labels['id'][i + 1]):
                match = False
                print(f"{labels['id'][i]}:{labels['id'][i + 1]}")
            if (labels['var_score'][i] != labels['var_score'][i + 1]):
                match = False
            if (int(labels['var_score'][i] / 10) * 10 < (labels['activity'][i] + labels['activity'][i + 1]) * 10):
                match = False

        self.assertEqual(match, True, "LitLabel array has incorrect values.")
        return -1

    def test_var_labels(self):
        """
        This test checks to see if for every variable, its complement is also part of the
        lits_label.
        """
        self.solver = SharpSAT(branching_oracle = {"branching_cb": self.var_label_cb})
        self.solver.solve("datasets/sharp/uf20-91/uf20-0208.cnf")

###############################################################################
    def lit_stack_cb(self, row, col, data, labels, lit_stack):
        indices = labels['id_dimacs']
        overlap = np.intersect1d(indices, lit_stack)

        self.assertEqual(len(overlap), 0, "LitLabel array has incorrect values.")

        return -1

    def test_lit_stack(self):
        """
        An insufieicnt test for the correctness of the lit_stack. The litstack should only
        contain literals that are NOT present in the current component. This test checks that.
        """
        self.solver = SharpSAT(branching_oracle = {"branching_cb": self.lit_stack_cb})
        self.solver.solve("datasets/sharp/uf20-91/uf20-0208.cnf")

###############################################################################
    def units_cb(self, row, col, data, labels, lit_stack):
        obs_units = self.solver.get_problem_units().tolist()
        intersect = np.intersect1d(obs_units, self.units)

        self.assertEqual(np.array_equal(intersect, self.units), True, "The initial literal stack does not fully contain the problem units")

        return -1

    def test_get_units(self):
        """
        Testing the get_problem_units(). The output of get_problem_units should contain the units
        in the cnf, as well as all their implications. This method just tests the first assertion.
        """
        cnf_file = "datasets/sharp/sudoku/sudoku-9x9-25-A383KWeH.cnf"
        cnf = CNF()
        cnf.from_file(cnf_file)
        self.units = [unit[0] for unit in cnf.clauses if len(unit)==1]

        self.solver = SharpSAT(branching_oracle = {"branching_cb": self.units_cb})
        self.solver.solve(cnf_file)

###############################################################################
    def empty_cb(*args):
        return -1

    def test_empty_callback(self):
        """
        This test checks to see if the sequence of branching variables is the same between
        the empty callback (just returning -1) and SharpSAT's default...
        """
        self.solver = SharpSAT(branching_oracle = {"branching_cb": self.empty_cb})
        self.solver.solve("datasets/sharp/uf20-91/uf20-0208.cnf")
        observed_seq = self.solver.get_branching_seq()
        observed_reward = self.solver.reward()
        self.solver.delete()

        self.solver = SharpSAT()
        self.solver.solve("datasets/sharp/uf20-91/uf20-0208.cnf")
        expected_seq = self.solver.get_branching_seq()
        expected_reward = self.solver.reward()
        self.solver.delete()

        self.assertTrue(np.array_equal(expected_seq, observed_seq),
            f"""Mismatch between the branching literals:
            Expected: {expected_seq}
            Got     : {observed_seq}""")

        self.assertTrue(np.array_equal(expected_reward, observed_reward),
            f"""Mismatch between the rewards:
            Expected: {expected_reward}
            Got     : {observed_reward}""")

    def tearDown(self):
        self.solver.delete()



if __name__ == '__main__':
    unittest.main()
