import random

class Rule:
    def __init__(self, rule_id):
        self.rule_id = rule_id

        clauses = set()

        for i in range(8):
            output = (rule_id >> i) & 1
            in_l = (i >> 2) & 1
            in_c = (i >> 1) & 1
            in_r = (i >> 0) & 1

            clauses.add((not in_l, not in_c, not in_r, output))

        minimal_clauses = set()

        while clauses:
            next_clauses = set()
            for clause in clauses:
                is_minimal = True
                for i, c_i in enumerate(clause):
                    if c_i != 2:
                        flipped = list(clause)
                        flipped[i] = not c_i
                        if tuple(flipped) in clauses:
                            flipped[i] = 2
                            next_clauses.add(tuple(flipped))
                            is_minimal = False
                if is_minimal:
                    minimal_clauses.add(clause)
            clauses = next_clauses

        self.template = sorted(minimal_clauses)

    def clauses(self, *literals):
        for clause in self.template:
            yield [
                [-lit, lit][pol]
                for pol, lit in zip(clause, literals)
                if pol != 2
            ]

    def apply_cell(self, l, c, r):
        return bool((self.rule_id >> ((l << 2) + (c << 1) + (r << 0))) & 1)

    def apply(self, state):
        size = len(state)
        return [
            self.apply_cell(
                state[(i - 1) % size],
                state[i],
                state[(i + 1) % size]
            ) for i in range(size)
        ]


class Benchmark:
    def __init__(self, rule, state_size, fwd_steps, rev_steps, seed):
        self.rule = Rule(rule)
        self.state_size = state_size
        self.fwd_steps = fwd_steps
        self.rev_steps = rev_steps
        self.seed = seed

        self.variable_count = 0
        self.clauses = []

        final_state_values = self.generate_final_state()
        final_state_variables = self.generate_transitions()

        for value, variable in zip(final_state_values, final_state_variables):
            if value:
                self.clauses.append([variable])
            else:
                self.clauses.append([-variable])

    @property
    def name(self):
        return 'ecarev-%i-%i-%i-%i-%i' % (
            self.rule.rule_id, self.state_size, self.fwd_steps,
            self.rev_steps, self.seed
        )

    def make_variable(self):
        self.variable_count += 1
        return self.variable_count

    def generate_final_state(self):
        rng = random.Random()
        rng.seed(self.name, version=2) # make sure its random

        state = [rng.getrandbits(1) for x in range(self.state_size)]

        for t in range(self.fwd_steps):
            state = self.rule.apply(state)

        return state

    def generate_transitions(self):
        state = [self.make_variable() for x in range(self.state_size)]

        for t in range(self.rev_steps):
            next_state = [self.make_variable() for x in range(self.state_size)]

            for x in range(self.state_size):
                pl = state[(x - 1) % self.state_size]
                pc = state[x]
                pr = state[(x + 1) % self.state_size]
                c = next_state[x]

                self.clauses.extend(self.rule.clauses(pl, pc, pr, c))

            state = next_state

        return state

    def print_dimacs(self, file=None):
        print('c', self.name, file=file)
        print('p cnf', self.variable_count, len(self.clauses), file=file)
        for clause in self.clauses:
            print(' '.join(map(str, clause + [0])), file=file)
