#!/usr/bin/env python3

import os
import sys
import argparse
import inspect
import tempfile
import git
from IPython.core.debugger import Tracer
import ipdb
from random import randint, seed
from aigerbv.aigbv import AIGBV
from aigerbv.expr import SignedBVExpr, atom
from cadet_cmdline_utils import eval_formula
import aiger_analysis as aa

# word_length = 8

# arith_ops = [SignedBVExpr.__add__, SignedBVExpr.__sub__]
# bitwise_ops = [SignedBVExpr.__and__, SignedBVExpr.__or__, SignedBVExpr.__xor__]
# unary_ops = [SignedBVExpr.__invert__, SignedBVExpr.__abs__, SignedBVExpr.__neg__]
# cmp_ops = [SignedBVExpr.__eq__, SignedBVExpr.__ne__,
#            SignedBVExpr.__lt__, SignedBVExpr.__le__, SignedBVExpr.__gt__, SignedBVExpr.__ge__]

# variables = None


# def variable():
#     # variables[randint(0, len(variables) - 1)]
#     global variables
#     return atom(word_length, variables.pop(), signed=True)


# def constant_expr():
#     if randint(0, 4) == 0:
#         c = atom(word_length, randint(- 2**(word_length-1), 2**(word_length-1) - 1), signed=True)
#     else: 
#         c = atom(word_length, 1, signed=True)
#     return c


# def leaf_expr(size):  # constant or variable
#     assert size == 1
#     return variable() if randint(0, 1) == 0 else constant_expr()


# def unary_expr(size):
#     assert size > 1
#     arg = random_expr(size - 1)
#     op = unary_ops[randint(0, len(unary_ops) - 1)]
#     return op(arg)


# def arithmetic_expr(size):
#     assert size > 2
#     operator = arith_ops[randint(0, len(arith_ops)-1)]
#     # arg_num = len(inspect.getargspec(operator)[0])  # arity of the function
#     split = randint(1, size - 2)  # at least one operation on either side
#     left = random_expr(split)
#     right = random_expr(size - split - 1)
#     return operator(left, right)


# def random_expr(size):
#     if size <= 1:
#         return leaf_expr(size)
#     if size == 2:
#         return unary_expr(size)
#     if randint(0, 2) == 0:
#         return bitwise_expr(size)
#     else:
#         return arithmetic_expr(size)


# def bitwise_expr(size):
#     assert size > 2
#     op = bitwise_ops[randint(0, len(bitwise_ops)-1)]
#     split = randint(1, size - 2)  # at least one operation on either side
#     left = random_expr(split)
#     right = random_expr(size - split - 1)
#     return op(left, right)

# def random_bool_expr(size):
#     global variables
#     variables = ['2 y1', '1 x1', '2 y2', '1 x2']

#     assert size > 2
#     op = cmp_ops[randint(0, len(cmp_ops)-1)]
#     split = randint(1, size - 2)  # at least one operation on either side
#     left = random_expr(split)
#     right = random_expr(size - split - 1)
#     return op(left, right)


# def random_circuit(size):
#     while True:
#         e = random_bool_expr(size)
#         e = aa.simplify(e)
#         if e is not None:
#             return e
#         else: 
#             print('    Failed to generate expression; trying again')


def parse_cmdline():
    print('')
    p = argparse.ArgumentParser()
    p.add_argument('-s', '--seed', dest='seed', action='store',
                   nargs='?', default=None, type=int, metavar='S',
                   help='Seed for the PNG. Uses fresh seed every '
                   'run per default.')
    p.add_argument('--max_hardness', dest='max_hardness', action='store',
                   nargs='?', default=None, type=int, metavar='H',
                   help='The maximal average number of decisions required '
                   'to solve the problem (default None).')
    p.add_argument('--min_hardness', dest='min_hardness', action='store',
                   nargs='?', default=1, type=int, metavar='h',
                   help='The minimal average number of decisions required'
                   'to solve the problem.')
    # p.add_argument('--maxvars', dest='maxvars', action='store',
    #                nargs='?', default=50, type=int, metavar='V',
    #                help='The maximal number of variables (default 50).')
    p.add_argument('-n', '--number', dest='num_generated', action='store',
                   nargs='?', default=1, type=int, metavar='N',
                   help='Number of files to be generated.')
    p.add_argument('-r', '--repetitions', dest='repetitions', action='store',
                   nargs='?', default=1, type=int, metavar='R',
                   help='Number of runs of CADET to compute average decisions.')
    p.add_argument('-p', '--prefix', dest='file_prefix', action='store',
                   nargs='?', default='', type=str, metavar='P',
                   help='Prefix given to all files.')
    p.add_argument('-c', '--cadet_path', action='store',
                   nargs='?', default=None, type=str, metavar='P',
                   help='Cadet path')
    p.add_argument('-w', '--word_size', dest='word_size',
                   action='store', nargs='?', default=8, type=int, metavar='W',
                   help='Word size (default 8).')
    p.add_argument('-e', '--expr_size', dest='expr_size',
                   action='store', nargs='?', default=8, type=int, metavar='W',
                   help='Number of nodes in the syntax tree of the expressions'
                   ' (default 8).')
    p.add_argument('-d', '--directory', dest='directory',
                   action='store',
                   default='../data/',
                   help='Directory to write the formulas to.')
    p.add_argument('--only_unsat', dest='only_unsat', action='store_true',
                   help='Only accept unsat formulas.')
    return p.parse_args()

def log_parameters(args):
    filename = os.path.join(args.directory, 'README')
    textfile = open(filename, "w")
    textfile.write(str(sys.argv))
    textfile.write('\n')
    textfile.write(str(args))
    textfile.write('\n')
    repo = git.Repo(search_parent_directories=True)
    textfile.write(f'Git hash: {repo.head.object.hexsha}\n')
    textfile.close()

def main():
    args = parse_cmdline()

    if args.repetitions != 1:  # TODO
        print('Repetitions != 1 not implemented')
        quit()

    if not os.path.exists(args.directory):
        os.makedirs(args.directory)
    log_parameters(args)

    if args.seed is not None:
        print(f'Setting seed: {args.seed}')
        seed(args.seed)

    global word_length
    word_length = args.word_size

    file_extension = 'qaiger'
    num_sat = 0
    num_unsat = 0
    num_unknown = 0
    num_generated = 0
    num_attempts = 0

    while num_generated < args.num_generated:
        if num_attempts == 0:
            print('Generating file no {}'.format(num_generated+1))
        num_attempts += 1

        try:
            e = random_circuit(args.expr_size)
            ipdb.set_trace()
        except Exception as e:
            print('Got an exception when creating!')
            print(e)
            continue
        # print(e.aig.comments)

        # if len(e.aig.gates) == 0:
        #     print('    Too few variables')
        #     continue
        # if len(e.aig.gates) > args.maxvars:
        #     print('    Too many variables')
        #     continue
        if not any([v.startswith('1 x1') or v.startswith('1 x2') for v in e.inputs]):
            print('    No universals')
            continue

        f = tempfile.NamedTemporaryFile()
        f.write(str(e).encode())
        f.seek(0)

        if args.max_hardness != None:
            decision_limit = 10*args.max_hardness
        else:
            decision_limit = max(args.min_hardness, 100)

        (returncode, _, decisions) = eval_formula(f.name,
                                                  decision_limit=decision_limit,
                                                  VSIDS=args.max_hardness==None,
                                                  CEGAR=args.max_hardness==None,
                                                  cadet_path=args.cadet_path)

        f.close()  # this deletes the temporary file

        if args.only_unsat and returncode != 20:
            continue
        
        if args.max_hardness != None and returncode == 30:
            print('    Hit the decision limit')
            continue
        if returncode not in [10, 20, 30]:
            errfiledir = '{}/err{}_{}.{}'.format(args.directory,
                                                 str(num_generated),
                                                 returncode,
                                                 file_extension)
            print(f"Warning: unexpected return code: {returncode};"
                  "writing formula to {errfiledir} and ignoring it")
            textfile = open(errfiledir, "w")
            textfile.write(str(e))
            textfile.close()
            continue

        if decisions >= args.min_hardness and\
                (args.max_hardness is None or decisions <= args.max_hardness):

            if returncode == 10:
                result_string = 'SAT'
                num_sat += 1
            elif returncode == 20:
                result_string = 'UNSAT'
                num_unsat += 1
            else:
                result_string = 'UNKNOWN'
                num_unknown += 1

            print('    Found a good formula! Decisions'
                  f' {decisions}; {result_string}')

            ipdb.set_trace()
            filedir = f'{args.directory}/{args.file_prefix}{num_generated}.{file_extension}'

            textfile = open(filedir, "w")
            textfile.write(str(e))
            textfile.close()
            num_generated += 1
            num_attempts = 0
        else:
            print(f'    Not the right number of decisions: {decisions}')

    print(f'Generated {num_sat} SAT; {num_unsat} UNSAT; {num_unknown} UNKNOWN')


if __name__ == "__main__":
    main()
