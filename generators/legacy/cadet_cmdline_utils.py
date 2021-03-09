import os
import re
from IPython.core.debugger import Tracer
import numpy as np
import tempfile
from subprocess import Popen, PIPE, STDOUT

from aux_utils import is_number


def extract_num_conflicts(s):
    res = re.findall(' Conflicts: (\d+)', str(s))
    if len(res) == 1:
        return int(res[0])
    else:
        print('  ERROR: {}'.format(s))
        return 0


def extract_num_decisions(s):
    res = re.findall(' Decisions: (\d+)', str(s))
    if len(res) == 1:
        return int(res[0])
    else:
        print('  ERROR: {}'.format(s))
        return 0


def _ignore_output(p):
    # p.wait()
    # print(p.poll())
    # print('Poll end')
    for line in p.stdout:
        # print(line[:-1])
        newfile = line == 'Enter new filename:\n'
        if line.startswith('s ') or newfile:
            return newfile

    # while True:
    #     out = p.stdout.readline()
    #     print(out)
    #     if not out:
    #         break

def _rl_interaction(cmd, filename):
    p = Popen(cmd, stdout=PIPE, stdin=PIPE, universal_newlines=True, bufsize=1)

    _ignore_output(p)
    # print('writing ' + f'{filename}')
    p.stdin.write(f'{filename}\n')
    # print('Written!')
    i = 0
    while not _ignore_output(p):
        i += 1
        p.stdin.write('?\n')
    p.terminate()
    print(f'Terminated after {i} steps')
    return 30, None, i


def eval_formula(filename,
                 decision_limit=None, 
                 soft_decision_limit=False,
                 VSIDS=False,
                 fresh_seed=False, 
                 CEGAR=False,
                 RL=False,
                 debugging=True,
                 projection=False,
                 cadet_path=None
                 ):
    assert isinstance(filename, str)

    if cadet_path is None:
        # cadet_path = './../../cadet/dev/cadet'
        cadet_path = './../cadet'

    cmd = [cadet_path, '-v', '1', '--sat_by_qbf']
    if debugging:
        cmd += ['--debugging']
    if decision_limit != None:
        cmd += ['-l', f'{decision_limit}']
    if soft_decision_limit:
        cmd += ['--cegar_soft_conflict_limit']
    if CEGAR:
        cmd += ['--cegar']
    if not VSIDS:
        cmd += ['--random_decisions']
    if fresh_seed:
        cmd += ['--fresh_seed']
    if projection:
        cmd += ['-e', 'elim.aag']

    if RL:
        cmd += ['--rl']
    else:
        cmd += [filename]

    if RL:
        return _rl_interaction(cmd, filename)

    p = Popen(cmd, stdout=PIPE, stdin=PIPE)
    stdout, stderr = p.communicate()

    if p.returncode not in [10, 20, 30]:
        print(f'Command: {cmd}')
        print(stdout)
        print(stderr)
        return None, None, None

    if p.returncode is 30:
        conflicts = None
        num_decisions = decision_limit
    else:
        conflicts = extract_num_conflicts(stdout)
        num_decisions = extract_num_decisions(stdout)

    if decision_limit != None and num_decisions > decision_limit:
        print('Error: decision limit was violated')
        print(formula)
        print(' '.join(cmd))
        print(stdout)
        quit()

    return p.returncode, conflicts, num_decisions

