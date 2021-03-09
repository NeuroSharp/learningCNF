#!/usr/bin/env python3

import sys
from os import listdir
from os.path import isfile, join
from generators.cnf_tools import *
# import ipdb

def simplify_clause(c):
    s = set(c)
    for x in s:
        if -x in s:
            return None
    return list(s)

def add_clauses(CNF, clauses):
    for t in clauses:
        c = simplify_clause(t)        
        if c != None:
            CNF['clauses'] += [c]
            for l in c:
                v = abs(l)
                if v not in CNF['clauses_per_variable']:
                    CNF['clauses_per_variable'][v] = []
                CNF['clauses_per_variable'][v] += [c]

def dimacs_to_cnf(filename):
    
    CNF = {'topvar' : None, \
           'maxvar' : None, \
           'origvars': {},  \
           'auxvars': [],   \
           'clauses': [],   \
           'clauses_per_variable' : {}}
    
    with open(filename, 'r') as f:
        
        numclauses = None
        
        for line in f.readlines():
            words = line.split()
            if is_number(words[0]):
                lits = list(map(int,words[0:-1]))
                add_clauses(CNF,[lits])
                
                # Assigns first singleton clause as topvar
                if len(lits) == 1 and CNF['topvar'] == None:
                    CNF['topvar'] = lits[0]
                
            else:
                if type(words[0]) == bytes and words[0] == b'p' \
                    or  \
                   type(words[0]) == str and words[0] == 'p':
                    CNF['maxvar'] = int(words[2])
                    numclauses = int(words[3])
        
        # if numclauses != len(CNF['clauses']):
        #     print('WARNING: Number of clauses in file is inconsistent.')
        
        assert(CNF['maxvar'] != None)
        CNF['origvars'] = {i: i for i in range(1,CNF['maxvar']+1)}
    
    for v in CNF['clauses_per_variable'].keys():
        if len(CNF['clauses_per_variable'][v]) > MAX_CLAUSES_PER_VARIABLE:
            print('Error: too many clauses for variable ' + str(v))
            # ipdb.set_trace()
            quit()

    return CNF


def qdimacs_to_cnf(filename, zero_based=False):
    cvars = {}
    clauses = []
    maxvar = 0
    num_clauses = 0
    # offset = -1 if zero_based else 0
    offset = 0
    with open(filename, 'r') as f:
        while True:
            a = f.readline()        # header
            if a[0] == 'p':
                break
        line = a.split(' ')        # Number of vars/clauses - "p cnf 159 312"
        maxvar = int(line[2])
        num_clauses = int(line[3])        
        for line in f.readlines():            
            if line[0] == 'a' or line[0] == 'e':                
                vs = [int(x)+offset for x in line.split(' ')[1:-1]]
                for x in vs:
                    if x>maxvar:
                        print('Warning: maxvar is lower than whats in the formula, updating')
                        maxvar = x
                    cvars[x] = {'universal': line[0]=='a', 'clauses': []}

        # read all clauses
        
            else:
                words = line.split()        
                lits = [int(x) for x in words[:-1]]            
                clauses.append(simplify_clause(lits))

    if not cvars:           # No var declarations, we'll assume its a regular cnf, everything is existential
        for x in range(1,maxvar+1):
            cvars[x] = {'universal': False, 'clauses': []}

    for c in clauses:
        for lit in c:
            cvars[abs(lit)]['clauses'].append(c)

    # Make sure not too many clauses per var            
    for x in cvars.keys():
        if len(cvars[x]['clauses']) > MAX_CLAUSES_PER_VARIABLE:
            clauses = len(cvars[x]['clauses'])
            print('Error: too many clauses ({}) for variable {} in file {}'.format(clauses,str(x),filename))
            return None
            

                        
    return {'maxvar' : maxvar, 
           'num_clauses': num_clauses, 
           'clauses': clauses,
           'cvars' : cvars,
           'fname': filename
           }
    
def load_class(directory):
    files = [join(directory, f) for f in listdir(directory)]
    return list(map(dimacs_to_cnf, files))


def main(argv):
    
    print(load_class('data/randomCNF_5/sat/'))
    # eq_classes = {}
    # for eq_class_data in classes:
    #     eq_class = []
    #
    # eq_classes[formula_node['Symbol']] = eq_class
if __name__ == "__main__":
    main(sys.argv)
    