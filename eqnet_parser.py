import json
from pprint import pprint

max_fixed_var = 10
fixed_variables = {}
for i in range(max_fixed_var):
    fixed_variables[chr(ord('a') + i)] = i + 1

# with open('expressions-synthetic/largeSimpleBoolean5.json') as json_file:

def load_bool_data(fname): 
    import json
    rc = []   
    with open(fname) as json_file:    
        json_data = json.load(json_file)
        
        eq_class_keys = json_data.keys()
        
        for k in eq_class_keys:
            # print(k)
            eq_class = []
            eq_class.append(json_data[k]['Original']['Tree']['Children']['child'])
            num_formulas = 1
            for j in range(len(json_data[k]['Noise'])):
                eq_class.append(json_data[k]['Noise'][j]['Tree']['Children']['child'])
                num_formulas += 1
            # print('  ' + str(num_formulas) + ' variants')
            rc.append(eq_class)
        del eq_class_keys
        del json_data
    return rc
    
def simplify_clause(c):
    s = set(c)
    for x in s:
        if -x in s:
            return None
    return list(s)

def add_clauses(CNF, clauses):
    for c in clauses:
        c = simplify_clause(c)
        if c != None:
            CNF['clauses'] += [c]
            for l in c:
                v = abs(l)
                if v not in CNF['clauses_per_variable']:
                    CNF['clauses_per_variable'][v] = []
                CNF['clauses_per_variable'][v] += [c]
            

def translate_expression(CNF, node):
    name = node['Name']
    if len(name) == 1 and name.islower(): # Variable
        assert(name in fixed_variables)
        if name not in CNF['origvars']:
            CNF['origvars'][name] = fixed_variables[name]
            
            # Use this code in case we don't want to use fixed_variables
            # CNF['maxvar'] += 1
            # CNF['origvars'][name] = CNF['maxvar']
            
        CNF['topvar'] = CNF['origvars'][name]
        
    elif name == 'Not':
        subexpr = - translate_expression(CNF, node['Children']['child'])
        CNF['topvar'] = subexpr
        return subexpr
        
        # Alternative encoding for Not, introducing a new variable
#         subexpr = translate_expression(CNF, node['Children']['child'])
#         CNF['maxvar'] += 1
#         CNF['topvar'] = CNF['maxvar']
#         CNF['auxvars'] += CNF['maxvar']
#         add_clauses(CNF, [[subexpr, CNF['maxvar']],[- subexpr, - CNF['maxvar']])
#         return CNF['maxvar']
        
    elif name == 'Or':
        subexpr_left  = translate_expression(CNF, node['Children']['left'])
        subexpr_right = translate_expression(CNF, node['Children']['right'])
        CNF['maxvar'] += 1
        CNF['topvar'] = CNF['maxvar']
        CNF['auxvars'] += [CNF['topvar']]
        add_clauses(CNF,  [ [- subexpr_left,  CNF['topvar']], \
                            [- subexpr_right, CNF['topvar']], \
                            [subexpr_left, subexpr_right, - CNF['topvar']]
                        ])
        
    elif name == 'Implies':
        subexpr_left  = translate_expression(CNF, node['Children']['left'])
        subexpr_right = translate_expression(CNF, node['Children']['right'])
        CNF['maxvar'] += 1
        CNF['topvar'] = CNF['maxvar']
        CNF['auxvars'] += [CNF['topvar']]
        add_clauses(CNF,  [ [  subexpr_left,  CNF['topvar']], \
                            [- subexpr_right, CNF['topvar']], \
                            [- subexpr_left, subexpr_right, - CNF['topvar']]
                        ])
        
    elif name == 'And':
        subexpr_left  = translate_expression(CNF, node['Children']['left'])
        subexpr_right = translate_expression(CNF, node['Children']['right'])
        CNF['maxvar'] += 1
        CNF['topvar'] = CNF['maxvar']
        CNF['auxvars'] += [CNF['topvar']]
        add_clauses(CNF,  [ [subexpr_left,  - CNF['topvar']], \
                            [subexpr_right, - CNF['topvar']], \
                            [ - subexpr_left, - subexpr_right, CNF['topvar']]
                        ])
        
    elif name == 'Xor':
        subexpr_left  = translate_expression(CNF, node['Children']['left'])
        subexpr_right = translate_expression(CNF, node['Children']['right'])
        CNF['maxvar'] += 1
        CNF['topvar'] = CNF['maxvar']
        CNF['auxvars'] += [CNF['topvar']]
        add_clauses(CNF,  [ [  subexpr_left,   subexpr_right,  - CNF['topvar']], \
                            [- subexpr_left, - subexpr_right,  - CNF['topvar']], \
                            [  subexpr_left, - subexpr_right,    CNF['topvar']], \
                            [- subexpr_left,   subexpr_right,    CNF['topvar']], \
                        ])
        
    else:
        print('Error: Unknown operator ' + name)
        quit() 
        
    return CNF['topvar']

# Translate formulas to CNF

def to_cnf(classes):
    eq_classes = {}
    for eq_class_data in classes:
        eq_class = []
        for formula_node in eq_class_data:
            # print('Processing: ')
            # pprint(formula_node)
            CNF = {'topvar' : None, \
                   'maxvar' : 10,   \
                   'origvars': {},  \
                   'auxvars': [],   \
                   'clauses': [],   \
                   'clauses_per_variable' : {}}
            eq_class += [CNF]        
            translate_expression(CNF, formula_node)
            # add final clause asserting the topvar (the expression has to be true)
            add_clauses(CNF, [[CNF['topvar']]])
            # print(CNF)
            
            for v in CNF['clauses_per_variable'].keys():
                if len(CNF['clauses_per_variable'][v]) > 20:
                    print('Error: too many clauses for variable ' + str(v))
                    quit()
            
        eq_classes[formula_node['Symbol']] = eq_class
        

    return eq_classes

# print(eq_classes.keys())
# print(eq_classes[u'Or(And(Not(a), Not(b)), c)'][0])
