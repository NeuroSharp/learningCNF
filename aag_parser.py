#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 22:16:14 2019

@author: ryanbrill
"""

def read_qaiger(filename):
    """
    read from a `.qaiger` file (which contains an `aag` circuit).
    returns a dictionary..
    """
    maxvar = 0
    num_inputs = 0
    num_latches = 0
    num_outputs = 0
    num_and_gates = 0
    inputs = []
    latches = []
    outputs = []
    and_gates = []
    input_symbols = []
    output_symbols = []
    avars = {}

    with open(filename, 'r') as f:
        while True:
            a = f.readline()        # header
            if a[0:3] == 'aag':
                break
        line = a.split(' ')         
        # read the first line, like "aag 256 32 0 1 224"
        maxvar = int(line[1])
        num_inputs = int(line[2])
        num_latches = int(line[3])
        num_outputs = int(line[4])
        num_and_gates = int(line[5])
        
        # initialize avars
        for v in range(maxvar):
            avars[v + 1] =  {'universal': 'Not yet implemented', 'and_gates': []}
        
        # read inputs
        k = num_inputs
        while k > 0 and a:
            a = f.readline()
            line = a.split()
            inputs.append(int(line[0]))
            k -= 1
        
        # ignore latches, for now
        
        # read outputs
        k = num_outputs
        while k > 0 and a:
            a = f.readline()
            line = a.split()
            outputs.append(int(line[0]))
            k -= 1
        
        # read and gates
        k = num_and_gates
        while k > 0 and a:
            a = f.readline()
            line = a.split()
            and_gate = [int(line[0]), int(line[1]), int(line[2])]
            and_gates.append(and_gate)
            k -= 1
            
            # update avars
            for l in and_gate:
                v = int(l/2) # the variable v corresponding to the literal l
                avars[v]['and_gates'].append(and_gate)
            
        # read input symbols
        k = num_inputs
        while k > 0 and a:
            a = f.readline()
            line = a.split()
            input_symbols.append( ' '.join(line[1:]) )
            k -= 1
            
        # read output symbols
        k = num_outputs
        while k > 0 and a:
            a = f.readline()
            line = a.split()
            output_symbols.append( ' '.join(line[1:]) )
            k -= 1
        
    return {'maxvar': maxvar,
            'num_inputs': num_inputs,
            'num_latches': num_latches,
            'num_outputs': num_outputs,
            'num_and_gates': num_and_gates,
            'inputs': inputs,
            'latches': latches,
            'outputs': outputs,
            'and_gates': and_gates,
            'input_symbols': input_symbols,
            'output_symbols': output_symbols,
            #'avars': avars,
            'fname': filename
            }
