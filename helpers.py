import sys
import random
import subprocess
import pandas as pd
import time
import re
from vfmc_core import Cube # type: ignore
from vfmc import attempt


def get_dr_scrambles(n):

    scrambles = subprocess.check_output(
        ["nissy", "scramble", "dr", "-n", str(n)],
        text=True                    # auto‑decode UTF‑8
        ).strip().splitlines()
    
    return scrambles


def half_turns(k):
    '''generates a sequence of non-redundant halfturns of length k or k+1 (for parity reasons)'''

    moves = ["R2","L2","F2","B2","U2","D2"]
    scramble = []
    n = 0
    while n < random.sample([k,k+1],1)[0]: 
        move = moves[random.randint(0,5)]
        if n > 0 and move == scramble[n-1]:
            continue
        if n > 1 and move == "R2" and scramble[n-1] == "L2" and scramble[n-2] == move:
            continue
        elif n > 1 and move == "L2" and scramble[n-1] == "R2" and scramble[n-2] == move:
            continue
        elif n > 1 and move == "F2" and scramble[n-1] == "B2" and scramble[n-2] == move:
            continue
        elif n > 1 and move == "B2" and scramble[n-1] == "F2" and scramble[n-2] == move:
            continue
        elif n > 1 and move == "U2" and scramble[n-1] == "D2" and scramble[n-2] == move:
            continue
        elif n > 1 and move == "D2" and scramble[n-1] == "U2" and scramble[n-2] == move:
            continue
        scramble.append(move)
        n = n+1
    return scramble
def get_corner_solns(scrambles):
    '''Generates a list of optimal corner solution lengths to the scrambles.'''

    #open a subprocess
    p = subprocess.Popen(["nissy", "-b"], shell = True, cwd = "/Users/user/Desktop/nissy-2.0.8",stdout=subprocess.PIPE,stdin=subprocess.PIPE)
    
    #format the scramble list into a string to pass to stdin for nissy
    mylist = []
    for scramble in scrambles:
        mylist.append("\n" + scramble)
    to_input = "solve corners -i -t 20\n" + "\n".join(scrambles) + "\n"
    nissy_output, _ = p.communicate(input=bytes(to_input,'utf-8'))
    nissy_output = nissy_output.decode()

    ''' At this point, output appears as:

        >>> Line: D U2 F D B' F L2 D' F2 R2 L B2 L' U2 B2 R F2 L' D2
        U2 R2 F2 L B2 D' R2 D' F U L2 B' U' R2 D2 R2 U (17)
        >>> Line: D B R U' B' L2 U L U D2 R L B2 U2 L2 x2 R U2 B2 L F2
        D' F R' D B L2 B R2 L U L U2 B D' U R U F2 (18)
    '''
    nissy_output = re.split(r'>>> Line: |nissy', nissy_output)
    nissy_output.pop(0)
    nissy_output.pop(0)
    nissy_output.pop()
    nissy_output.pop()

    solns = []
    lengths = []

    #loop through each block of the output which corresponds to one scramble.
    for i in nissy_output:
        soln = i.split('\n')[1]
        soln = re.split(r'\(|\)',soln)
        #solns.append(soln[0])
        lengths.append(int(soln[1]))
    #return solns, lengths
    time.sleep(0.5)
    return lengths




def get_solns(scrambles):
    '''Generates a list of optimal solution lengths to the scrambles.'''

    #open a subprocess
    p = subprocess.Popen(["nissy", "-b"], shell = True, cwd = "/Users/user/Desktop/nissy-2.0.8",stdout=subprocess.PIPE,stdin=subprocess.PIPE)
    
    #format the scramble list into a string to pass to stdin for nissy
    mylist = []
    for scramble in scrambles:
        mylist.append("\n" + scramble)
    to_input = "solve drfin -i -t 20" + ' '.join(mylist)
    nissy_output, _ = p.communicate(input=bytes(to_input,'utf-8'))
    nissy_output = nissy_output.decode()

    ''' At this point, output appears as:

        >>> Line: D U2 F D B' F L2 D' F2 R2 L B2 L' U2 B2 R F2 L' D2
        U2 R2 F2 L B2 D' R2 D' F U L2 B' U' R2 D2 R2 U (17)
        >>> Line: D B R U' B' L2 U L U D2 R L B2 U2 L2 x2 R U2 B2 L F2
        D' F R' D B L2 B R2 L U L U2 B D' U R U F2 (18)
    '''
    nissy_output = re.split(r'>>> Line: |nissy', nissy_output)
    nissy_output.pop(0)
    nissy_output.pop(0)
    nissy_output.pop()
    nissy_output.pop()

    solns = []
    lengths = []

    #loop through each block of the output which corresponds to one scramble.
    for i in nissy_output:
        soln = i.split('\n')[1]
        soln = re.split(r'\(|\)',soln)
        #solns.append(soln[0])z
        lengths.append(int(soln[1]))
    #return solns, lengths
    time.sleep(0.5)
    return lengths 




def get_subset(scramble):
    cube = Cube(scramble)
    step = attempt.PartialSolution("htr","ud")
    return (step.step_info.case_name(cube))
