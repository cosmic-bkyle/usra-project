import sys
import random
import subprocess
import pandas as pd
import time
import math
import re
from vfmc_core import Cube # type: ignore
from vfmc import attempt
import re
from collections import OrderedDict
from typing import List, Tuple
MOVE = r"[URFDLB][2']?"
SEQ  = rf"(?:{MOVE}(?:\s+|$))+"

def parse_nissy_output_with_lengths(text: str):
    """
    Parse Nissy output into:
    - solutions: list of lists of solution sequences (strings)
    - lengths:   list of integers (optimal length per scramble)
    """
    lines = text.splitlines()
    lines[0] = lines[0][8:]
    solutions = []
    lengths = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith(">>> Line:"):
            current_solutions = []
            current_length = None
            i += 1
            while i < len(lines):
                s = lines[i].strip()
                if s.startswith(">>> Line:") or s.startswith("nissy-#") or s == "":
                    break
                m = re.match(r"^(.*?)(?:\s*\((\d+)\))\s*$", s)
                if m:
                    sol = m.group(1).strip()
                    length = int(m.group(2))
                    current_solutions.append(sol)
                    current_length = length  # they are all the same
                i += 1
            if current_solutions:
                solutions.append(current_solutions)
                lengths.append(current_length)
            continue
        i += 1
    return solutions, lengths
def parse_solutions(text):
    '''
    from a string from stdout containing nissy solutions, compile a list of lists of the solutions and a list of lists of their respective lengths
    '''
    solutions_by_scramble: List[List[str]] = []
    lengths_by_scramble: List[List[int]] = []

    lines = text.splitlines()
    i = 0
    lines[0] = lines[0][8:]
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith(">>> Line:"):
            # Start a new block
            current_solutions: List[str] = []
            current_lengths: List[int] = []
            i += 1
            # Consume solution lines until next header/prompt/blank
            while i < len(lines):
                s = lines[i].strip()
                if not s or s.startswith(">>> Line:") or s.startswith("nissy-#"):
                    break
                m = re.match(r"^(.*?)[ \t]*\((\d+)\)\s*$", s)
                if m:
                    moves = m.group(1).strip()
                    length = int(m.group(2))
                    current_solutions.append(moves)
                    current_lengths.append(length)
                # non-matching lines are ignored safely
                i += 1
            solutions_by_scramble.append(current_solutions)
            lengths_by_scramble.append(current_lengths)
            continue  # inner loop moved i appropriately
        i += 1
    return solutions_by_scramble, lengths_by_scramble



def get_n_eos_axis(axis, n, scrambles):
    '''
    returns two lists of length n
    list1 is a a list of eo solution sequences 
    list2 is a list of their respective lengths 

    note: -a flag is used so e.g. F and F' are considered different eos
    '''
    p = subprocess.Popen(["nissy", "-b"], shell = True, cwd = "/Users/user/Desktop/nissy-2.0.8",stdout=subprocess.PIPE,stdin=subprocess.PIPE)
    mylist = []
    for scramble in scrambles:
        mylist.append("\n" + scramble)
    to_input = "solve eo" + axis + " -a -n " + str(n) + " -i -t 20" + ' '.join(mylist) + '\n'
    nissy_output, _ = p.communicate(input=bytes(to_input,'utf-8'))
    nissy_output = nissy_output.decode()

    return parse_solutions(nissy_output)
    
def get_m_drs_axis(eoaxis, draxis, n, m, scrambles, eos):
    '''
    returns two lists of length SxN, each containing one element for each scramble x eo.
    list1 is a a list of eo solution sequences 
    list2 is a list of their respective lengths
    '''
    p = subprocess.Popen(["nissy", "-b"], shell = True, cwd = "/Users/user/Desktop/nissy-2.0.8",stdout=subprocess.PIPE,stdin=subprocess.PIPE)
    
    mylist = []
    for outer_index, scramble in enumerate(scrambles):
        list_of_eos = eos[0][outer_index]
        for eo in list_of_eos:
            eostate_sequence = scramble + " " + eo
            mylist.append("\n" + eostate_sequence)
    to_input = "solve dr" + draxis + "-eo"+ eoaxis +" -n " + str(m) + " -i -t 20 " + ' '.join(mylist) + '\n'
    nissy_output, _ = p.communicate(input=bytes(to_input,'utf-8'))
    nissy_output = nissy_output.decode()
    drs_from_eos, lengths = parse_solutions(nissy_output) # these lists should have length S x N
    drs_from_eos_sequences = [drs_from_eos[i:i+n] for i in range(0, len(drs_from_eos), n)]
    dr_from_eo_lengths = [lengths[i:i+n] for i in range(0, len(lengths), n)]

    return drs_from_eos_sequences, dr_from_eo_lengths



def get_optimal_drs_normal(scrambles):
    '''
    Returns two lists list1, list2
    list1 is the list of lists of optimal solution sequencess to dr 
    list2 is the list of ints of optimal solution lengths to dr.
    
    '''
    p = subprocess.Popen(["nissy", "-b"], shell = True, cwd = "/Users/user/Desktop/nissy-2.0.8",stdout=subprocess.PIPE,stdin=subprocess.PIPE)
    mylist = []
    for scramble in scrambles:
        mylist.append("\n" + scramble)
    to_input = "solve dr -o -i -t 20" + ' '.join(mylist) + '\n'
    nissy_output, _ = p.communicate(input=bytes(to_input,'utf-8'))
    nissy_output = nissy_output.decode()


    ''' At this point, output appears as:

        >>> Line: D U2 F D B' F L2 D' F2 R2 L B2 L' U2 B2 R F2 L' D2
        U2 R2 F2 L B2 D' R2 D' F U L2 B' U' R2 D2 R2 U (17)
        >>> Line: D B R U' B' L2 U L U D2 R L B2 U2 L2 x2 R U2 B2 L F2
        D' F R' D B L2 B R2 L U L U2 B D' U R U F2 (18)
    '''
    print(nissy_output)
    return parse_nissy_output_with_lengths(nissy_output)

def get_dr_scrambles(n):
    

    scrambles = subprocess.check_output(
        ["nissy", "scramble", "dr", "-n", str(n)],
        text=True                    # auto‑decode UTF‑8
        ).strip().splitlines()
    
    return scrambles
def get_full_scrambles(n):
    scrambles = subprocess.check_output(
        ["nissy", "scramble", "-n", str(n)],
        text=True                    # auto‑decode UTF‑8
        ).strip().splitlines()
    return scrambles

def get_shortest_eos(scrambles, axis, side):
    '''
    return a list of the scrambles' distances to eo on given axis and side

    axis: fb, rl, ud
    side: n, i
    '''
    p = subprocess.Popen(["nissy", "-b"], shell = True, cwd = "/Users/user/Desktop/nissy-2.0.8",stdout=subprocess.PIPE,stdin=subprocess.PIPE)
    
    #format the scramble list into a string to pass to stdin for nissy
    mylist = []
    if side == "i":
        for scramble in scrambles:
            mylist.append("\n (" + scramble + ")")

    else: 
        for scramble in scrambles:
            mylist.append("\n" + scramble)


    to_input = "solve eo" +axis + " -i -t 20" + ' '.join(mylist)
    nissy_output, _ = p.communicate(input=bytes(to_input,'utf-8'))
    nissy_output = nissy_output.decode()

    nissy_output = re.split(r'>>> Line: |nissy', nissy_output)
    nissy_output.pop(0)
    nissy_output.pop(0)
    nissy_output.pop()
    nissy_output.pop()
    lengths = []

    for i in nissy_output:
        soln = i.split('\n')[1]
        soln = re.split(r'\(|\)',soln)
        #solns.append(soln[0])z
        lengths.append(int(soln[1]))
    





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

def get_all_solns(scrambles, solution_lengths):
    '''
    generates a list of lists of all optimal solutions to the scrambles.
    '''
    bugs = 0
    list_of_lists = []
    for i, scramble in enumerate(scrambles):
        p = subprocess.Popen(["nissy", "-b"], shell = True, cwd = "/Users/user/Desktop/nissy-2.0.8",stdout=subprocess.PIPE,stdin=subprocess.PIPE)
        mylist = []
        to_input = "solve drfin -o -t 64 "+scramble
        nissy_output, _ = p.communicate(input=bytes(to_input,'utf-8'))
        nissy_output = nissy_output.decode()

        cleaned = re.sub(r"^nissy-#\s*", "", nissy_output, flags=re.M)
        m = re.search(r"\((\d+)\)\s*$", cleaned, flags=re.M)
        number = 0
        if m:
            number = int(m.group(1))
            if number != int(solution_lengths[i]):
                bugs += 1
        sequences = [
            re.sub(r"\s+\(\d+\)$", "", line)
            for line in cleaned.splitlines()
            if line.strip()
            ]
        print(scramble)
        print("solved by ")
        print(sequences)
        print(bugs)
        print()
        


    '''
    for scramble in scrambles:
        mylist.append("\n" + scramble)
    to_input = "solve drfin -i -o -t 20" + ' '.join(mylist)
    nissy_output, _ = p.communicate(input=bytes(to_input,'utf-8'))
    nissy_output = nissy_output.decode()
    print(nissy_output)
    nissy_output = re.split(r'>>> Line: |nissy', nissy_output)
    nissy_output.pop(0)
    nissy_output.pop(0)
    nissy_output.pop()
    list_of_lists = []
    print(nissy_output)
    for string in nissy_output:
        print(string)
        print("hi")
        sequences = string.split("\n")
        sequences.pop()
        scramble = sequences.pop(0)
        print(scramble)
        solns = []
        for soln in sequences:
            soln2 = re.sub(r" \((?:[0-9]|1[0-9]|20)\)$", "", soln)
            solns.append(soln2)
        list_of_lists.append(solns)
        
    print(scrambles)
    print(list_of_lists)
    print(len(list_of_lists))
    print(len(list_of_lists[-1]))
    '''

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

    #loop through each block of the output, which corresponds to one scramble.
    for i in nissy_output:
        soln = i.split('\n')[1]
        soln = re.split(r'\(|\)',soln)
        #solns.append(soln[0])z
        lengths.append(int(soln[1]))
    #return solns, lengths
    time.sleep(0.5)
    return lengths 
    print(nissy_output)


def get_corner_solns(scrambles):
    '''Generates a list of optimal corner solution lengths to the scrambles.
    
    NOTE: bugged; neither parses nor appends final corner solution
    '''

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

def get_subsets(scrambles):
    subsets = []
    for scramble in scrambles:
        cube = Cube(scramble)
        step = attempt.PartialSolution("htr","ud")
        subsets.append(str(step.step_info.case_name(cube)))
    return subsets

def main():
    #print the probability of sub-9 domino finish.
    old_data = pd.read_parquet("dr_to_solved/filtered_scrambles.parquet")
    solns = old_data['soln'].tolist()
    counter = 0
    size = len(solns)
    total = 0
    for soln in solns:
        total = total + int(soln)
    print(total)
    print(size)
    summ = 0
    mean = total / size 
    for soln in solns:
        dev = abs(soln - mean)**2
        summ = summ + dev
    print(math.sqrt(summ/size))


if __name__ == "__main__":
    main()