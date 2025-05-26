import sys
import random
import subprocess
import pandas
import time
import re

corner_cases = {
    "0c0": "",
    "0c3": "R B2 R2 D2 R' B2 R2 B2 D2 R",
    "0c4": "U' R2 U R2 U2 B2 U B2 U",
    "2c3": "U F2 U2 R2 F2 U' F2 R2 U",
    "2c4": "U2 R2 U L2 U R2 U' L2 U",
    "2c5": "U L2 U R2 U' R2 U R2 U",
    "4a1": "U R2 L2 U F2 B2 D",
    "4a2": "U R2 U2 F2 U' F2 U F2 U2 R2 U",
    "4a3": "U' F2 R2 U L2 U2 B2 D",
    "4a4": "U R2 U' F2 U R2 F2 U B2 U' F2 U",
    "4b2": "U R2 F2 R2 F2 U",
    "4b3": "U R2 U2 B2 U' F2 L2 D",
    "4b4": "U R2 U' R2 F2 U2 F2 U' F2 U",
    "4b5": "U B2 U B2 U' F2 U F2 U",
}

edge_cases = {
    "0e": "",
    "2e": "U R2 F2 R2 U",
    "4e": "U L2 R2 D",
    "6e": "U L2 R2 B2 L2 B2 D",
    "8e": "U R2 L2 F2 B2 U",
}
def get_dr_in_subset(corners, edges):
    '''Genereates a random element of the dr subset 
    specified by the corners and edges arguments'''

    pre_half_turns = random.choices(['R2', 'L2', 'B2', 'F2', 'U2', 'D2'], k = 25)
    post_half_turns = random.choices(['R2', 'L2', 'B2', 'F2', 'U2', 'D2'], k = 25)
    dr = (" ").join(pre_half_turns) + " "+ corner_cases[corners] + " "+ edge_cases[edges] + " " + (" ").join(post_half_turns)
    inverse = subprocess.check_output(["nissy", "solve", "-p", dr])
    return (subprocess.check_output(["nissy", "invert", inverse])).decode("utf-8").strip()

def get_subset_to_optimal_dict():
    subset_dict = {}
    for cc in corner_cases:
        for ec in edge_cases:
            for i in range(1):
                dr = get_dr_in_subset(cc, ec)
                optimal_length = dr.count(" ") + 1 #number of spaces + 1
                subset_dict.setdefault(cc + " " +ec, []).append(optimal_length)
            print(subset_dict)
    return subset_dict

def get_corner_optimal_to_optimal_dict():
    corner_opt_dict = {}
    dr = get_dr_in_subset

def gen_soln_table(scrambles, p):
    '''Generates a table (return type to be decided) of solutions and their lengths to the provided scrambles '''

    #format the scramble list into a string to pass to stdin for nissy
    mylist = []
    for scramble in scrambles:
        mylist.append("\n" + scramble)
    to_input = "solve drfin -i -t 20" + ' '.join(mylist)
    nissy_output = p.communicate(input=bytes(to_input,'utf-8'))[0].decode()

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
        solns.append(soln[0])
        lengths.append(int(soln[1]))
    print(solns)
    print(lengths)



def gen_subset_scrambles(corners, edges, n):
    '''Generates a list of n scrambles of the provided subset. Generation of 1000 scrambles takes about 0.12s.'''

    scrambles = []
    for i in range(n):
        scramble = (" ").join(half_turns(20)) + " "+ corner_cases[corners] + " "+ edge_cases[edges] + " " + (" ").join(half_turns(20))
        scrambles.append(scramble)
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
            
#def get_random_dr():
if __name__ == "__main__":
    p = subprocess.Popen("nissy", shell = True, cwd = "/Users/user/Desktop/nissy-2.0.7",stdout=subprocess.PIPE,stdin=subprocess.PIPE)
    time.sleep(1)
    start = time.time()
    
    _2c3_2e = gen_subset_scrambles("2c3", "2e", 1000)
    my_table = gen_soln_table(_2c3_2e, p)
    print(time.time() - start)








    #the below is unnecessary if I use p.communicate.
    p.terminate()
    time.sleep(1)
    if p.poll() is None:
        print("nissy is still running!")
    else:
        print(f"nissy has exited with code {p.poll()}")

    
    # TODO: refactor this so that nissy is called literally only once! 
    # Start out by just filling a panda dataframe for one subset. Learn how to put this dataframe into a nice looking csv / excel.
    # For the optimal corner solution -> optimal stats, generate a lot of random drs using nissy scramble dr. generate (co, o) pair for each. Sort by co. Take the means of o. graph this.
    # 