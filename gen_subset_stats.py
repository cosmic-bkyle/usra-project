import helpers
import sys
import random
import subprocess
import pandas as pd
import time
import re

NUM_SCRAMBLES = 100

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


def gen_subset_scrambles(corners, edges, n):
    '''Generates a list of n scrambles of the provided subset. Generation of 1000 scrambles takes about 0.12s.'''

    scrambles = []
    for i in range(n):
        scramble = (" ").join(helpers.half_turns(20)) + " "+ corner_cases[corners] + " "+ edge_cases[edges] + " " + (" ").join(helpers.half_turns(20))
        scrambles.append(scramble)
    return scrambles

def main():
    rows = []
    time_start = time.time()
    for c in corner_cases:
        for e in edge_cases:
            subset_scrambles = gen_subset_scrambles(c, e, NUM_SCRAMBLES)
            lengths_list = helpers.get_solns(subset_scrambles)
            ser = pd.Series(lengths_list) #array
            average_len = round(ser.mean(), 3)
            distribution = (ser.value_counts(normalize=True)
                           .sort_index(ascending=False)
                           .mul(100).round(2)
                           .astype(str) + '%')
            distribution_string  = ";  ".join(f"{len}: {percent}" for len, percent in distribution.items())

            rows.append({
                "corner case": c,
                "edge case"  : e,
                "solution mean"   : average_len,
                "distribution": distribution_string
            })

            print("Done with "+ c + " " + e + ". mean: " + str(average_len))
    df = pd.DataFrame(rows)
    df.to_csv("subset_stats_ben.csv", index=False)
    print("Total time: " + str(time.time() - time_start))

if __name__ == "__main__":
    main()