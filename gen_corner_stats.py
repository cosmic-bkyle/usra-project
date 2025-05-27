import helpers
import sys
import random
import subprocess
import pandas as pd
import time
import re
import math

NUM_SCRAMBLES = 100

def main():
    '''generate ~10k scrambles. Compute corner solution length and optimal solution length on all. 
    Group by corner solution length, ordered from smallest to largest, and display corner mean, std'''
    
    #TODO: TURN THIS INTO POPEN
    start = time.time()
    scrambles = []
    corners_lengths = []
    solution_lengths = []
    for i in range(100):
        scramble_block = helpers.get_dr_scrambles(NUM_SCRAMBLES)
        scrambles += scramble_block
    
    print(len(scrambles))
    print(time.time() - start)

    corners_block = helpers.get_corner_solns(scrambles)
    solutions_block = helpers.get_solns(scrambles)
    corners_lengths += (corners_block)
    solution_lengths += (solutions_block) 
    print(len(scrambles))
    print(len(corners_block))
    print(len(solutions_block))
    scrambles.pop() # change logic later. I have a bug where the last scramble doesn't get solved. 
    
    
    if not (len(scrambles) == len(corners_block) == len(solution_lengths)):
        print("Error, some scrambles were not solved.")
    
    df = pd.DataFrame({
        "scramble"      : scrambles,
        "corner solution length"    : corners_lengths,
        "average solution length"   : solution_lengths,
    })

    stats = (df
             .groupby("corner solution length")["average solution length"]
             .agg(["count", "mean", "std"])
             .sort_index()
             .round(3))

    stats.to_csv("corner_stats_ben.csv", index_label="corner solution length ")
    print("it took " + str(time.time() - start))

    

        



if __name__ == "__main__":
    main()