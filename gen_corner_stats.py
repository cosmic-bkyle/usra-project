import helpers
import sys
import random
import subprocess
import pandas as pd
import time
import re
import math
import pathlib
import csv

NUM_SCRAMBLES = 100

def main():
    '''generate ~10k scrambles. Compute corner solution length and optimal solution length on all. 
    Group by corner solution length, ordered from smallest to largest, and display corner mean, std'''
    start = time.time()

    scrambles = []
    optimals = []
    data = pd.read_csv("scrambles.csv")
    scrambles = data['scramble'].tolist()
    solns = data['soln'].tolist()[:]
    corner_solns = helpers.get_corner_solns(scrambles)

    scrambles.pop() # unfortunately yes
    solns.pop()
    
    if not (len(scrambles) == len(corner_solns) == len(solns)):
        print("error, some scrambles were not solved")
    df = pd.DataFrame({
        "scramble"      : scrambles,
        "corner solution length"    : corner_solns,
        "average solution length"   : solns,
    })
    stats = (df
             .groupby("corner solution length")["average solution length"]
             .agg(["count", "mean", "std"])
             .sort_index()
             .round(3))

    stats.to_csv("corner_stats_ben.csv", index_label="corner solution length ")
    print("it took " + str(time.time() - start))

    return None



    '''
    
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

    
    '''
        



if __name__ == "__main__":
    main() 