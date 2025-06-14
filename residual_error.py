import csv
import statistics
import pathlib
import vfmc
from scipy.stats import pearsonr

def baseline_error(path):
    '''
    compute error stats for guessing simply the mean optimal dr->soln length.
    '''
    values = []
    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            values.append(int(row[1]))

    mean = statistics.mean(values)
    stdev = statistics.stdev(values)
    baseline_mae = statistics.mean(abs(y - mean) for y in values) #to be compared with the model's mae
    baseline_mse = statistics.mean((y - mean) ** 2 for y in values)
    corr = "undefined"
    return mean, stdev, baseline_mae, baseline_mse, corr

def corners_only_error(path):
    '''
    compute error stats for guessing corner equivalence class's mean optimal length.
    - corner solution must be >= 3 moves.
    '''
    




    
def subset_only_error(path):
    '''
    compute error stats for guessing the htr subset's mean optimal length 
    '''

def blocks_only_error(path):
    '''
    compute error stats for guessing mean optimal + block penalties learned in blockiness.py.
    
    '''
def subset_and_blocks_error(path):
    '''
    comptue error for guessing htr subset mean + block penalties
    '''

def subset_corner_blocks_error(path):
    '''
    compute full error stats for final scoring function using subset, corners, and blocks.
    '''





print("\nBaseline mean, stdev, mae, mse: \n" + str(baseline_error(pathlib.Path("scrambles.csv"))) + "\n")

#output: (13.579197902062791, 1.0858620331962971, 0.8900005218446605)

