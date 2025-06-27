import csv
import statistics
import pathlib
import vfmc
from scipy.stats import pearsonr
import pandas as pd
import dr_to_solved.helpers as helpers
import numpy as np
from dr_to_solved.learn_score import SUBSET_MEANS

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
    
    residuals = []
    data = pd.read_csv("labelled_drs.csv")
    print(data.columns.tolist())

    scrambles = data['scramble'].tolist()
    solns = data['soln'].tolist()

    corner_solns = helpers.get_corner_solns(scrambles)
    scrambles.pop()
    solns.pop()

    data = pd.read_csv("corner_stats_ben.csv")
    data.set_index('corner solution length', inplace=True)

    for i in range(len(scrambles)): #len(scrambles)
        corner_soln = corner_solns[i]
        guess = (data.loc[corner_soln, 'mean'])
        answer = solns[i]
        print(guess)
        print(corner_soln)
        print(answer)
        residuals.append(guess - answer)

    
    corner_data = pd.read_csv("corner_stats_ben.csv")
    corner_data.columns = corner_data.columns.str.strip()
    corner_data.set_index('corner solution length', inplace=True)

    residuals = []

    for i in range(len(scrambles)):
        corner_soln = corner_solns[i]
        guess = corner_data.loc[corner_soln, 'mean']
        answer = solns[i]
        residuals.append(guess - answer)

    residuals = np.array(residuals)
    mse = np.mean(residuals**2)
    mae = np.mean(np.abs(residuals))

    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")

    with open("residual.txt", "a") as f:
        f.write(f"Guessing based on corners solution length only; MSE: {mse:.4f}, MAE: {mae:.4f}\n")
    

def subset_only_error(path):
    '''
    compute average difference between subset mean and corner optimal mean 
    '''
    df = pd.read_csv(path)
    df["subset_dev"] = df["soln"] - df["subset"].map(SUBSET_MEANS)
    return df["subset_dev"].abs().mean()


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

print(subset_only_error("labelled_drs.csv"))


print("\nBaseline mean, stdev, mae, mse: \n" + str(baseline_error(pathlib.Path("labelled_drs.csv"))) + "\n")

#output: (13.579197902062791, 1.0858620331962971, 0.8900005218446605)

