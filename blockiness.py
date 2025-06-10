
''' Module for learning the weights of various types of blocks on general scrambles.

Features are computed using functions in state.py and helpers.py as follows:

TEST_SCRAMBLE = "R2 U R2 D F2 B2 D' L2 D R2"
helpers.get_subset(TEST_SCRAMBLE)
mystate = state.State()
mystate.amt_lines()
mystate.amt_pairs()
mystate.amt_Ls()
#corner length is also possible

'''

import numpy as np
import state
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error
import pathlib
import csv
import json


SEARCH_SPACE = [ #combinations of features to iterate through 
    {
        "use_pairs": 1,
        "use_lines": 1,
        "use_Ls": 1,
        "alpha": [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
    },
    {
        "use_pairs": 1,
        "use_lines": 1,
        "use_Ls": 0,
        "alpha": [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
    },
    {
        "use_pairs": 1,
        "use_lines": 0,
        "use_Ls": 1,
        "alpha": [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
    },
    ]


def features_count(scramble) -> np.ndarray:
    cube = state.State()
    cube.apply(scramble)
    a, b          = cube.amt_pairs()        # style‑a, style‑b pairs
    goodL, badL   = cube.amt_Ls()
    lines         = cube.amt_lines()
    return np.array([a, b, goodL, badL, lines], dtype=np.int8)

def setup():
    df   = pd.read_csv("scrambles.csv", names=["scramble", "opt_len"], nrows=400000,
                   header=None)                       
    X    = np.vstack([features_count(s) for s in df["scramble"]])
    y    = df["opt_len"].to_numpy(dtype=np.int8)
    return X, y

def plot_predictions(y_true, y_pred, title="Predicted vs Actual"):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')  
    plt.xlabel("Actual Optimal Length")
    plt.ylabel("Predicted Optimal Length")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def make_design(scrambles, cfg):
    ''' 
    builds and returns the design matrix X with features chosen in cfg dict which maps feature -> on/off
    
    '''
    features = []
    for scramble in scrambles: 
        #add a row to the matrix

        cube = state.State()
        cube.apply(scramble)
        row = []
        if cfg["use_pairs"]: #extend row by both types of pairs
            row.extend(cube.amt_pairs())
        if cfg["use_lines"]:
            row.append(cube.amt_lines())
        if cfg["use_Ls"]:
            row.extend(cube.amt_Ls())
        features.append(row)
    return np.vstack(features, dtype = np.int8)

def run(cfg, scrambles_trainvalid, solns_trainvalid, seed =26):
    ''' 
    Run a linear regression with elastic net regularization using the specified features and hyperparams in cfg
    Return mean actual error and the model object

    '''


    scrambles_tr, scrambles_valid, y_tr, y_valid = train_test_split(
        scrambles_trainvalid, solns_trainvalid, test_size=0.10, random_state=seed) #9% of all scrambles become 

    # feature build
    X_train  = make_design(scrambles_tr,  cfg) 
    X_valid = make_design(scrambles_valid, cfg)
    model = make_pipeline(
        StandardScaler(),
        LassoCV(
            alphas=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1], 
            cv=5,                  
            random_state=seed,
            n_jobs=-1,             
            max_iter=10_000 
        ))

    model.fit(X_train, y_tr)
    val_mae = mean_absolute_error(y_valid, model.predict(X_valid))
    return val_mae, model



def main():
    '''
    Iterates through specified combinations of hyperparameters and features, 
    picking the model that performs the best on the validation dataset. Runs said model
    on the test set and prints the learned weights of the features. 
    
    '''


    df   = pd.read_csv("scrambles.csv", names=["scramble", "opt_len"], nrows=500000,
                   header=None)
    all_scrambles = df["scramble"].tolist() 
    all_solns = df["opt_len"].to_numpy(dtype=np.int8) 

    #allocate 10% of data to final testing, validation will be further sliced from training data
    scrambles_trainvalid, scrambles_test, solns_trainvalid, y_test = train_test_split(
        all_scrambles, all_solns, test_size = 0.10, random_state = 42 
    ) 

    #to log performance of each combination of hyperparams
    logfile = pathlib.Path("experiments_log.csv") 
    if not logfile.exists():
        with logfile.open("w", newline="") as f:
            csv.writer(f).writerow(["cfg", "val_mae"])

    #initialize bests
    best_mae, best_cfg, best_model = np.inf, None, None

    #Run on each configuration, pick the best model on the validation set to move on to the test set
    for cfg in SEARCH_SPACE:
        val_mae, model = run(cfg, scrambles_trainvalid, solns_trainvalid)
        print(cfg, "gives mean actual error ", val_mae)

        with logfile.open("a", newline="") as f:
            #log the config dict and error
            csv.writer(f).writerow([json.dumps(cfg), val_mae]) 

        if val_mae < best_mae:
            best_mae, best_cfg, best_model = val_mae, cfg, model 

    #Run the best model on the test set
    X_test = make_design(scrambles_test, best_cfg) 
    if best_model != None:

        test_mae = mean_absolute_error(y_test, best_model.predict(X_test)) #final result
        print("\nBest config:", best_cfg)
        print("Validation MAE:", best_mae)
        print("**Final Test MAE:**", test_mae)

        #print final weights
        enet = best_model.named_steps["lassocv"]
        feature_names = ["pair_a", "pair_b", "goodL", "badL", "lines"]

        print("\nLearned weights:")
        for name, weight in zip(feature_names, enet.coef_): #enet.coef is the weights
            print(name + ": " + str(weight))

        print("Bias: " + str(enet.intercept_))


    
if __name__ == "__main__":
    main()

