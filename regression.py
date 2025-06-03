import helpers
import state
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import time
from sklearn.linear_model import ElasticNetCV 
import csv, pathlib, helpers, state
import pandas as pd

NUM_SCRAMBLES = 1000


def graph_to_vec(G):
    
    ''' Generate the 1x97 sparse row vector for one cube's adjacency graph.
    '''
    rows, cols, data = [],[],[] #define these lists for sparse.csr_matrix
    for c_id, e_plus_8 in G.edges():
        e_id = e_plus_8 -8
        pair_id = c_id * 12 + e_id #create unique id for this corner, edge pair.
        rows.append(0); cols.append(pair_id); data.append(1)
    return sparse.csr_matrix((data, (rows, cols)), shape = (1, 96))


def make_design_matrix(bips) :
    '''takes a list of bipartite domino graphs and produces the design matrix X of the dataset.
    
    Notes:
    - There is one row for each scramble, with columns for each possible corner-edge relation.
    - The matrix is binary; 1 if the scramble has the relationship present, 0 otheriwse. 
    - The matrix is sparse since while there are 96 columns, there are exactly 24 1s per row.
    '''

    vectors = []
    for bip in bips:
        vectors.append(graph_to_vec(bip))
    return sparse.vstack(vectors) #This is X.

def visualize(X):
    plt.figure(figsize=(8, 6))
    plt.spy(X, markersize=3)          # 500 rows × 96 cols
    plt.title("Design matrix – dots are 1’s")
    plt.xlabel("feature index (0‑95)")
    plt.ylabel("scramble index")
    plt.show()

def append_to_csv(filename):
    '''
    add (scramble, soln length) pairs to the specified csv file
    '''
    OUT = pathlib.Path(filename)
    header_written = OUT.exists()      # don’t duplicate header

    with OUT.open("a", newline="") as f:
        writer = csv.writer(f)
        if not header_written:
            writer.writerow(["scramble", "opt_len"])
        for block in range(100): #because nissy is often giving seg faults, compute scrambles in blocks
            scramble_block = helpers.get_dr_scrambles(NUM_SCRAMBLES)
            soln_block = helpers.get_solns(scramble_block)
            writer.writerows(zip(scramble_block, soln_block)) #guarantees same lengths
            print(f"block {block:03d} done")


def regress(X, y):
    '''
    Apply a lasso model to learn the weights of the features.
    X: 
    
    '''



    start = time.time()
    model = ElasticNetCV(cv=5, l1_ratio=0.7).fit(X, y) #lasso model
    w  = model.coef_.reshape(8, 12)    # w[c, e]
    bias = model.intercept_
    np.save("w_bipartite.npy", w)       # easy reload later
    
    w = np.load("w_bipartite.npy")
    plt.figure(figsize=(6, 4))
    plt.imshow(w)                 # default colormap; no explicit colors
    plt.colorbar(label="Weight")
    plt.xlabel("Edge ID (0‒11)")
    plt.ylabel("Corner ID (0‒7)")
    plt.tight_layout()
    print("this took "+ str(time.time() - start))
    plt.show()



def main():
    '''
    
    '''
    #append_to_csv("scrambles.csv") #generate scramble, solution pairs.
    start = time.time()
    df = pd.read_csv("scrambles.csv",
                 names=["scramble", "opt_len"],   # give column names
                 header=None)
    scrambles = df["scramble"].tolist()
    soln_lengths    = df["opt_len"].astype(int).tolist()
    y = np.array(soln_lengths)
    bips = state.Cube.get_bips(scrambles) # finishing this line takes 10.83s
    X = make_design_matrix(bips) # finishing this line takes 100s.

    regress(X, y)
    
    

if __name__ == "__main__":
    main()
