import helpers
import state
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import time

NUM_SCRAMBLES = 20


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


def main():
    '''
    compute data for the linear regression.
    '''
    scrambles = []
    for i in range(400): #because nissy is often giving seg faults, compute scrambles in blocks
        scramble_block = helpers.get_dr_scrambles(NUM_SCRAMBLES)
        scrambles += scramble_block
        print("Done generating a block of scrambles")

    soln_lengths = helpers.get_solns(scrambles)
    scrambles.pop()

    bips = state.Cube.get_bips(scrambles)
    X = make_design_matrix(bips)
    y = np.array(soln_lengths)
    print(len(scrambles))
    print(len(soln_lengths))
    visualize(X)


        
    return None

if __name__ == "__main__":
    main()
