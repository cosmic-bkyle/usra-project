'''
add remaining features to current 500k and then add another 500k for 1 million entries in a parquet file.
'''
import pandas as pd
import dr_to_solved.helpers as helpers
import numpy as np
import csv
import time
import pathlib
import dr_to_solved.state as state


def extract_features(row):
    
    cube = state.State()
    cube.apply(row['scramble'])
    a, b = cube.amt_pairs()
    lines = cube.amt_lines()
    goodL, badL = cube.amt_Ls()
    return pd.Series({
        "pairs_a": a,
        "pairs_b": b,
        "lines":   lines,
        "good_L":  goodL,
        "bad_L":   badL
    })

def addrows():
    start = time.time()
    old_data = pd.read_csv("scrambles.csv", nrows = 517842)
    scrambles = old_data['scramble'].tolist()
    corner_series = pd.Series(helpers.get_corner_solns(scrambles), name = "corners")
    subset_series = pd.Series(helpers.get_subsets(scrambles), name = "subset")
    blocks = old_data.apply(extract_features, axis=1)


    df_new = pd.concat([old_data, corner_series, subset_series, blocks], axis=1)

    df_new.to_csv("feature_table.csv", index=False)

    min_val = df_new["soln"].min()
    count = (df_new["soln"] == min_val).sum()
    print(f"There are {count} scrambles with solution length = {min_val}")

    print(time.time() - start)

def add_block(n):
    scrambles = helpers.get_dr_scrambles(n)
    solns = helpers.get_solns(scrambles)
    corner_solns = helpers.get_corner_solns(scrambles)
    scrambles.pop() # defensive
    subsets = helpers.get_subsets(scrambles)

    rows = []

    for i, scramble in enumerate(scrambles):
        cube = state.State()
        cube.apply(scramble)
        a, b = cube.amt_pairs()
        lines = cube.amt_lines()
        goodL, badL = cube.amt_Ls()

        row = {
            "scramble": scramble,
            "soln": solns[i],
            "corners": corner_solns[i],
            "subset": subsets[i],
            "pairs_a": a,
            "pairs_b": b,
            "lines": lines,
            "good_L": goodL,
            "bad_L": badL
        }

        rows.append(row)
    df_to_add = pd.DataFrame(rows)

    parquet_path = pathlib.Path("labelled_drs.parquet")
    if parquet_path.exists():
        existing_df = pd.read_parquet(parquet_path)
        combined_df = pd.concat([existing_df, df_to_add], ignore_index=True)
    else:
        combined_df = df_to_add

    combined_df.to_parquet(parquet_path, index=False)


def main():
    add_block(1000001) # will give 1m labelled drs


if __name__== "__main__":
    main()