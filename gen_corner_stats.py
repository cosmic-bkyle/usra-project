'''
Generate corner solution length x solution mean.
'''
import pandas as pd

def main():
    df = pd.read_parquet("labelled_drs.parquet")
    grouped = df.groupby("corners")["soln"]
    summary = grouped.agg(["count", "mean"]).reset_index()
    summary.to_csv("corner_solution_summary.csv", index=False)
if __name__ == "__main__":
    main()
