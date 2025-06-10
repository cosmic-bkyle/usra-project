import csv
import statistics
import pathlib

def compute_mean_std_and_baseline_mae(path):
    values = []
    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            values.append(int(row[1]))

    mean = statistics.mean(values)
    stdev = statistics.stdev(values)
    baseline_mae = statistics.mean(abs(y - mean) for y in values) #to be compared with the model's mae
    
    return mean, stdev, baseline_mae

print(compute_mean_std_and_baseline_mae(pathlib.Path("scrambles.csv")))

#output: (13.579197902062791, 1.0858620331962971, 0.8900005218446605)