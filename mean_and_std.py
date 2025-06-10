import csv
import statistics
import pathlib

def compute_mean_std(path):
    values = []
    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            values.append(int(row[1]))
    mean = statistics.mean(values)
    stdev = statistics.stdev(values)
    return mean, stdev
print(compute_mean_std(pathlib.Path("scrambles.csv")))

#output: (13.579197902062791, 1.0858620331962971)