from pathlib import Path
import random
import datetime
import sys
import os
import time
import getopt

import numpy as np
import pandas as pd
from tqdm import tqdm

import seaborn as sns
import matplotlib.pyplot as plt

from util.evaluation import initiate_train_and_test_ESN

"""parameters """
adjacency_matrices = Path("/home/wehak/data/deepCA/adjacency matrices/celegans131matrix")

# training datasets
datasets = ["mg"]
forecast_levels = {
    "mg" : [200],
    # "mg" : [50, 100, 200, 400],
    "santafe" : [2, 4, 8, 16, 32],
}

# hyperparameter ranges:
n = 200
only_plot = False # True for viewing parameter distribution only, no training performed

leak_rate_low = 0.0
leak_rate_high = 1.0

input_conn_low = 0.1
input_conn_high = 1.0

# input_scaling_low = 0.1
# input_scaling_high = 1.1

regularization_low = 1e-8
regularization_high = 1e-4

""" program """
# get args
folder_name = None
filename = None
try:
    opts, args = getopt.getopt(sys.argv[1:], "i:f:")
except:
    print("-i <input matrix> -f <folder name>")

for opt, arg in opts:
    if opt == "-f":
        folder_name = arg
    elif opt == "-i":
        filename = arg

if (folder_name is None) and (only_plot is False):
    print("Must have a folder name -f <name>")
    exit()
elif (filename is None):
    print("Must have a input matrix -i <input matrix>")
    exit()
save_folder = Path(f"deepCA/output/data/{folder_name}")
save_folder.mkdir(parents=True, exist_ok=True)

# input
# test_matrix = Path("deepCA/adjacency_matrices/batch 2/Network outputs2/SR_0.9/Random/SR_0.9Random5.csv")
# test_matrix = Path("deepCA/adjacency_matrices/batch 2/Network outputs2/SR_0.9/Scalefree/SR_0.9Scalefree1.csv")
# test_matrix = Path("deepCA/adjacency_matrices/batch 2/Network outputs2/SR_0.9/RealNetwork/SR_0.9RealNetwork1.csv")

# find all .csv files in filepath folder and subfolders
test_matrix = list(Path(adjacency_matrices).rglob(f"{filename}.csv"))
if len(test_matrix) == 0:
    print(f"Error: \"{filename}\" not found")
elif len(test_matrix) > 1:
    print(f"Error: Found {len(test_matrix)} duplicates of \"{filename}\"")
else:
    test_matrix = test_matrix[0]

# input matrix
# input_scaling_range = np.random.uniform(input_scaling_low, input_scaling_high, n)
input_connectivity_range = np.random.uniform(input_conn_low, input_conn_high, n)

# training
leak_rate_range = np.random.uniform(leak_rate_low, leak_rate_high, n)
# ridge_reg_range = np.random.uniform(regularization_low, regularization_high, n)

fraction = (regularization_high / regularization_low)
decimal_places = abs(int(f'{fraction:e}'.split('e')[-1])) + 1
regs = np.geomspace(regularization_low, regularization_high, num=decimal_places, endpoint=True)
ridge_reg_range = [random.choice(regs) for _ in range(n)]

# dont run tests, just plot the distribution
if only_plot:
    sns.scatterplot(
        x=input_connectivity_range,
        y=leak_rate_range,
        hue=ridge_reg_range,
    )
    plt.show()

# run the actual training
else:
    p_id = os.getpid()
    print(f"{datetime.datetime.now().strftime('%H:%M:%S')}: #{p_id} processing {n} tests on \"{test_matrix.stem}\"")
    t_start = time.time()
    results = []
    for i in range (n):
        # t_start = time.time()
        for dataset in datasets:
            for forecast in forecast_levels[dataset]:
                results.append(initiate_train_and_test_ESN(
                    test_matrix,
                    dataset=dataset,
                    forecast=forecast,
                    leak_rate=leak_rate_range[i],
                    input_scaling=1,
                    input_connectivity=input_connectivity_range[i],
                    regularization=ridge_reg_range[i],
                    seed=1234
                    ))
        # print(f"#{os.getpid()}: Trained \"{adjacency_matrix.stem}\" in {time.time() - t_start:.2f} s ({i+low}/{n+low})")

    print(f"{datetime.datetime.now().strftime('%H:%M:%S')}: #{p_id} finished in {time.time() - t_start:.2f} s")

    # save results
    df = pd.DataFrame(results)
    filename = Path(f"{save_folder}/{p_id}_{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.pkl")
    df.to_pickle(filename)