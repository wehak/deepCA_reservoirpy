from pathlib import Path
import sys
import datetime
import re
import getopt
import os
import time

import pandas as pd
from tqdm import tqdm

from util.evaluation import initiate_train_and_test_ESN

""" parameters """
# limit number of matrices tested of each type
trim_limit = None

# path to folder containing all relevant adjacency matrices
matrix_folder_path = Path("input\celegans131matrix")

# what datasets to test
datasets = ["mg", "santafe", "lorenz", "ett"] # test all datasets
# datasets = ["mg"] # only mackey-glass

# what forecast levels to test for each dataset
forecast_levels = {
    "mg" : [100, 200, 400],
    "santafe" : [5, 10, 20],
    "lorenz" : [10, 20, 40],
    # "ett" : [1, 2, 4] # not working?
}

""" parse parameters from terminal """
# get input args
low = None
high = None
folder_name = None

try:
    opts, args = getopt.getopt(sys.argv[1:], "l:h:f:")
except:
    print("-l <low> -h <high>")

for opt, arg in opts:
    if opt == "-l":
        low = int(arg)
    elif opt == "-h":
        high = int(arg)
    elif opt == "-f":
        folder_name = arg

if low is None or high is None:
    print("Must have a range")
    exit()
elif folder_name is None:
    print("Must have a output data folder name -f <name>")
    exit()

# where to save the output files
save_folder = Path(f"output/data/{folder_name}")

""" program """

# create output folder if not exist
save_folder.mkdir(parents=True, exist_ok=True)

# find adjacency matrices. assume 2 level structure
# (alternatively, use rglob() instead of glob() to search folder and subfolders recursively)
adjacency_matrices = []
for folder_lvl_1 in matrix_folder_path.glob("*"):
    for folder_lvl_2 in folder_lvl_1.glob("*"):
        matrices = folder_lvl_2.glob("*.csv")
        for f in matrices:
            # print(f.stem)
            m = re.search(r'\d+$', f.stem)
            n = int(m.group()) if m else 0
            if (trim_limit is None):
                adjacency_matrices.append(f)
            elif (n <= trim_limit):
                adjacency_matrices.append(f)

# print start-status to terminal
n = len(adjacency_matrices[low:high])
p_id = os.getpid()
print(f"{datetime.datetime.now().strftime('%H:%M:%S')}: #{p_id} processing {n} matrices [{low}, {high})")

# start the training
t_start = time.time()
results = []
#for i, adjacency_matrix in enumerate(adjacency_matrices[low:high], start=1):
for i, adjacency_matrix in enumerate(tqdm(adjacency_matrices[low:high]), start=1):
    for dataset in datasets:
        for forecast in forecast_levels[dataset]:
            results.append(initiate_train_and_test_ESN(
                adjacency_matrix,
                dataset=dataset,
                forecast=forecast,
                leak_rate=0.25,
                input_scaling=1.0,
                input_connectivity=0.8,
                regularization=1e-6,
                seed=12345
                ))
    # print(f"#{os.getpid()}: Trained \"{adjacency_matrix.stem}\" in {time.time() - t_start:.2f} s ({i+low}/{n+low})")

# training complete, print end-status to terminal
print(f"{datetime.datetime.now().strftime('%H:%M:%S')}: #{p_id} trained [{low}, {high}) in {str(datetime.timedelta(seconds=time.time() - t_start)).split('.')[0]}")

# convert to pandas dataframe to save data
df = pd.DataFrame(results)

# save results
filename = Path(f"{save_folder}/{p_id}_{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.pkl")
df.to_pickle(filename)

# print?
# df.sort_values(by=["r2"], inplace=True, ascending=True)
# print(df)
# print(df[["r2", "nrmse", "ev"]])
# print("Mean", round(df["r2"].mean(), 3))
