from pathlib import Path
import random
import datetime
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

from util.evaluation import initiate_train_and_test_ESN


# input
test_matrix = Path("deepCA/adjacency_matrices/batch 2/Network outputs2/SR_0.9/Random/SR_0.9Random1.csv")
# test_matrix = Path("deepCA/adjacency_matrices/batch 2/Network outputs2/SR_0.9/Scalefree/SR_0.9Scalefree1.csv")


# hyperparameter ranges:
n = 7
# input matrix
input_bias_range = [True, False]
input_scaling_range = np.linspace(0.1, 1.1, num=n)

# training
leak_rate_range = np.linspace(0.2, 1, num=n, endpoint=True)
ridge_reg_range = np.geomspace(1e-5, 1e-9, num=n, endpoint=True)

results = []
# for ridge_reg in tqdm(ridge_reg_range):
for leak_rate in leak_rate_range:
    for input_scaling in input_scaling_range:
        results.append(initiate_train_and_test_ESN(
            test_matrix,
            dataset="mg",
            forecast=200,
            leak_rate=leak_rate,
            input_scaling=input_scaling,
            regularization=0,
            seed=1234,
        ))

df = pd.DataFrame(results)
print(df)


# output
save_folder = Path(f"deepCA/output/hyperparams/{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}")
save_folder.mkdir(parents=True, exist_ok=True)
filename = Path(f"{save_folder}/{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.pkl")
df.to_pickle(filename)