from pathlib import Path
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

filepath = Path("deepCA/output/hyper2_scalefree")
# data = pd.read_pickle(filepath)


# get the data
pkl_files = list(Path(filepath).glob("*.pkl"))
if len(pkl_files) == 0:
    print(f"No files found in '{filepath}'")
    exit()

# make a unique save folder
savepath = Path(f"deepCA/figures/{filepath.stem}")
# savepath = Path(f"deepCA/figures/{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}")
savepath.mkdir(parents=True, exist_ok=True)

# join datasets from filepath folder
data = pd.concat(
    [pd.read_pickle(f) for f in pkl_files],
    axis=0,
    join="outer",
    # ignore_index=True,
    keys=[f.stem for f in pkl_files],
)


x = len(data[data["nrmse"] > 1])
print(f"Dropping {x} tests with NRMSE > 1")
data.drop(data[data["nrmse"] > 1].index, inplace=True)
x = len(data[data["r2"] < 0])
print(f"Dropping {x} tests with R2 < 0")
data.drop(data[data["r2"] < 0].index, inplace=True)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

x_dependent = "leak_rate"
y_dependent = "regularization"
# y_dependent = "input_connectivity"
# controlled = "regularization"
# controlled_value = random.choice(data[controlled].unique())
# data = data.loc[
#     (data[controlled] == controlled_value)
# print("Control:", controlled_value)
ax.scatter(
    data["leak_rate"],
    data["input_connectivity"],
    # data["regularization"],
    data["r2"],
)
plt.gca().update(dict(title=filepath.stem, xlabel='leak rate', ylabel='input conn'))#, ylim=(0,10)))


# X, Y = np.meshgrid(data[x_dependent].unique(), data[y_dependent].unique())
# print(X)
# print(X.shape)



ax.legend()
plt.show()