from pathlib import Path
import random

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

filepath = Path("deepCA/output/hyper2_scalefree")
show = False

# get the data
pkl_files = list(Path(filepath).glob("*.pkl"))
if len(pkl_files) == 0:
    print(f"No files found in '{filepath}'")
    exit()
else: 
    print(f"Found {len(pkl_files)} .pkl files")

# make a unique save folder
savepath = Path(f"deepCA/figures/{filepath.stem}")
savepath.mkdir(parents=True, exist_ok=True)

# join datasets from datapath folder
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

print(data)

def make_plot(x, y):
    ax = sns.scatterplot(
        x=data[x],
        # y=data["input_connectivity"],
        y=data[y],
        hue=data["nrmse"],
        size=data["r2"],
    )

    if (y == "regularization"):
        plt.yscale("log")
    elif (x == "regularization"):
        plt.xscale("log")

    # save and/or display
    fig = ax.get_figure()
    # fig.set_size_inches(6, 6)
    fig.savefig(Path(f"{savepath}/{x}_vs_{y}"))
    if show:
        plt.show()
    plt.close()

""" leak rate vs regularization """
make_plot("leak_rate", "regularization")
make_plot("leak_rate", "input_connectivity")
make_plot("input_connectivity", "regularization")