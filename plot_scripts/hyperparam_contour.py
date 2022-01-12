
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm

# custom functions
from utils import get_data
from utils import clean_data

""" parameters """
filepath = Path(r"output/data/celegans_random_hyperparam_sr4") # where to find data
x_label = "input_connectivity" # x variable
y_label = "leak_rate" # y variable
z_labels = ["nrmse", "r2"] # quality measure
controlled_variable = "regularization" # a third variable we will control for in subplots
# levels = 12 # number of levels in the contour map
n_levels = 10 # levels in the contour map

""" program """
# make a unique save folder
savepath = Path(f"output/figures/{filepath.stem}")
savepath.mkdir(parents=True, exist_ok=True)

# load and clean data
data = get_data(filepath)
data = clean_data(data) # remove entries with crazy values (nrmse > 1 and r2 < 0)

controls = data[controlled_variable].unique()
controls.sort()

# iterate and save figure for the relevant combinations
for topology in tqdm(data["filename:topology_type"].unique()): # all the different adj. matrices in the input folder
    for z_label in z_labels: # r2 and nrmse
        fig, axs = plt.subplots(ncols=3, nrows=2)
        for i, controlled_value in enumerate(controls): # control for a third variable in each plot
            df = data.loc[ # exclude some data for this specific plot
                (data[controlled_variable] == controlled_value) &
                (data["filename:topology_type"] == topology)
                ]

            # do unecessary memory allocation to make code prettier
            x = df[x_label]
            y = df[y_label]
            z = df[z_label]

            # skip if data is missing 
            if len(x) < 3:
                print(f"\"{topology}\" skipped {controlled_value}")
                continue

            # set esthetics specific for the quality measure z
            if z_label == "r2":
                levels = np.linspace(0.3,0.8,n_levels+1) # number of levels in the contour map
                cmap = "RdYlGn" # for r2 higher is better (green)
                vmin = 0.4 # min limits for color, experiment and adjust
                vmax = 0.8 # max limits for color

            else:
                cmap = "RdYlGn_r" # for nrmse lower is better
                levels = np.linspace(0.0,0.5,n_levels+1) # number of levels in the contour map
                vmin = 0.0
                vmax = 0.4

            # iterate over the axs (2 x 3 matrix)
            row = i // 3
            col = i % 3

            # make contour map
            axs[row, col].tricontour(
                x, y, z,
                levels=levels,
                colors="k",
                # vmin=vmin,
                # vmax=vmax,
            )
            # fill with pretty colors
            contour = axs[row, col].tricontourf(
                x, y, z,
                levels=levels,
                cmap=cmap,
                # vmin=vmin,
                # vmax=vmax,
            )

            # title and label info 
            fig.colorbar(contour, ax=axs[row, col])
            axs[row, col].plot(x, y, "ko", ms=3)
            axs[row, col].set_xlabel(x_label)
            axs[row, col].set_ylabel(y_label)
            forecast = df["forecast"].unique()[0]
            axs[row, col].set_title(f"{controlled_variable}={controlled_value} forecast={forecast}")

        # save the figure
        fig.suptitle(f"{topology} {z_label}")
        fig.set_size_inches(15, 10)
        fig.savefig(Path(f"{savepath}/{z_label}_{topology}_{x_label}_vs_{y_label}.png"))
        plt.close() # remember trash disposal <:o)

print("Done")