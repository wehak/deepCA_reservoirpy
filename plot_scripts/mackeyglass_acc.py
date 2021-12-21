from pathlib import Path
import datetime

from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def make_plot(datapath, show=False):

    # get the data
    pkl_files = list(Path(datapath).glob("*.pkl"))
    if len(pkl_files) == 0:
        print(f"No files found in '{datapath}'")
        exit()
    
    # make a unique save folder
    savepath = Path(f"deepCA/output/figures/{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}")
    savepath.mkdir(parents=True, exist_ok=True)
    
    for file in tqdm(pkl_files):
        df = pd.read_pickle(file)

        # print(type(df["filename:topology_type"].unique()))

        # make plot
        ax = sns.boxplot(
            x="filename:topology_type",
            y="r2",
            hue="filename:sr",
            data=df,
            order=np.sort(df["filename:topology_type"].unique()),
            hue_order=np.sort(df["filename:sr"].unique()),
        )

        # include hyperparameters in title
        leak_rate = df["leak_rate"].iat[0]
        input_scaling = df["input_scaling"].iat[0]
        input_conn = df["input_connectivity"].iat[0]

        # esthetics
        ax.set(
            title=f"leak_rate={leak_rate}, input_scaling={input_scaling}, input_conn={input_conn}, ",
            xlabel="Topology type",
            ylabel="R-squared",
            yticks=(0, 0.5, 1),
            
        )

        ax.legend(
            title="Spectral radius",
        )

        # save and/or display
        fig = ax.get_figure()
        fig.savefig(Path(f"{savepath}/{file.stem}-mackeyglass"))
        if show:
            plt.show()
        plt.close()

# make plots
if __name__ == "__main__":
    make_plot("deepCA/output/data/celegans131_1")