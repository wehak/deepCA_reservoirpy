from pathlib import Path
import datetime

from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def make_plots(datapath, measures, show=False):

    # get the data
    pkl_files = list(Path(datapath).glob("*.pkl"))
    if len(pkl_files) == 0:
        print(f"No files found in '{datapath}'")
        exit()
    
    # make a unique save folder
    savepath = Path(f"output/figures/{datapath.stem}")
    # savepath = Path(f"deepCA/figures/{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}")
    savepath.mkdir(parents=True, exist_ok=True)
    
    # join datasets from datapath folder
    df = pd.concat(
        [pd.read_pickle(f) for f in pkl_files],
        axis=0,
        join="outer",
        # ignore_index=True,
        keys=[f.stem for f in pkl_files],
    )

    print("Matrices =", len(df["filename"].unique()))

    # delete "Real_Network" (duplicate of "Real Network")
    delete_idxs = df[ df["filename:topology_type"] == "Real_Network"].index
    df.drop(delete_idxs, inplace=True)

    for dataset in df["dataset"].unique(): # look though all datasets
        for forecast in df["forecast"].loc[(df["dataset"] == dataset)].unique(): # look though all forecast levels for each dataset type
            for measure in measures: # looks through all given quality measures

                # make plot
                ax = sns.boxplot(
                    x="filename:topology_type",
                    y=measure,
                    hue="filename:sr",
                    data=df.loc[
                        (df["dataset"] == dataset) &
                        (df["forecast"] == forecast),
                        ],
                    order=np.sort(df["filename:topology_type"].unique()),
                    hue_order=np.sort(df["filename:sr"].unique()),
                )

                # include hyperparameters and mean in title
                leak_rate = df["leak_rate"].iat[0]
                input_scaling = df["input_scaling"].iat[0]
                input_conn = df["input_connectivity"].iat[0]
                regularization = df["regularization"].iat[0]
                mean = df[measure].loc[
                        (df["dataset"] == dataset) &
                        (df["forecast"] == forecast),
                        ].mean()
                max = df[measure].loc[
                        (df["dataset"] == dataset) &
                        (df["forecast"] == forecast),
                        ].max()
                min = df[measure].loc[
                        (df["dataset"] == dataset) &
                        (df["forecast"] == forecast),
                        ].min()
                # print(measure, dataset, forecast, mean, max, min)

                # esthetics
                ax.set(
                    title=f"leak_rate={leak_rate}, input_scaling={input_scaling}, input_conn={input_conn}, reg={regularization}\ndataset={dataset}, forecast={forecast}, avg. {measure}={mean:.2f}, max={max:.2f}, min={min:.2f}",
                    xlabel="Topology type",
                    ylabel=measure.upper(),
                    yticks=(0, 0.5, 1),
                )
                ax.set_ylim(0, 1)

                ax.legend(
                    title="Spectral radius",
                )

                # save and/or display
                fig = ax.get_figure()
                fig.set_size_inches(12, 6)
                fig.savefig(Path(f"{savepath}/{measure}-{dataset}-{forecast}"))
                if show:
                    plt.show()
                plt.close()

# make plots
if __name__ == "__main__":
    make_plots(
        Path("output\data\demo"),
        measures=["r2", "nrmse"])