from pathlib import Path
import pandas as pd
from pandas.core.reshape.concat import concat
import numpy as np
from tqdm import tqdm

def get_data(filepath):
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

    return data

def clean_data(df):
    x = len(df[df["nrmse"] > 1])
    print(f"Dropping {x} tests with NRMSE > 1")
    df.drop(df[df["nrmse"] > 1].index, inplace=True)
    x = len(df[df["r2"] < 0])
    print(f"Dropping {x} tests with R2 < 0")
    df.drop(df[df["r2"] < 0].index, inplace=True)
    return df

# check for duplicates
def control_uniqueness(folderpath):
    files = list(Path(folderpath).rglob("*.csv"))
    if len(files) == 0:
        print(f"No files found in \"{folderpath}\"")
        return False
    else:
        matrices = [np.genfromtxt(f, delimiter=",") for f in files]
        duplicates = {"n": 0, "paths": []}

        n = len(matrices)
        if n != len(files):
            print(f"Error: Found {n} matrices in {len(files)} .csv files")
            return False
        else:
            print(f"Controlling {n} matrices")
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    else:
                        if (matrices[i] == matrices[j]).all():
                            duplicates["n"] += 1
                            duplicates["paths"].append(files[i].stem)

            # print(f"Found {duplicates['n']} duplicate matrices.")
            # for f in duplicates["paths"]:
            #     print(f)
            return duplicates["n"]
                    