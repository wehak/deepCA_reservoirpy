from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

"""
Returns train and test data for the ETT dataset

"""
def load_test_train_split(
    timesteps = 25000,
    train_length = 20000,
    forecast = 10,
    dataset = "ETTm1",
    plot=False
    ):

    # load and crop data
    data_path = Path(f"deepCA/datasets/ETT-small/{dataset}.csv")
    X = pd.read_csv(data_path)
    X.drop(columns=["date"], inplace=True)
    X = X.to_numpy()
    X = X[:timesteps]

    # normalize
    scaler = MinMaxScaler(
        feature_range=(-1, 1),
    )
    scaler.fit(X)
    X = scaler.transform(X)

    # split X and y
    y = X[: ,   -1]
    X = X[: ,   :-1]

    print(X[0])
    print(X.shape)
    print(y[0])
    print(y.shape)

    X_train = np.array(X[:train_length])#.reshape(-1, 1)
    y_train = np.array(y[forecast: train_length + forecast])#.reshape(-1, 1)

    X_test = np.array(X[train_length: -forecast])#.reshape(-1, 1)
    y_test = np.array(y[train_length + forecast:])#.reshape(-1, 1)


    if plot:
        plt.plot(range(0, train_length), X_train, ls="-", label="X train")
        plt.plot(range(forecast, train_length + forecast), y_train, ls=":", label="y train")
        plt.plot(range(train_length, timesteps - forecast), X_test, linestyle='-', label="X test")
        plt.plot(range(train_length + forecast, timesteps), y_test, ls=':', label="y test")
        plt.legend()
        plt.show()

    return (X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    load_test_train_split(plot=True)