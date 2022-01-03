from pathlib import Path

import pyreadr
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

"""
Returns train and test data of the Santa Fe laser dataset.
In the original competition the forecast was t+100.
"""
def load_test_train_split(
    timesteps = 1100,
    train_length = 900,
    forecast = 1,
    plot=False
    ):

    # path to laser data
    data_A = Path("util/datasets/SantaFe.A.rda")
    data_B = Path("util/datasets/SantaFe.A.cont.rda")

    # load data
    santafe_A = list(pyreadr.read_r(data_A)["SantaFe.A"]["V1"]) + list(pyreadr.read_r(data_B)["SantaFe.A.cont"]["V1"])
    santafe_A = np.array(santafe_A).reshape(-1, 1)

    # normalize to [-1, 1]
    scaler = MinMaxScaler(
        feature_range=(-1, 1),
        # copy=False,
        )
    scaler.fit(santafe_A)
    santafe_A = scaler.transform(santafe_A)

    # split into training and testing data
    X_train = np.array(santafe_A[:train_length])#.reshape(-1, 1)
    y_train = np.array(santafe_A[forecast: train_length + forecast])#.reshape(-1, 1)

    X_test = np.array(santafe_A[train_length: -forecast])#.reshape(-1, 1)
    y_test = np.array(santafe_A[train_length + forecast:])#.reshape(-1, 1)

    # plot the data if desired
    if plot:
        plt.plot(range(0, train_length), X_train, ls="-", label="X train")
        plt.plot(range(forecast, train_length + forecast), y_train, ls=":", label="y train")
        plt.plot(range(train_length, timesteps - forecast), X_test, linestyle='-', label="X test")
        plt.plot(range(train_length + forecast, timesteps), y_test, ls=':', label="y test")
        plt.legend()
        plt.show()

    return (X_train, X_test, y_train, y_test)

# run test
if __name__ == "__main__":
    load_test_train_split(plot=True)