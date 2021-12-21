# return a mackey-glass time series
import matplotlib.pyplot as plt

from reservoirpy.datasets import mackey_glass

def load_test_train_split(
    timesteps = 25000,
    tau = 17,
    train_length = 20000,
    forecast = 10,
    plot = False
    ):

    # create and normalize
    X = mackey_glass(timesteps, tau=tau)
    X = 2 * (X - X.min()) / (X.max() - X.min()) - 1

    X_train, y_train = X[:train_length], X[forecast: train_length + forecast]
    X_test, y_test = X[train_length: -forecast], X[train_length + forecast:]

    if plot:
        sample = 500
        fig = plt.figure(figsize=(15, 5))
        plt.plot(X_train[:sample], label="Training data")
        plt.plot(y_train[:sample], label="True prediction")
        plt.legend()

        plt.show()
    
    return (X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    load_test_train_split(plot=True)