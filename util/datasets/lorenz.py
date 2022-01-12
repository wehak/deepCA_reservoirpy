import numpy as np
import matplotlib.pyplot as plt

from reservoirpy.datasets import lorenz


def load_test_train_split(
    timesteps = 25000,
    train_length = 20000,
    forecast = 1,
    plot = False
    ):
    # create and normalize
    X = lorenz(timesteps)
    X = 2 * (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0)) - 1

    print(X.shape)
    print(X[0])

    X_train, y_train = X[:train_length], X[forecast: train_length + forecast]
    X_test, y_test = X[train_length: -forecast], X[train_length + forecast:]

    print(y_test.shape)
    print(y_test[0])

    if plot:
        sample=1000 # number of samples to plot

        t = np.arange(X.shape[0])
        x, y, z = X[:, 0], X[:, 1], X[:, 2]
        fig = plt.figure(figsize=(13, 5))

        plt.subplot((121))
        plt.title(f"Timeserie - {sample} timesteps")

        plt.plot(t[:sample], x[:sample], color="lightgray", zorder=0)
        plt.scatter(t[:sample], x[:sample], c=t[:sample], cmap="viridis", s=2, zorder=1)

        plt.xlabel("$t$")
        plt.ylabel("$x$")

        cbar = plt.colorbar()
        cbar.ax.set_ylabel('$t$', rotation=270)

        ax = plt.subplot((122), projection="3d")
        ax.margins(0.05)
        plt.title(f"Phase diagram: $z = f(x, y)$")
        plt.plot(x[:sample], y[:sample], z[:sample], lw=1,
                color="lightgrey", zorder=0)
        plt.scatter(x[:sample], y[:sample], zs=z[:sample],
                lw=0.5, c=t[:sample], cmap="viridis", s=2)
        plt.xlabel("$x$")
        plt.ylabel("$y$")
        ax.set_zlabel("$z$")

        plt.tight_layout()
        plt.show()
    
    return (X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    load_test_train_split(plot=True)