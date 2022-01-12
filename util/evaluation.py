from pathlib import Path
import sys
import datetime
import re
import warnings

from scipy import sparse
import numpy as np
import pandas as pd
from sklearn.metrics import explained_variance_score

from reservoirpy import ESN, mat_gen

# import custom function
toolbox_dir = Path(__file__).parents[1].absolute()
sys.path.append(str(toolbox_dir))
from util.datasets import mackeyglass
from util.datasets import ett
from util.datasets import santa_fe_laser
from util.datasets import lorenz

def sparsity(sparse_matrix):
    return 1.0 - (sparse_matrix.count_nonzero() / np.prod(sparse_matrix.shape))

def r2_score(y_true, y_pred):
    return 1 - (np.sum((y_true - y_pred)**2) / np.sum((y_true - y_true.mean())**2))

def nrmse(y_true, y_pred):
    return np.sqrt((np.sum(y_true - y_pred)**2) / len(y_true)) / (y_true.max() - y_true.min())

# calculate the spectral radius of a reservoir
def sr(W_matrix):
    try:
        eigenvalues = sparse.linalg.eigs(
            W_matrix,
            which="LM", # largest magnitude
            # which="LR", # largest real part
            return_eigenvectors=False)
        # return np.amax(eigenvalues.real) # real part
        return round(np.amax(abs(eigenvalues)), 0) # magnitude 
    except: # if no convergence
        return np.nan


def initiate_train_and_test_ESN(
    adj_matrix,
    dataset,
    forecast,
    Win=None,
    leak_rate = 0.3,
    input_scaling = 0.9,
    input_connectivity = 0.2,
    regularization = 1e-6,
    seed = False,
    input_bias = False,
    ):


    """ load data """
    if dataset == "mg":
        X_train, X_test, y_train, y_test = mackeyglass.load_test_train_split(
            forecast=forecast,
            tau=17,
            )

    elif dataset == "santafe":
        X_train, X_test, y_train, y_test = santa_fe_laser.load_test_train_split(
            timesteps=1100,
            train_length=700,
            forecast=forecast,
        )

    elif dataset == "ett":
        X_train, X_test, y_train, y_test = ett.load_test_train_split(
            timesteps=60000,
            train_length=50000,
            forecast=forecast,
        )

    elif dataset == "lorenz":
        X_train, X_test, y_train, y_test = lorenz.load_test_train_split(
            timesteps=25000,
            train_length=20000,
            forecast=forecast,
        )
    
    else:
        print(f"Not a valid dataset: '{dataset}'")
        exit()
    
    """ initiate the ESN """
    # read the custom matrix from a csv file and convert to sparse matrix (class 'scipy.sparse.csr.csr_matrix')
    W = sparse.csr_matrix(
        np.loadtxt(
            open(adj_matrix, "rb"),
            delimiter=","
        )
    )
    units = W.shape[0] # size of the reservoir

    # if no custom input matrix is given, generate a random input matrix
    if Win is None:
        Win = mat_gen.generate_input_weights(
            units, # reservoir size
            X_train.shape[1], # dimensionality of the training data 
            input_scaling=input_scaling,
            proba=input_connectivity,
            input_bias=input_bias,
            seed=seed
            )
    
    reservoir = ESN(
        lr=leak_rate,
        W=W,
        Win=Win,
        ridge=regularization,
        input_bias=input_bias,
    )

    """ train ESN """
    # reservoir.train(
    states = reservoir.train(
        X_train,
        y_train,
        # y_train.reshape(-1, 1),
        return_states=True,
        # verbose=True
        )

    y_pred = reservoir.run(
    # y_pred, state1 = reservoir.run( # if return_states=True
        X_test,
        init_state=states[0][-1],
        # return_states=True,
        # verbose=True
        )
    
    y_pred = y_pred[0][0]

    # get data encoded in filename
    r = re.search("SR_([0-9.]*)([a-zA-Z_]*)(\d*)", Path(adj_matrix).stem)

    # write metrics to dictionary
    result = {
        "filename": Path(adj_matrix).stem,
        "filename:sr": r.group(1),
        "filename:topology_type": r.group(2),
        "filename:id": r.group(3),
        "size" : units,
        "sr": sr(W),
        "sparsity": sparsity(W),
        "r2": r2_score(y_test, y_pred),
        "nrmse": nrmse(y_test, y_pred),
        "ev": explained_variance_score(y_test, y_pred),
        "leak_rate": leak_rate,
        "input_scaling": input_scaling,
        "input_connectivity": input_connectivity,
        "regularization": regularization,
        "seed": seed,
        "dataset": dataset,
        "forecast": forecast
    }

    return result



# example run on single matrix
if __name__ == "__main__":
    matrix = Path("input\celegans131matrix\SR_0.9\Random\SR_0.9Random1.csv")
    results = []
    results.append(initiate_train_and_test_ESN(
        matrix,
        dataset="mg",
        forecast=200,
        regularization=1e-6
        ))
    
    # convert to pandas df for presentation
    df = pd.DataFrame(results)
    df.sort_values(by=["r2"], inplace=True, ascending=False)
    print(df.to_string(index=False))
    print("Mean", round(df["r2"].mean(), 3))
    
    # write to file?
    filename = Path(f"/output/data/{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.pkl")
    # filename.mkdir(parents=True, exist_ok=True)
    # df.to_pickle(filename)
