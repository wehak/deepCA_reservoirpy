# deepCA_reservoirpy
Intended to run on ReservoirPy 0.2.x (tested using 0.2.4) with Python 3.8

Python dependencies:
- reservoirpy
- pyreadr
- sklearn
- seaborn

## Use on Windows
Clone this repo to a local folder and open it in the windows command line
### Batch test accuracy of adjacency matrices
Open compare_accuracy_all_matrices.py in a text editor and review the parameters: 
- Path to folder with adjacency matrices
- Datasets to be used for testing
- Forecast levels
- Limit the number of adjacency matrices of each type to test

To run the script from command line a few more parameters are required:
- Low (-l) where in the list of matrices to start testing
- High (-h) where in the list of matrices to end testing
- Output folder (-f) where to save the output data

This is example is run from the command line and will test 1000 matrices from #0 up until #999
```
python compare_accuracy_all_matrices.py -l 0 -h 1000 -f test_run_accuracy
```

Run the script from multiple command line windows for parallell processing. ReservoirPy will leave temporary files in C:\Users\<your_user>\AppData\Local\Temp\reservoirpy-temp that need to be cleaned manually. Roughy 9 GB for 3500 131x131 matrices testet over 4 forecast levels on Mackey-Glass.

### Batch test random hyperparameters
Open hyper_param_random_search.py in a text editor and review the parameters: 
- Path to folder with adjacency matrices
- Datasets to be used for testing
- Forecast levels
- Hyperparameter ranges
- Number of samples

To run the script from command line a few more parameters are required:
- Test matrix (-i) name of the adjacency matrix to test
- Output folder (-f) where to save the output data

This is example is run from the command line and will test "SR_0.9Random1.csv"
```
python hyper_param_random_search.py -i SR_0.9Random1 -f test_run_hyperparameters
```

## Use on Linux
Similar as on Windows, but use the scripts in the shell_scripts folder. If running a batch that is too large on a single python script, Linux will shut it down. This is because of the excessive amount of temp files reservoirpy will generate without cleaning up. 