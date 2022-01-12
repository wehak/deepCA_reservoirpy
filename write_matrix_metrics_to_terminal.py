from pathlib import Path

import pandas as pd

# custom functions
from plot_scripts.utils import get_data
from plot_scripts.utils import clean_data

files = Path(r"output\data\test_1")

# load and clean data
data = get_data(files)
# data = clean_data(data) # remove entries with crazy values (nrmse > 1 and r2 < 0)

print(data)