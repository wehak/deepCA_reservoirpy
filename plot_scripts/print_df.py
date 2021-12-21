from pathlib import Path

import pandas as pd

# custom functions
from utils import get_data
from utils import clean_data

files = Path("/home/wehak/data/deepCA/output/data/celegans_random_hyperparam_sr09")

# load and clean data
data = get_data(files)
data = clean_data(data) # remove entries with crazy values (nrmse > 1 and r2 < 0)

print(data)