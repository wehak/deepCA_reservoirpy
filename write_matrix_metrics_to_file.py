
from pathlib import Path
import matplotlib.pyplot as plt

# custom functions
from plot.utils import get_data
from plot.utils import clean_data

""" parameters """
filepath = Path("deepCA/output/data/celegans131_1")

""" program """
# make a unique save folder
savepath = Path(f"deepCA/output/figures")
savepath.mkdir(parents=True, exist_ok=True)

# load and clean data
data = get_data(filepath)
data.sort_values(by=["r2"], inplace=True, ascending=False)
# data.to_csv(f"{savepath}/{filepath.stem}.csv")
data.to_excel(f"{savepath}/{filepath.stem}.xlsx")