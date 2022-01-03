
from pathlib import Path
import matplotlib.pyplot as plt

# custom functions
from plot_scripts.utils import get_data
from plot_scripts.utils import clean_data

""" parameters """
folder_path = Path("output/data/acc_testing/")

""" program """
# make a unique save folder
savepath = Path(f"output/excel/")
savepath.mkdir(parents=True, exist_ok=True)

# load and clean data
data = get_data(folder_path)
data.sort_values(by=["r2"], inplace=True, ascending=False)
# data.to_csv(f"{savepath}/{folder_path.stem}.csv")
data.to_excel(f"{savepath}/{folder_path.stem}.xlsx")