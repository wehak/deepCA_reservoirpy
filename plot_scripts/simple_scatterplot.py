import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_pickle("deepCA/output/20210616144120.pkl")
# print(df)

sns.scatterplot(
    data=df,
    x="sr",
    y="r2"
)

plt.ylim(.8, 1)
plt.xlim(0, 0.15)

plt.show()