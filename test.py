from fitter import Fitter
import pandas as pd
from scipy.stats import foldcauchy
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1)

df = pd.read_csv("output/csvs/mmwave_lat_df.csv")
#f = Fitter(df["Latency"])
#f.fit()
#results = f.summary()
#print(results)

# shape, loc, scale
shape, loc, scale = foldcauchy.fit(df["Latency"])

data = foldcauchy.rvs(shape, loc=loc, scale=scale, size=2800)

new_df = pd.DataFrame(data, columns=["Latency"])
new_df = new_df[new_df["Latency"] > 5.26]
new_df = new_df[new_df["Latency"] < 300.0]
new_df.to_csv("new_df.csv")