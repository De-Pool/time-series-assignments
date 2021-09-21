import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import helper_functions

# data is a dataframe, first column is year, second column is GDP
data = pd.read_csv('./data.csv', delimiter=',', names=['year', 'gdp'], skiprows=1)

# Opdracht 1
# acfs is an array with [0, acf(1), acf(2), ..., acf(period)]
acfs = helper_functions.sample_acf(data, period=12)
plt.bar(np.arange(1, len(acfs)), acfs[1:])
plt.show()
