import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import statsmodels.graphics.tsaplots as stats

import helper_functions

# data is a dataframe, first column is year, second column is GDP
data = pd.read_csv('./data.csv', delimiter=',', names=['year', 'gdp'], skiprows=1)
data['datetime'] = pd.to_datetime(data['year'])

# Opdracht 1
plt.plot(data['datetime'], data['gdp'])
plt.show()

# acfs is an array with [0, acf(1), acf(2), ..., acf(period)]
acfs = helper_functions.sample_acf(data, period=12)
plt.bar(np.arange(1, len(acfs)), acfs[1:])
plt.show()

# pacfs is an array with [pacf(0), pacf(1), pacf(2), ..., pacf(period)]
pacfs = helper_functions.sample_pacf(data, period=12)
plt.bar(np.arange(0, len(pacfs)), pacfs)
plt.show()
