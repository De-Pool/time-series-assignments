import numpy as np
import pandas as pd


# data is an array, each row (first index) contains an array of length 2
# that array has as first index the year + quarter (string) and as second index the GDP (float)
# data[i][1] -> the GDP of the i-th row
file = './data.csv'
data = pd.read_csv(file, delimiter=',', names=['year', 'gdp'], skiprows=1)
print(data)
