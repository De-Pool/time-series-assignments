import numpy as np

# data is an array, each row (first index) contains an array of length 2
# that array has as first index the year + quarter (string) and as second index the GDP (float)
# data[i][1] -> the GDP of the i-th row
data = np.loadtxt('data.csv', delimiter=',', skiprows=1, dtype=np.dtype(np.str_, np.float_))
print(data)

