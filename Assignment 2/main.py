import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.graphics.tsaplots as tsplots
import statsmodels.tsa.arima.model as arma
import numpy as np
import helper_functions_2
import warnings
import sys

# TimeSeries data generation:
import statsmodels.api as sm

# Test statistic calculation:
import statsmodels.stats as sm_stat

# Model estimation:
import statsmodels.tsa as smt

# Optimization:
import scipy.optimize as optimize

# We also need additional data:
import statsmodels.formula.api as smf

if not sys.warnoptions:
    warnings.simplefilter("ignore")

# data is a dataframe, first column is year, second column is GDP
data = pd.read_csv('./data_assign_p2.csv', delimiter=',', names=['obs', 'GDP_QGR', 'UN_RATE'], skiprows=1)
data['datetime'] = pd.to_datetime(data['obs'])

# helper_functions_2.best_model(data, 0)
lags = [1, 3]
p = len(lags)
ar_model = arma.ARIMA(data['GDP_QGR'], order=(lags, 0, 0))
ar_model_fit = ar_model.fit()
# print(ar_model_fit.summary())

# fittedValues = ar_model_fit.fittedvalues.values
# plt.plot(data['datetime'], fittedValues, label="Fitted AR(" + str(p) + ")  model")
# plt.plot(data['datetime'], data['GDP_QGR'], label="Data")
# plt.xlabel('years')
# plt.ylabel('growth rate percentages')
# plt.legend()
# plt.show()

# helper_functions_2.adl_ols(p, q, data)
# adl_model = helper_functions_2.adl_ols(2, 2, data)
# adl_model_fit = adl_model.fit()
# print(adl_model_fit.summary())
#
# a = adl_model_fit.summary().tables[1].data[2:]
# print(adl_model_fit.summary().tables[1].data[2:])

# p, q = helper_functions_2.significant_adl(data, 0.1, [1, 2, 3, 4],
#                                           [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], 0, no_p=False, no_q=False)
# p, q = helper_functions_2.significant_adl(data, 0.1, [1, 2, 3, 4],
#                                           [1, 2, 3, 4], 0, no_p=False, no_q=False)
# for i in range(2, 7):
#     adl_model = helper_functions_2.adl_ols_arr([1], np.arange(1, i), data)
#     adl_model_fit = adl_model.fit()
#     print(adl_model_fit.summary())

p, q = [1], [1]
adl_model = helper_functions_2.adl_ols_arr(p, q, data)
adl_model_fit = adl_model.fit()
print(adl_model_fit.summary())
#