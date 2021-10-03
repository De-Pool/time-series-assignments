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


# Opdracht 1
# AR model:
lags = [1, 3]
p = len(lags)
ar_model = arma.ARIMA(data['GDP_QGR'], order=(lags, 0, 0))
ar_model_fit = ar_model.fit()

fittedValues = ar_model_fit.fittedvalues.values
plt.plot(data['datetime'], fittedValues, label="Fitted AR(" + str(p) + ")  model")
plt.plot(data['datetime'], data['GDP_QGR'], label="Data")
plt.xlabel('years')
plt.ylabel('growth rate percentages')
plt.legend()
plt.show()

# p, q = helper_functions_2.significant_adl_arr(data, 0.1, np.arange(1, 20),
#                                           np.arange(0, 20), 0, no_p=False, no_q=False)
# sig = helper_functions_2.significant_adl(data, 0.05, 10, 10, False, False)
# print(sig)
# The only model with significant lags is ADL(1, 0)
# ADL Model
p, q = 1, 0
adl_model, formula = helper_functions_2.adl_ols(p, q, data)
adl_model_fit = adl_model.fit()
plt.plot(data['datetime'].values[len(data['datetime'].values) - len(adl_model_fit.fittedvalues.values):], adl_model_fit.fittedvalues.values, label="Fitted ADL(" + str(p) + "," + str(q) + ")  model")
plt.plot(data['datetime'], data['UN_RATE'], label="Data")
plt.xlabel('years')
plt.ylabel('growth rate percentages')
plt.legend()
plt.show()
