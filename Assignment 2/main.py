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
plt.xlabel('Period (Quarters)')
plt.ylabel('Unemployment Rate (Percentage)')
plt.legend()
plt.show()

# ADL Model
# If we use General-Specific starting at lag 4, we get the following model
# ADL(2, 1) p = [1, 3], q = [1]
p, q = helper_functions_2.significant_adl_arr(data, 0.05, np.arange(1, 5),
                                              np.arange(0, 5), 0, no_p=False, no_q=False)
adl_model = helper_functions_2.adl_ols_arr(p, q, data)

adl_model_fit = adl_model.fit()
print(adl_model_fit.summary())
plt.plot(data['datetime'].values[len(data['datetime'].values) - len(adl_model_fit.fittedvalues.values):],
         adl_model_fit.fittedvalues.values, label="Fitted ADL(" + str(p) + "," + str(q) + ")  model")
plt.plot(data['datetime'], data['UN_RATE'], label="Data")
plt.xlabel('Period (Quarters)')
plt.ylabel('Unemployment Rate (Percentage)')
plt.legend()
plt.show()
