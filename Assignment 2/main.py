import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.graphics.tsaplots as tsplots
import statsmodels.tsa.arima.model as arma
import helper_functions
import warnings
import sys

if not sys.warnoptions:
    warnings.simplefilter("ignore")

# data is a dataframe, first column is year, second column is GDP
data = pd.read_csv('./data_assign_p2.csv', delimiter=',', names=['obs', 'GDP_QGR', 'UN_RATE'], skiprows=1)
data['datetime'] = pd.to_datetime(data['obs'])

# helper_functions.best_model(data, 0)
lags = [1, 3]
p = len(lags)
ar_model = arma.ARIMA(data['GDP_QGR'], order=(lags, 0, 0))
ar_model_fit = ar_model.fit()
print(ar_model_fit.summary())

fittedValues = ar_model_fit.fittedvalues.values
plt.plot(data['datetime'], fittedValues, label="Fitted AR(" + str(p) + ")  model")
plt.plot(data['datetime'], data['GDP_QGR'], label="Data")
plt.xlabel('years')
plt.ylabel('growth rate percentages')
plt.legend()
plt.show()
