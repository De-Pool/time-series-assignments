import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.graphics.tsaplots as tsplots
import statsmodels.tsa.arima.model as arma
import helper_functions
import statistics

# data is a dataframe, first column is year, second column is GDP
data = pd.read_csv('./data.csv', delimiter=',', names=['year', 'gdp'], skiprows=1)
data['datetime'] = pd.to_datetime(data['year'])

# Opdracht 1
plt.title("GDP over time")
plt.plot(data['datetime'], data['gdp'])
tsplots.plot_acf(data['gdp'], lags=12, zero=False)
tsplots.plot_pacf(data['gdp'], lags=12, zero=False)
plt.show()

# Opdracht 2
# Set beta to 0 for a deterministic result. -> (lags = [1])
# lags = helper_functions.significant_lags(data, alpha=0.05, lags=np.arange(1, 25), beta=0)
# print("Best params are: ", helper_functions.best_model(data, 0))
lags = [1]
p = len(lags)
ar_model = arma.ARIMA(data['gdp'], order=(lags, 0, 0))
ar_model_fit = ar_model.fit()
print(ar_model_fit.summary())

fittedValues = ar_model_fit.fittedvalues.values
plt.plot(data['datetime'], fittedValues, label="Fitted AR(" + str(p) + ")  model")
plt.plot(data['datetime'], data['gdp'], label="Data")
plt.xlabel('years')
plt.ylabel('growth rate percentages')
plt.legend()
plt.show()

# Opdracht 3
residuals = pd.DataFrame(ar_model_fit.resid)
print(f'residuals: {residuals.mean()}')
residuals.plot(title="Residuals")
plt.xlabel('periods (in quarters)')
plt.ylabel('growth rate percentages')
plt.show()
tsplots.plot_acf(residuals, zero=False, lags=50)
plt.xlabel('lag amount')
plt.show()

# Opdracht 4
forecastDate = pd.to_datetime(['2009Q2', '2009Q3', '2009Q4', '2010Q1', '2010Q2', '2010Q3', '2010Q4', '2011Q1'])
forecast = ar_model_fit.get_forecast(8)
result = forecast.prediction_results.results.forecasts[0]
forecast_series = pd.Series(result, forecastDate)
plt.title("2 year period forecasted with an AR(" + str(p) + ") model")
plt.plot(data['datetime'][50:], data['gdp'][50:], label="Data")
plt.plot(forecast_series, label="Forecasted")
plt.legend()
plt.show()

# Opdracht 5
ar_model = arma.ARIMA(data['gdp'], order=(lags, 0, 0)).fit()
forecast = ar_model.get_forecast(8)
confidence_interval = forecast.conf_int(0.05)
result = forecast.prediction_results.results.forecasts[0]
forecast_series = pd.Series(result, index=forecastDate)
lower_conf = pd.Series(confidence_interval['lower gdp'].values, index=forecastDate)
higher_conf = pd.Series(confidence_interval['upper gdp'].values, index=forecastDate)

# plt.figure(figsize=(12, 5), dpi=100)
# plt.title("2 year period forecasted with an AR(" + str(p) + ") model with a 95% confidence interval")
# plt.plot(data['datetime'][50:], data['gdp'][50:], label="Data")
# plt.plot(forecast_series, label='Forecast')
# plt.fill_between(lower_conf.index, lower_conf, higher_conf,
#                  color='k', alpha=.15)
# plt.legend(loc='upper left', fontsize=13)
# plt.show()

# Opdracht 7
actualData = [-1.63, 0.28, 0.33, 0.66, 1.59, 0.51, 0.71, 0.81]
plt.figure(figsize=(12, 5), dpi=100)
plt.title("2 year period forecasted with an AR(" + str(p) + ") model compared to the actual values")
plt.plot(forecastDate, actualData, label="Actual")
plt.plot(data['datetime'][50:], data['gdp'][50:], label="Data")
plt.plot(forecast_series, label='Forecast')
plt.fill_between(lower_conf.index, lower_conf, higher_conf,
                 color='k', alpha=.15)
plt.legend(loc='upper left', fontsize=13)
plt.show()

#  print("Best Params: ", helper_functions.best_model(data, 0.5))
