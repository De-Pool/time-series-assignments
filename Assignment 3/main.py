import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.graphics.tsaplots as tsplots
import statsmodels.tsa.arima.model as arma
import numpy as np
import helper_functions_3
import warnings
import sys
import helper_functions


def main():
    # data is a dataframe, first column is year, second column is GDP
    data = pd.read_csv('./data.csv', delimiter=',',
                       names=['date', 'apple', 'exxon', 'ford', 'gen_electric', 'intel', 'microsoft', 'netflix',
                              'nokia', 'sp500', 'yahoo'], skiprows=1)
    data['datetime'] = pd.to_datetime(data['date'])

    opdracht2(data)
    opdracht4(data)


# plot acf and pacf
def opdracht2(data):
    # acf and pacf apple
    tsplots.plot_acf(data['apple'].values, lags=12, zero=False, title="Autocorrelation Apple")
    tsplots.plot_pacf(data['apple'].values, lags=12, zero=False, title="Partial Autocorrelation Apple")
    plt.show()

    # acf and pacf netflix
    tsplots.plot_acf(data['netflix'].values, lags=12, zero=False, title="Autocorrelation Netflix")
    tsplots.plot_pacf(data['netflix'].values, lags=12, zero=False, title="Partial Autocorrelation Netflix")
    plt.show()


# assume that the error term ~ N(1, 0)
def opdracht4(data):
    apple = data['apple'].values
    microsoft = data['microsoft'].values
    forecast_apple = helper_functions_3.random_walk_forecast(apple[-1], 0, 1, 5)
    conf_lower_apple, conf_upper_apple = helper_functions_3.significance(forecast_apple, 1.96)
    print("Forecasted values of stock Apple: ", forecast_apple)
    print("Lower bound Apple: ", conf_lower_apple)
    print("Upper bound Apple: ", conf_upper_apple)

    forecast_ms = helper_functions_3.random_walk_forecast(microsoft[-1], 0, 1, 5)
    conf_lower_ms, conf_upper_ms = helper_functions_3.significance(forecast_ms, 1.96)
    print("Forecasted values of stock Microsoft: ", forecast_ms)
    print("Lower bound Microsoft: ", conf_lower_ms)
    print("Upper bound Microsoft: ", conf_upper_ms)



if __name__ == '__main__':
    main()
