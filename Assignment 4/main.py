import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.graphics.tsaplots as tsplots
import statsmodels.tsa.arima.model as arma
import numpy as np
import helper_functions_4
import statsmodels.formula.api as smf
import statsmodels.tsa.stattools as stats


def main():
    # data is a dataframe, first column is year, second column is GDP
    data = pd.read_csv('./data.csv', delimiter=',',
                       names=['obs', 'cons', 'inc', 'time'], skiprows=1)
    data['datetime'] = pd.to_datetime(data['obs'])

    print()
    # opdracht2(data)
    # opdracht4(data)
    # opdracht5(data)


# plot acf and pacf
def opdracht2(data):
    pass
    # acf and pacf apple
    # tsplots.plot_acf(data['apple'].values, lags=12, zero=False, title="Autocorrelation Apple")
    # tsplots.plot_pacf(data['apple'].values, lags=12, zero=False, title="Partial Autocorrelation Apple")
    # plt.show()
    #
    # acf and pacf netflix
    # tsplots.plot_acf(data['netflix'].values, lags=12, zero=False, title="Autocorrelation Netflix")
    # tsplots.plot_pacf(data['netflix'].values, lags=12, zero=False, title="Partial Autocorrelation Netflix")
    # plt.show()


if __name__ == '__main__':
    main()
