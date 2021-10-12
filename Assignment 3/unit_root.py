import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.graphics.tsaplots as tsplots
import statsmodels.tsa.arima.model as arma
import numpy as np
import warnings
import sys
import random

import matplotlib.pyplot as plt
import statsmodels.graphics.tsaplots as tsplots
import statsmodels.tsa.arima.model as arma
import statistics

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
# data is a dataframe, first column is year, second column is GDP
data = pd.read_csv('./data.csv', delimiter=',',
                    names=['date', 'apple', 'exxon', 'ford', 'gen_electric', 'intel', 'microsoft', 'netflix',
                            'nokia', 'sp500', 'yahoo'], skiprows=1)
data['datetime'] = pd.to_datetime(data['date'])


company = data['apple']

def significant_lags(data, alpha, lags, beta, company):
    removed_lags = []
    lags = list(lags)

    ar_model = arma.ARIMA(data[company], order=(lags, 0, 0)).fit()
    summary = ar_model.summary(alpha).tables[1].data[1:]
    p_values = np.array([float(i[4]) for i in summary[1:len(summary) - 1]])

    while len(np.where(p_values < alpha)[0]) != len(p_values):
        remove_lags, removed_lag, done = remove_lag(lags, removed_lags, summary, alpha, beta)
        if done:
            break
        lags.remove(removed_lag)
        ar_model = arma.ARIMA(data[company], order=(lags, 0, 0)).fit()
        summary = ar_model.summary(0.05).tables[1].data[1:]
        p_values = np.array([float(i[4]) for i in summary[1:len(summary) - 1]])

    return sorted(lags)


def remove_lag(lags, removedLags, summary, alpha, beta):
    removed_lag = -1
    biggest = 0
    # control for greedy algorithm with beta:
    if random.random() < beta:
        removed_lag = np.random.choice(lags)
        removedLags.append(removed_lag)
        return removedLags, removed_lag, False
    else:
        # search for the least insignificant lag
        for lag in lags:
            if lag in removedLags:
                continue
            index = np.where(np.array(lags) == lag)[0][0]
            p_value = float(summary[index + 1][4])
            if p_value > biggest and p_value > alpha:
                removed_lag = lag
                biggest = p_value
    if removed_lag != -1:
        removedLags.append(removed_lag)
    else:
        return None, None, True
    return removedLags, removed_lag, False


company_string_list = ['apple', 'exxon', 'ford', 'gen_electric', 'intel', 'microsoft', 'netflix',
                        'nokia', 'sp500', 'yahoo']
for name in company_string_list:
    print(f'------ Company name: {name}')
    alpha = 0.05
    lags = [1,2,3,4,5]
    beta = 0
    company = name
    sig_lags = significant_lags(data, alpha, lags, beta, company)
    print(f'------ significant lags: {sig_lags}')

    p = len(sig_lags)
    ar_model = arma.ARIMA(data[company], order=(sig_lags, 0, 0))
    ar_model_fit = ar_model.fit()
    print(ar_model_fit.summary())

