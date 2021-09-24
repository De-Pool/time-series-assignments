import numpy as np
import statsmodels.tsa.ar_model as ar
import statsmodels.tsa.arima.model as arma
import random
import sklearn.metrics as sk
import math


def best_model_gs(data):
    beta = [0.2, 0.5]
    smallest_rmse = 1
    best_params = []
    print(rmse_lags(data, [1]))

    for b in beta:
        for i in range(1000):
            lags = significant_lags(data, 0.05, np.arange(1, random.randint(10, 39)), b)
            rmse = rmse_lags(data, lags)
            if rmse < smallest_rmse:
                best_params = [lags, rmse]
                smallest_rmse = rmse
                print(best_params)

    return best_params


def rmse_lags(data, lags):
    ar_model = arma.ARIMA(data['gdp'], order=(lags, 0, 0)).fit()
    forecast = ar_model.get_forecast(8)
    result = forecast.prediction_results.results.forecasts[0]
    actualData = [-1.63, 0.28, 0.33, 0.66, 1.59, 0.51, 0.71, 0.81]
    mse = sk.mean_squared_error(actualData, result)
    return math.sqrt(mse)


# Greedy algorithm to find the best combination of lags.
# Returns a sorted list with all
def significant_lags(data, alpha, lags, beta):
    removed_lags = []
    lags = list(lags)
    ar_model = ar.AutoReg(data['gdp'], lags=lags, old_names=False).fit()
    summary = ar_model.summary(alpha).tables[1].data[1:]
    p_values = np.array([float(i[4]) for i in summary[1:]])

    while len(np.where(p_values < alpha)) != len(lags) or len(p_values) == 1:
        ar_model = ar.AutoReg(data['gdp'], lags=lags, old_names=False).fit()
        summary = ar_model.summary(0.05).tables[1].data[1:]
        remove_lags, removed_lag, done = remove_lag(lags, removed_lags, summary, alpha, beta)
        if done:
            break
        lags.remove(removed_lag)
        p_values = np.array([float(i[4]) for i in summary[1:]])
        # print(lags)
    # ar_model = ar.AutoReg(data['gdp'], lags=lags, old_names=False).fit()
    # print(ar_model.summary())
    # print(lags)
    # print(len(lags))
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


# calculates covariance of series X with specified lag.
# input is dataframe of data.csv and 0 < lag < len(x)
# returns covariance matrix of Xt and Xt-lag (offdiagonal entries are the autocovariances)
def auto_cov(X, lag):
    if lag < 0 or lag > len(X):
        raise Exception("invalid lag")

    Xt = X.iloc[: len(X) - lag, 1].values
    Xt_lag = X.iloc[lag:, 1].values

    # np.cov returns a covariance matrix
    sigma = np.cov([Xt, Xt_lag])
    return sigma


# partial autocovariance = covariance(Xt, Xt-lag | Xt-lag+1, Xt-lag+2, ..., Xt-1)
# Say we want to know the partial auto covariance of Xt-3
# we can write Xt as Xt = B0 + B1*Xt-1 + B2*Xt-2 + B3*Xt-3
# to solve this we need to use OLS to solve the following matrix equation where:
# Y = XB
# X = [e, Xt-1, Xt-2, Xt-3], B = [B0, B1, B2, B3], Y = Xt
# B_hat = (X^T X)^-1 X^T Y  where B^ contains the estimators for B0, B1, B2, B3
# where B3 is an estimator for the partial autocorrelation
def p_auto_corr(data, lag):
    if lag < 0 or lag > len(data):
        raise Exception("invalid lag")
    elif lag == 0:
        return 1
    elif lag == 1:
        # pacf(1) = autocov(1) / variance
        return auto_cov(data, 1)[0][1] / auto_cov(data, 0)[0][0]

    # X is a matrix of m x n where m = len(X) - lag, n = lag + 1
    X = np.zeros((len(data) - lag, lag + 1))
    X[:, 0] = np.ones(len(data) - lag)
    for i in range(1, lag + 1):
        X[:, i] = data.iloc[i:len(data) - lag + i, 1].values

    Xt = data.iloc[:len(data) - lag, 1].values
    # np.linalg.lstsq returns solution, residuals, rank, singular values
    B_hat, _, _, _ = np.linalg.lstsq(X, Xt, rcond=None)
    return B_hat[lag]


# input is dataframe of data.csv and 1 < period <= len(X)
# calculates acf from 1, 2, 3, ..., period
# returns an array with [0, acf(1), acf(2), ..., acf(period)]
def sample_acf(X, period):
    if period < 0 or period > len(X):
        raise Exception("invalid period")

    variance = auto_cov(X, 0)[0][0]
    acfs = np.zeros(period + 1)
    for i in range(1, period + 1):
        autocovariance = auto_cov(X, i)
        # acf = autocov(h) / autocov(0) where h=lag and autocov(0) = variance(X)
        acfs[i] = autocovariance[0][1] / variance
    return acfs


# input is dataframe of data.csv and 1 < period <= len(X)
# calculates acf from 1, 2, 3, ..., period
# returns an array with [0, acf(1), acf(2), ..., acf(period)]
def sample_pacf(X, period):
    if period < 0 or period > len(X):
        raise Exception("invalid period")

    pacfs = np.zeros(period + 1)
    for i in range(0, period + 1):
        pacfs[i] = p_auto_corr(X, i)

    return pacfs

# acfs is an array with [0, acf(1), acf(2), ..., acf(period)]
# acfs = helper_functions.sample_acf(data, period=50)
# plt.bar(np.arange(1, len(acfs)), acfs[1:])
# plt.show()

# pacfs is an array with [pacf(0), pacf(1), pacf(2), ..., pacf(period)]
# pacfs = helper_functions.sample_pacf(data, period=20)
# plt.bar(np.arange(0, len(pacfs)), pacfs)
# plt.show()
