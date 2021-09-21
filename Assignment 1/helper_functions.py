import numpy as np


# calculates covariance of series X with specified lag.
# input is dataframe of data.csv and 0 < lag < len(x)
# returns covariance matrix of Xt and Xt-lag (offdiagonal entries are the acf)
def autocov(X, lag):
    if lag < 0 or lag > len(X):
        raise ("invalid lag")

    Xt = X.iloc[: len(X) - lag, 1].values
    Xt_lag = X.iloc[lag:, 1].values

    # np.cov returns a covariance matrix
    sigma = np.cov([Xt, Xt_lag])
    return sigma


# input is dataframe of data.csv and 1 < period <= len(X)
# calculates acf from 1, 2, 3, ..., period
# returns an array with [0, acf(1), acf(2), ..., acf(period)]
def sample_acf(X, period):
    if period < 0 or period > len(X):
        raise ("invalid period")

    variance = autocov(X, 0)[0][0]
    acfs = np.zeros(period + 1)
    for i in range(1, period + 1):
        autocovariance = autocov(X, i)
        # acf = autocov(h) / autocov(0) where h=lag and autocov(0) = variance(X)
        acfs[i] = autocovariance[0][1] / variance
    return acfs
