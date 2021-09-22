import numpy as np


# calculates covariance of series X with specified lag.
# input is dataframe of data.csv and 0 < lag < len(x)
# returns covariance matrix of Xt and Xt-lag (offdiagonal entries are the autocovariances)
def autocov(X, lag):
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
def pautocorr(data, lag):
    if lag < 0 or lag > len(data):
        raise Exception("invalid lag")
    elif lag == 0:
        return 1
    elif lag == 1:
        # pacf(1) = autocov(1) / variance
        return autocov(data, 1)[0][1] / autocov(data, 0)[0][0]
    # X is a matrix of m x n where m = len(X) - lag, n = lag + 1
    X = np.zeros((len(data) - lag, lag + 1))
    Xt = data.iloc[:len(data) - lag, 1].values
    for i in range(lag + 1):
        if i == 0:
            X[:, i] = np.ones(len(data) - lag)
        else:
            X[:, i] = data.iloc[i:len(data) - lag + i, 1].values

    # np.linalg.lstsq returns solution, residuals, rank, singular values
    B_hat, _, _, _ = np.linalg.lstsq(X, Xt, rcond=None)
    return B_hat[lag]


# input is dataframe of data.csv and 1 < period <= len(X)
# calculates acf from 1, 2, 3, ..., period
# returns an array with [0, acf(1), acf(2), ..., acf(period)]
def sample_acf(X, period):
    if period < 0 or period > len(X):
        raise Exception("invalid period")

    variance = autocov(X, 0)[0][0]
    acfs = np.zeros(period + 1)
    for i in range(1, period + 1):
        autocovariance = autocov(X, i)
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
        pacfs[i] = pautocorr(X, i)

    return pacfs
