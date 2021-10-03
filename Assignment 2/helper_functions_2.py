import numpy as np
import statsmodels.tsa.arima.model as arma
import random
import sklearn.metrics as sk
import math
import statsmodels.formula.api as smf
import pandas as pd

# Assignment 2

# This function is used to create an ADL model object.
# Usage:
# p, q = 1, 0
# adl_model = helper_functions_2.adl_ols(p, q, data)
# adl_model_fit = adl_model.fit()
# The documentation on adl_model_fit can be found on:
# https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.RegressionResults.html
def adl_ols(p, q, data):
    model_data = pd.DataFrame()
    formula = 'Yt ~ 1 + '
    # the maximum length of each vector is determined by the highest lag
    length = len(data) - max(p, q)
    Yt = data['UN_RATE'].values[:length]
    model_data['Yt'] = Yt
    # we sum here from (1, p), since Yt is dependent on Yt-1, Yt-2 ... Yt-p
    # add Yt to formula
    for lag in range(1, p + 1):
        phi_string = 'phi' + str(lag)
        formula += phi_string
        if lag != p + 1:
            formula += ' + '
        phi_data = ols_phi_data(data, lag, length)
        model_data[phi_string] = phi_data
    # we sum here from (0, q) since Yt is dependent on Xt, Xt-1, ..., Xt-q
    # add Xt to formula
    for lag in range(q + 1):
        beta_string = 'beta' + str(lag)
        formula += beta_string
        if lag != q:
            formula += ' + '
        beta_data = ols_beta_data(data, lag, length)
        model_data[beta_string] = beta_data

    model = smf.ols(formula=formula, data=model_data)
    return model, formula


# p and q are ints
def significant_adl(data, alpha, p, q, no_p, no_q):
    q_arr = np.arange(0, q + 1)
    p_arr = np.arange(1, p + 1)
    significant_models = []

    if no_p:
        p_arr = [p]
    elif no_q:
        q_arr = [q]

    for p_int in p_arr:
        for q_int in q_arr:
            adl_model = adl_ols(p_int, q_int, data).fit()
            summary = adl_model.summary(alpha).tables[1].data[2:]
            p_values = np.array([float(i[4]) for i in summary[:len(summary)]])
            if len(np.where(p_values < alpha)[0]) == len(p_values):
                significant_models.append([p_int, q_int])
    return significant_models


# p is an array of p, q is an array of q, beta is used to control for greedy algorithm.
def significant_adl_arr(data, alpha, p, q, beta, no_p, no_q):
    removed_p_arr = []
    removed_q_arr = []
    p = list(p)
    q = list(q)

    adl_model = adl_ols_arr(p, q, data).fit()
    summary = adl_model.summary(alpha).tables[1].data[2:]
    p_values = np.array([float(i[4]) for i in summary[:len(summary)]])
    while len(np.where(p_values < alpha)[0]) != len(p_values):
        removed_p_arr, removed_q_arr, removed_p, removed_q, done = remove_p_q(p, q, removed_p_arr, removed_q_arr,
                                                                              summary, alpha, beta, no_p, no_q)
        if done:
            break
        if removed_q == -1:
            p.remove(removed_p)
        else:
            q.remove(removed_q)
        adl_model = adl_ols_arr(p, q, data).fit()
        summary = adl_model.summary(alpha).tables[1].data[2:]
        p_values = np.array([float(i[4]) for i in summary[:len(summary)]])

    return sorted(p), sorted(q)


# returns removed_p_arr, removed_q_arr, removed_p, removed_q, done
def remove_p_q(p, q, removed_p_arr, removed_q_arr, summary, alpha, beta, no_p, no_q):
    removed_p = -1
    removed_q = -1
    biggest_p = True
    biggest = 0
    # control for greedy algorithm with beta:
    if random.random() < beta:
        return None, None, None, None, True
        # removed_lag = np.random.choice(lags)
        # removedLags.append(removed_lag)
        # return removedLags, removed_lag, False
    else:
        # search for the least insignificant lag
        if not no_p:
            for p_int in p:
                if p_int in removed_p_arr:
                    continue
                index = np.where(np.array(p) == p_int)[0][0]
                p_value = float(summary[index][4])
                if p_value > biggest and p_value > alpha:
                    removed_p = p_int
                    biggest = p_value
        if not no_q:
            for q_int in q:
                if q_int in removed_q_arr:
                    continue
                index = np.where(np.array(q) == q_int)[0][0] + len(p)
                p_value = float(summary[index][4])
                if p_value > biggest and p_value > alpha:
                    biggest_p = False
                    removed_q = q_int
                    biggest = p_value

    if biggest_p:
        if removed_p != -1:
            removed_p_arr.append(removed_p)
            return removed_p_arr, removed_q_arr, removed_p, -1, False
        else:
            return None, None, None, None, True
    else:
        removed_q_arr.append(removed_q)
        return removed_p_arr, removed_q_arr, -1, removed_q, False


# p and q are both arrays.
def adl_ols_arr(p, q, data):
    model_data = pd.DataFrame()
    formula = 'Yt ~ 1 + '
    length = 0
    if len(p) == 0 and len(q) == 0:
        return None
    elif len(p) == 0:
        length = len(data) - max(q)
    elif len(q) == 0:
        length = len(data) - max(p)
    else:
        length = len(data) - max(max(p), max(q))
    Yt = data['UN_RATE'].values[:length]
    model_data['Yt'] = Yt
    counter = 0
    # add Yt to formula
    for lag in p:
        phi_string = 'phi' + str(lag)
        formula += phi_string
        if counter == len(p) - 1 and len(q) == 0:
            formula = formula
        else:
            formula += ' + '
        phi_data = ols_phi_data(data, lag, length)
        model_data[phi_string] = phi_data
        counter += 1
    # add Xt to formula
    counter = 0
    for lag in q:
        beta_string = 'beta' + str(lag)
        formula += beta_string
        if counter != len(q) - 1:
            formula += ' + '
        beta_data = ols_beta_data(data, lag, length)
        model_data[beta_string] = beta_data
        counter += 1

    model = smf.ols(formula=formula, data=model_data)
    return model


def ols_phi_data(data, lag, length):
    unemployment = data['UN_RATE'].values
    end_index = length + lag
    return unemployment[lag:end_index]


def ols_beta_data(data, lag, length):
    gdp = data['GDP_QGR'].values
    end_index = length + lag
    return gdp[lag:end_index]

# Assignment 1
def best_model(data, beta):
    # AR(1) model rmse
    smallest_rmse = 0.8602185697579161
    best_params = []

    for i in range(2, 35):
        lags = significant_lags(data, 0.05, np.arange(1, i), beta)
        rmse = rmse_lags(data, lags)
        print(lags, rmse)
        if rmse < smallest_rmse:
            best_params = [lags, rmse]
            smallest_rmse = rmse
            print(best_params)

    return best_params


def rmse_lags(data, lags):
    ar_model = arma.ARIMA(data['GDP_QGR'], order=(lags, 0, 0)).fit()
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

    ar_model = arma.ARIMA(data['GDP_QGR'], order=(lags, 0, 0)).fit()
    summary = ar_model.summary(alpha).tables[1].data[1:]
    p_values = np.array([float(i[4]) for i in summary[1:len(summary) - 1]])

    while len(np.where(p_values < alpha)[0]) != len(p_values):
        remove_lags, removed_lag, done = remove_lag(lags, removed_lags, summary, alpha, beta)
        if done:
            break
        lags.remove(removed_lag)
        ar_model = arma.ARIMA(data['GDP_QGR'], order=(lags, 0, 0)).fit()
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
