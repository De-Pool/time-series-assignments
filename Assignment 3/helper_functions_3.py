import numpy as np


def random_walk_forecast(xt, mean, variance, steps):
    error_terms = np.random.normal(mean, variance, (steps))
    forecast = np.zeros(steps)
    for i in range(steps):
        if i == 0:
            forecast[0] = error_terms[0] + xt
        else:
            forecast[i] = error_terms[i] + forecast[i - 1]
    return forecast


def significance(forecast, z):
    conf_lower = np.zeros(len(forecast))
    conf_upper = np.zeros(len(forecast))
    for i in range(len(forecast)):
        conf_lower[i] = forecast[i] - z
        conf_upper[i] = forecast[i] + z
    return conf_lower, conf_upper