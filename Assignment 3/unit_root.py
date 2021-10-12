import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.graphics.tsaplots as tsplots
import statsmodels.tsa.arima.model as arma
import numpy as np
import warnings
import sys

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
data = pd.read_csv('./data_assign_p2.csv', delimiter=',', names=['obs', 'GDP_QGR', 'UN_RATE'], skiprows=1)