import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.graphics.tsaplots as tsplots
import statsmodels.tsa.arima.model as arma
import helper_functions
import warnings
import sys

if not sys.warnoptions:
    warnings.simplefilter("ignore")

# data is a dataframe, first column is year, second column is GDP
data = pd.read_csv('./data_assign_p2.csv', delimiter=',', names=['obs', 'GDP_QGR', 'UN_RATE'], skiprows=1)
