import numpy as np
import pandas as pd
from sklearn.preprocessing import minmax_scale
from statsmodels.tsa.stattools import adfuller

import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
# measure the unstability of time series

def get_std(value):
    return np.std(value)

def get_diff(value):
    sum = 0
    temp = []
    for i in range(len(value) - 1):
        sum += abs(value[i+1]-value[i])
        temp.append(value[i+1]-value[i])

    return sum/len(value),temp

def mean_times(value):
    mean_value = np.median(value)
    count = 0
    for i in range(len(value)-1):
        if (value[i]>mean_value and value[i+1]<mean_value) or (value[i]<mean_value and value[i+1]>mean_value):
            count += 1
    return count/len(value)

def adf_test(value):
    return adfuller(value)

