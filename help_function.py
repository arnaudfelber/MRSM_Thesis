import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor 
import seaborn as sns 
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LassoCV, ElasticNetCV
from scipy import stats
import itertools 
from tqdm import tqdm
from itertools import combinations 
import functions as f
import warnings
import time


# Data Preprocessing part
# Function to comput log returns
def compute_log_return(series):
    shifted_series = series.shift(periods=1, axis='index').replace(to_replace=0, value=np.nan)
    ratio = series / shifted_series
    # Replace any negative and zero values with NaN (since log is undefined for negative numbers)
    ratio[ratio <= 0] = np.nan
    # Calculate the log returns
    log_returns = np.log(ratio)
    return log_returns


# Function to cap outliers
def cap_outliers(df, lower_quantile):
    capped_df = df.copy()
    upper_quantile= 1-lower_quantile
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]): # Apply only to numeric columns
            lower_cap = df[column].quantile(lower_quantile)
            upper_cap = df[column].quantile(upper_quantile)
            capped_df[column] = np.clip(df[column], lower_cap, upper_cap)
    return capped_df