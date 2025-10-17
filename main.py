import os
import pandas as pd
import numpy as np
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt


import statsmodels.api as sm
import statsmodels.formula.api as smf

from sklearn.preprocessing import LabelEncoder
la=LabelEncoder()

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import joblib
import json

import warnings
warnings.filterwarnings('ignore')

pd.options.display.max_columns=None
pd.options.display.float_format='{:.2f}'.format
#os.listdir('Salary_Data.csv')
data = pd.read_csv(r'C:\Users\kalas\OneDrive\Desktop\Scorekart\Salary_Data.csv')
df=data.copy()
df.info()
df.columns.str.strip()
