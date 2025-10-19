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
#df.info()
df.columns.str.strip()
df.isna().sum()
for col in df.columns:
    df[col].fillna(df[col].mode()[0], inplace=True)
df.isnull().sum().sum()
df.duplicated().sum()
df=df.drop_duplicates()
df.describe().T
df.describe(include='object').T
df.hist(bins=20, color='red', figsize=(20,10));
df['Age']=pd.to_numeric(df['Age'], errors='coerce')
df['Age_Categ']=pd.cut(df['Age'], bins=[19,30,40, 50,60, np.inf], labels=['Twenties','Thirties','Forties','Fifties','Above_Sixty'])
df['Years_Exper']=pd.cut(df['Years of Experience'], bins=[-1,5,10,15,20, np.inf], labels=['0-5','6-10','11-15','16-20','above 20'])
df['Salary_Categ'] = pd.cut(df['Salary'], bins=[0, 50000, 100000, 150000, 200000, np.inf], labels=['Low', 'Medium', 'High', 'Very High', 'Top Tier'])
df
obj=df[['Gender','Age_Categ','Education Level','Job Title','Years_Exper','Salary_Categ']]
for col in df.select_dtypes("object"):
    print(f'\n----------{col}--------------')
    print(f'\n{df[col].value_counts().reset_index().sort_values(by="count", ascending=False)[:25]}')
    print(f'\n Number of Items by {col}= {df[col].nunique()}\n')
    print("::"*33)
df['Education Level']=df['Education Level'].replace('phD', 'PhD')
df['Education Level']=df['Education Level'].replace("Bachelor's Degree", "Bachelor's")
df['Education Level']=df['Education Level'].replace("Master's Degree", "Master's")
df['Education Level']=df['Education Level'].replace("Bachelor's", "Bachelor")
df['Education Level']=df['Education Level'].replace("Master's", "Master")
df['Education Level'].value_counts(normalize=True)
num=df.select_dtypes('number')
plt.figure(figsize=(20,8))
for ind, col in enumerate(num):
    plt.subplot(1,3,1+ind)
    plt.title("Box Plot of " + col, fontsize=20)
    sns.boxplot(data=df, y=col)
plt.tight_layout()
plt.show();
num = ['Age','Years of Experience','Salary']
for col in num:
    series = pd.to_numeric(df[col], errors='coerce')   # to confirm there is no str values
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    low, upp = q1 - 1.5*iqr, q3 + 1.5*iqr
    number_outliers = series[(series < low) | (series > upp)].shape[0]
    
    print(f"\nNumber of Outliers of {col} = {number_outliers}\n")
    # display highest and lowest values to confirm if this is a real outlier or not
    print("Highest Values")
    print(f'{df[col].value_counts().reset_index().sort_values(by=col, ascending=False)[:5]}\n') 
    print("Lowest Values")
    print(f'{df[col].value_counts().reset_index().sort_values(by=col, ascending=True)[:5]}\n')
    print(f'\nMax {col}= {df[col].max()}')
    print(f'Min {col}= {df[col].min()}\n')
    print("::"*33)
for col in num:
    Q1,Q3= df[col].quantile([0.25,0.75])
    IQR= q3-q1
    Low, Upp= Q1-1.5*IQR, Q3+1.5*IQR
    df=df[(df[col]>=Low) & (df[col]<=Upp)]
Female=df[df['Gender']=='Female']
Male=df[df['Gender']=='Male']
plt.figure(figsize=(25,10))
g= sns.catplot(kind='bar', data=Female, x='Years_Exper', y='Salary', hue='Education Level', col='Salary_Categ')
g.set_xticklabels(rotation=45,fontsize=15)
plt.suptitle("Female Variables Relationships", fontsize=25, color='red', weight='bold', y=1.10) 
plt.show();
plt.figure(figsize=(25,10))
g= sns.catplot(kind='bar', data=Male, x='Years_Exper', y='Salary', hue='Education Level', col='Salary_Categ')
g.set_xticklabels(rotation=45, fontsize=15)
plt.suptitle("Male Variables Relationships", fontsize=25, color='red', weight='bold', y=1.10) 
plt.show();
