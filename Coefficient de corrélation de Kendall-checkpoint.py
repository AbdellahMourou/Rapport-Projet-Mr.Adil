#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from pylab import rcParams
import seaborn as sb# import essential libraries
import matplotlib.pyplot as plt  # data-visualization
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns  # built on top of matplotlib
sns.set()
import pandas as pd  # working with data frames

import numpy as np  # scientific computing
import sqlalchemy  # handling databases
# from sqlalchemy_utils import database_exists, create_database
import missingno as msno  # analysing missing data
import tensorflow as tf  # used to train deep neural network architectures
from tensorflow.keras import layers
from sklearn.metrics import mean_absolute_error as mae
from sklearn.preprocessing import MinMaxScaler
import warnings
from scipy.stats.stats import kendalltau
# Data Visualisation Settings 
get_ipython().run_line_magic('matplotlib', 'inline')
rcParams['figure.figsize'] = 5,4
sb.set_style('whitegrid')
# Import the data

african_crises = pd.read_csv('C:/Users/Pc/Desktop/87/der5.csv')
african_crises.head()


# In[54]:


# load the data
df = pd.read_csv('C:/Users/Pc/Desktop/87/der5.csv')  # read the data
df["timestamp"] = df["timestamp"].astype(np.datetime64)  # set the data type of the datetime column to np.datetime64
df.set_index("timestamp", inplace=True)  # set the datetime columns to be the index
df.index.name = "datetime"  # change the name of the index

# show the data frame
df.head()


# In[55]:


# create date and time features from index that may be helpful in EDA and ML algorithms
def create_date_time_features(df):
    '''create time series features from the index '''
    
    df['hour'] = df.index.hour
    df['day_of_year'] = df.index.dayofyear
    
    return df

# perform feature engineering on datetime index, and show the data frame
df = create_date_time_features(df)
df.head()


# In[56]:


# check the effect of `hour` and `day_of_year` on the `SystemProduction`
sns.set()
g = sns.jointplot(data=df[["hour", "Active_Power"]].groupby("hour").mean(), x="hour", y="Active_Power")
g.set_axis_labels(xlabel=("Hour (24-H Format)"), # set x-label
                  ylabel=("Ppower Generation (MW)"))  # set y-label
g.fig.suptitle("Hourly Energy Generation", y=1.03);  # set the title


# In[57]:


# plot the time series data
sns.set_context("poster")
plt.subplots(figsize=(50, 5)) # set the figure dimensions
sns.lineplot(x=df.index, y=df.Active_Power.values, color="blue")

# set the labels and title
plt.xlabel("Date")
plt.ylabel("Acvtive_Power")
plt.title("Time Series Data");


# In[59]:


african_crises.Active_Power.unique()

corr = african_crises.corr(method='kendall')
rcParams['figure.figsize'] = 14.7,8.27
sb.heatmap(corr, 
           xticklabels=corr.columns.values, 
           yticklabels=corr.columns.values, 
           cmap="YlGnBu",
          annot=True)


# In[60]:


# plot pairwise relationships of the dataset
sns.set()  # set the seaborn's theme as default
g = sns.pairplot(df);

# set the title
g.fig.suptitle("Pairwise Relations between Features", y=1.03);


# In[ ]:




