
# coding: utf-8

# In[1]:


import os
import json
import pandas as pd
import numpy as np
from pandas.io.json import json_normalize
from IPython.display import display
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import lightgbm as lgb
import xgboost as xgb
import lightgbm
from sklearn.model_selection import GroupKFold
from sklearn import preprocessing
import datetime as datetime
from datetime import timedelta, date
from sklearn.model_selection import train_test_split, KFold
import warnings
warnings.simplefilter("ignore")
import time
import plotly.graph_objs as go
import plotly.tools as tools
import plotly.plotly as py
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from matplotlib import pyplot


# <font size=8> </font>
# <font size=8>Task 1 : Cleaning</font>

# Load our training and test data frames into main memory. 
# FullVisitorId column converted to string format.
# 4 columns (device, geoNetwork, totals, trafficSource) are in JSON format. So we normalize those columns.

# In[2]:


def load_df(csv_path='train.csv', nrows=None):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    
    df = pd.read_csv(csv_path, 
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype={'fullVisitorId': str},
                     nrows=nrows)
    
    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")
    return df


# In[3]:


train_df = load_df('train.csv',1_00_000)
test_df = load_df("test.csv",1_00_000)


# In[4]:


train_df.describe()


# In[5]:


train_df['totals.transactionRevenue'] = train_df['totals.transactionRevenue'].astype(float)
train_df['totals.transactionRevenue'] = train_df['totals.transactionRevenue'].fillna(0)


# In[6]:


train_df.head()


# As we can see some columns are containing constant values and some are containing Nan. Lets first remove constant columns since they are of no use in our modelling.

# In[7]:


const_cols = [c for c in train_df.columns if train_df[c].nunique(dropna=False)==1 ]
const_cols


# In[8]:


len(const_cols)


# There are 19 columns which contain constant values. So remove them.

# In[9]:


train_df = train_df.drop(const_cols, axis=1)
test_df = test_df.drop(const_cols, axis=1)


# In[10]:


train_df.head()


# Let's analyse which columns are in our training set and not in our test set.

# In[11]:


print("Variables in train but not in test : ", set(train_df.columns).difference(set(test_df.columns)))


# So there are 2 columns : campaign code and transactionRevenue which are in our training data and not in test data.
# TransactionRevenue is the target variable so we have to keep it. campaign code can be removed from our training 
# dataset.

# In[12]:


train_df.drop('trafficSource.campaignCode', 1, inplace=True)
train_df.drop('sessionId', 1, inplace=True)
test_df.drop('sessionId', 1, inplace=True)
orig_df = train_df.copy()


# Some values are missing from our training and test data sets. Let's impute those values.

# In[13]:


cols_with_missing = [col for col in train_df.columns 
                                 if train_df[col].isnull().any()]
cols_with_missing


# In[14]:


len(cols_with_missing)


# In[15]:


def RunEncoder():
    cat_cols = ["channelGrouping", "device.browser", 
                "device.deviceCategory", "device.operatingSystem", 
                "geoNetwork.city", "geoNetwork.continent", 
                "geoNetwork.country", "geoNetwork.metro",
                "geoNetwork.networkDomain", "geoNetwork.region", 
                "geoNetwork.subContinent", "trafficSource.adContent", 
                "trafficSource.adwordsClickInfo.adNetworkType", 
                "trafficSource.adwordsClickInfo.gclId", 
                "trafficSource.adwordsClickInfo.page", 
                "trafficSource.adwordsClickInfo.slot", "trafficSource.campaign",
                "trafficSource.keyword", "trafficSource.medium", 
                "trafficSource.referralPath", "trafficSource.source",
                'trafficSource.adwordsClickInfo.isVideoAd', 'trafficSource.isTrueDirect',
                'visitId','totals.bounces','totals.newVisits','totals.pageviews']

    for col in cat_cols:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train_df[col].values.astype('str')) + list(test_df[col].values.astype('str')))
        train_df[col] = lbl.transform(list(train_df[col].values.astype('str')))
        test_df[col] = lbl.transform(list(test_df[col].values.astype('str')))

    num_cols = ["totals.hits", "totals.pageviews", "visitNumber", "visitStartTime", 'totals.bounces',  'totals.newVisits']  

    for col in num_cols:
        train_df[col] = train_df[col].astype(float)
        test_df[col] = test_df[col].astype(float)


# In[16]:


RunEncoder()


# In[17]:


train_df.head()


# <font size=8> </font>
# <font size=8>Task 2 : HeatMap and Plots Generation</font>

# Let's now create a heatmap of all columns taking into consideration correlation matrix.

# In[18]:


def add_time_features(df):
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='ignore')
    df['year'] = df['date'].apply(lambda x: x.year)
    df['month'] = df['date'].apply(lambda x: x.month)
    df['day'] = df['date'].apply(lambda x: x.day)
    df['weekday'] = df['date'].apply(lambda x: x.weekday())
    df['visitStartTime_'] = pd.to_datetime(df['visitStartTime'],unit="s")
    df['visitStartTime_year'] = df['visitStartTime_'].apply(lambda x: x.year)
    df['visitStartTime_month'] = df['visitStartTime_'].apply(lambda x: x.month)
    df['visitStartTime_day'] = df['visitStartTime_'].apply(lambda x: x.day)
    df['visitStartTime_weekday'] = df['visitStartTime_'].apply(lambda x: x.weekday())
    return df


# In[19]:


corr_train = train_df.copy()
train_df = add_time_features(train_df)
test_df = add_time_features(test_df)


# In[20]:


train_df.info()


# In[21]:


corr_mat = corr_train.corr()
corr_mat


# In[22]:


f, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr_mat, ax=ax, cmap="YlGnBu", linewidths=0.1)


# Let's create a heatmap involving only subset of columns.

# In[23]:


cat_cols1 =  ['device.browser', 'device.deviceCategory', 'totals.pageviews', 
            'device.operatingSystem', 'geoNetwork.continent','geoNetwork.subContinent', 'totals.newVisits',
            'totals.hits','geoNetwork.city','totals.transactionRevenue']
new_train = train_df[cat_cols1].copy()
new_train['totals.transactionRevenue'] = new_train['totals.transactionRevenue'].fillna(0.0).astype(float)
new_train.head()


# In[24]:


corr_mat = new_train.corr()
f, ax = plt.subplots(figsize=(22, 10))
sns.heatmap(corr_mat,annot=True, ax=ax, cmap="magma_r", linewidths=0.1)


# It is showing direct positive correlation between page views and total hits. The more page views mean more page hits.

# In[25]:


def ImputeMissingValues(train):
    for df in [train]:
        df['trafficSource.adContent'].fillna('N/A', inplace=True)
        df['trafficSource.adwordsClickInfo.slot'].fillna('N/A', inplace=True)
        df['trafficSource.adwordsClickInfo.page'].fillna(0.0, inplace=True)
        df['trafficSource.adwordsClickInfo.isVideoAd'].fillna('N/A', inplace=True)
        df['trafficSource.adwordsClickInfo.adNetworkType'].fillna('N/A', inplace=True)
        df['trafficSource.adwordsClickInfo.gclId'].fillna('N/A', inplace=True)
        df['trafficSource.isTrueDirect'].fillna('N/A', inplace=True)
        df['trafficSource.referralPath'].fillna('N/A', inplace=True)
        df['trafficSource.keyword'].fillna('N/A', inplace=True)
        df['totals.bounces'].fillna(0.0, inplace=True)
        df['totals.newVisits'].fillna(0.0, inplace=True)
        df['totals.pageviews'].fillna(0.0, inplace=True)
        return df
    
orig_df = ImputeMissingValues(orig_df)
orig_df = add_time_features(orig_df)


# In[26]:


orig_df.info()


# In[27]:


orig_df["date"] = pd.to_datetime(orig_df["date"],format="%Y%m%d")
orig_df["visitStartTime"] = pd.to_datetime(orig_df["visitStartTime"],unit='s')
revenue_datetime_df = orig_df[["totals.transactionRevenue" , "date", "totals.pageviews"]].dropna()
revenue_datetime_df["totals.transactionRevenue"] =revenue_datetime_df["totals.transactionRevenue"].astype(np.int64)
revenue_datetime_df["totals.pageviews"] =revenue_datetime_df["totals.pageviews"].astype(np.int64)
revenue_datetime_df.head()


# In[28]:


total_revenue_daily_df = revenue_datetime_df.groupby(by=["date"],axis=0).sum()
total_visitNumber_daily_df = orig_df[["date","visitNumber"]].groupby(by=["date"],axis=0).sum()


# In[29]:


datetime_revenue_visits_df = pd.concat([total_revenue_daily_df,total_visitNumber_daily_df],axis=1)
fig, ax1 = plt.subplots(figsize=(25,10))
t = datetime_revenue_visits_df.index
s1 = datetime_revenue_visits_df["visitNumber"]
ax1.plot(t, s1, 'b-')
ax1.set_xlabel('day')
ax1.set_ylabel('visitNumber', color='b')
ax1.tick_params('y', colors='b')
ax2 = ax1.twinx()
s2 = datetime_revenue_visits_df["totals.transactionRevenue"]
ax2.plot(t, s2, 'r--')
ax2.set_ylabel('revenue', color='r')
ax2.tick_params('y', colors='r')
fig.tight_layout()


# The figure above shows that when there is more visit the revenue is more in those days. Also we can see the peak at around December which signifies that the number of visits are more in December and revenue is more as well. This signifies that people tend to make more visits and purchase lot of items during Christmas as compared to other days. 

# In[30]:


revenue_datetime_df["totals.transactionRevenue"] = np.log1p(revenue_datetime_df["totals.transactionRevenue"])
revenue_datetime_df["totals.transactionRevenue"] = revenue_datetime_df[revenue_datetime_df["totals.transactionRevenue"]>0]
plt.figure(figsize=(12, 8))
plt.title('pageviews vs revenue');
plt.scatter(revenue_datetime_df['totals.pageviews'], revenue_datetime_df["totals.transactionRevenue"]);
plt.xlabel('Pageviews');
plt.ylabel('Revenue');


# The plot above signifies that more pageviews signifies more natural log of transaction revenue.

# In[31]:


fig, axes = plt.subplots(2,2,figsize=(15,15))
orig_df["device.isMobile"].value_counts().plot(kind="bar",ax=axes[0][0],rot=25,legend="device.isMobile",color='red')
orig_df["device.browser"].value_counts().head(10).plot(kind="bar",ax=axes[0][1],rot=40,legend="device.browser",color='green')
orig_df["device.deviceCategory"].value_counts().head(10).plot(kind="bar",ax=axes[1][0],rot=25,legend="device.deviceCategory",color='blue')
orig_df["device.operatingSystem"].value_counts().head(10).plot(kind="bar",ax=axes[1][1],rot=80,legend="device.operatingSystem",color='brown')


# Operating systems results : 
# TOP 1 => Windows - 38%, 
# TOP 2 => Macintosh - 27%, 
# TOP 3 => Android - 15%, 
# TOP 4 => iOS - 11%, 
# TOP 5 => Linux - 4%

# Percentual of Device category: 
# desktop    73%, 
# mobile     23%, 
# tablet      3%

# Browser results :
# TOP 1 - CHROME - 69%,
# TOP 2 - SAFARI - 20%,
# TOP 3 - FIREFOX - 3%

# In[32]:


sns.set(style="white")
titanic = sns.load_dataset("titanic")
sns.countplot(orig_df['channelGrouping'],order=orig_df['channelGrouping'].value_counts().iloc[0:10].index, data = titanic)
plt.xticks(rotation=90);


# ChannelGrouping Results:
# TOP 1 => Organic Search - 43%, 
# TOP 2 => Social - 24%, 
# TOP 3 => Direct - 15%, 
# TOP 4 => Referral - 12%, 
# TOP 5 => Paid Search - 3%

# In[33]:


# plotting natural log of transactional revenue according to dates
new_copy = orig_df.copy()
new_copy.loc[:, "totals.transactionRevenue_ln"] = np.log1p(orig_df["totals.transactionRevenue"].fillna(0).astype("float"))
new_copy.groupby("date")["totals.transactionRevenue_ln"].agg(['sum', 'count', 'mean']).plot(subplots=True, sharex=True, title="Revenue Based On Date", linewidth=3)


# <font size=8> </font>
# <font size=8>Task 3 : Geography Plots</font>

# Let's do clustering of geography plots.

# In[34]:


def horizontal_bar_chart(cnt_srs, color):
    trace = go.Bar(
        y=cnt_srs.index[::-1],
        x=cnt_srs.values[::-1],
        showlegend=False,
        orientation = 'h',
        marker=dict(
            color=color,
        ),
    )
    return trace


# In[35]:


orig_df["totals.transactionRevenue"] = orig_df["totals.transactionRevenue"].astype('float')
cnt_srs = orig_df.groupby('geoNetwork.country')['totals.transactionRevenue'].agg(['size', 'count', 'mean'])
cnt_srs.columns = ["count", "count of non-zero revenue", "mean"]
cnt_srs = cnt_srs.sort_values(by="count", ascending=False)
trace1 = horizontal_bar_chart(cnt_srs["count"].head(5), 'rgba(128,0,128, 1.0)')
trace2 = horizontal_bar_chart(cnt_srs["count of non-zero revenue"].head(5), 'rgba(128,0,128, 1.0)')
trace3 = horizontal_bar_chart(cnt_srs["mean"].head(5), 'rgba(128,0,128, 1.0)')

cnt_srs = orig_df.groupby('geoNetwork.continent')['totals.transactionRevenue'].agg(['size', 'count', 'mean'])
cnt_srs.columns = ["count", "count of non-zero revenue", "mean"]
cnt_srs = cnt_srs.sort_values(by="count", ascending=False)
trace4 = horizontal_bar_chart(cnt_srs["count"].head(5), 'rgba(128,0,128, 1.0)')
trace5 = horizontal_bar_chart(cnt_srs["count of non-zero revenue"].head(5), 'rgba(128,0,128, 1.0)')
trace6 = horizontal_bar_chart(cnt_srs["mean"].head(5), 'rgba(128,0,128, 1.0)')

cnt_srs = orig_df.groupby('geoNetwork.subContinent')['totals.transactionRevenue'].agg(['size', 'count', 'mean'])
cnt_srs.columns = ["count", "count of non-zero revenue", "mean"]
cnt_srs = cnt_srs.sort_values(by="count", ascending=False)
trace7 = horizontal_bar_chart(cnt_srs["count"].head(10), 'rgba(128,0,128, 1.0)')
trace8 = horizontal_bar_chart(cnt_srs["count of non-zero revenue"].head(10), 'rgba(128,0,128, 1.0)')
trace9 = horizontal_bar_chart(cnt_srs["mean"].head(10), 'rgba(128,0,128, 1.0)')


# In[36]:


fig = tools.make_subplots(rows=3, cols=3, vertical_spacing=0.04, 
                          subplot_titles=["Country - Count", "Country - Non-zero Revenue Count","Country - Mean Revenue",
                                          "Continent - Count", "Continent - Non-zero Revenue Count","Continent - Mean Revenue",
                                          "SubContinent - Count", "SubContinent - Non-zero Revenue Count","SubContinent - Mean Revenue",  
                                          ])


# In[37]:


fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)
fig.append_trace(trace3, 1, 3)
fig.append_trace(trace4, 2, 1)
fig.append_trace(trace5, 2, 2)
fig.append_trace(trace6, 2, 3)
fig.append_trace(trace7, 3, 1)
fig.append_trace(trace8, 3, 2)
fig.append_trace(trace9, 3, 3)


# In[38]:


fig['layout'].update(height=1200, width=1200, paper_bgcolor='rgb(233,233,233)', title="Geography Plots")
py.iplot(fig, filename='Geography-plots')


# United States has the largest transaction revenue in terms of country results. In terms of continent Americas is at the top. Northern Anerica has maximum transaction revenue in subcontinents.

# In[39]:


fig, axes = plt.subplots(2,3,figsize=(15,15))
orig_df["geoNetwork.city"].value_counts().head(10).plot(kind="bar",ax=axes[0][0],rot=90,legend="geoNetwork.city",color='red')
orig_df["geoNetwork.continent"].value_counts().head(10).plot(kind="bar",ax=axes[0][1],rot=90,legend="geoNetwork.continent",color='green')
orig_df["geoNetwork.country"].value_counts().head(10).plot(kind="bar",ax=axes[0][2],rot=90,legend="geoNetwork.country",color='blue')
orig_df["geoNetwork.networkDomain"].value_counts().head(10).plot(kind="bar",ax=axes[1][0],rot=90,legend="geoNetwork.networkDomain",color='brown')
orig_df["geoNetwork.region"].value_counts().head(10).plot(kind="bar",ax=axes[1][1],rot=90,legend="geoNetwork.region",color='yellow')
orig_df["geoNetwork.subContinent"].value_counts().head(10).plot(kind="bar",ax=axes[1][2],rot=90,legend="geoNetwork.subContinent",color='purple')


# Sub-Continents Results :
# TOP 1 => Northern America - 44%, 
# TOP 2 => Southeast Asia - 8%, 
# TOP 3 => Western Europe - 6%, 
# TOP 4 => Southern Asia - 6%, 
# TOP 5 => Northern Europe - 6%

# <font size=8>Task 4 : Buying Score Probability Function </font>

# In[40]:


df_new = train_df[['fullVisitorId','visitNumber','totals.transactionRevenue']].copy()
df_new['totals.transactionRevenue'] = df_new['totals.transactionRevenue'].astype(bool)
agg_dict = {}
agg_dict['visitNumber'] = "max"
agg_dict['totals.transactionRevenue'] = "sum"
df_new = df_new.groupby('fullVisitorId').agg(agg_dict).reset_index()
df_new[df_new['totals.transactionRevenue'] > 1].head
df_new['buy_probability'] = df_new['totals.transactionRevenue']/df_new['visitNumber']
top_clients = df_new.sort_values('buy_probability', ascending= False)['fullVisitorId'][0:10]
print("Top 10 users who are most likely to buy from GStore are :\n" , top_clients.values)


# <font size=8>Task 5 : Addition of External Data Set </font>

# I have taken external data set from https://www.kaggle.com/satian/exported-google-analytics-data. There are 4 files : Test_external_data.csv, Test_external_data_2.csv, Train_external_data.csv, Train_external_data_2.csv. I have used only 2 of them. They contain 6 columns : Client Id, Sessions, Avg_Session_Duration, Bounce_Rate, Revenue Transactions, Goal Conversion Rate. It really helps in making the accurate predictions of the total revenue per user. My rank improves from around 1200 to 846 after incorporating this data set into my training model.

# Let's work on external data set. I merged the columns of external data set into our existing data sets.

# In[41]:


# load external dataset
train_data = pd.read_csv('Train_external_data.csv', low_memory=False, skiprows=6, dtype={"Client Id":'str'})
test_data = pd.read_csv('Test_external_data.csv', low_memory=False, skiprows=6, dtype={"Client Id":'str'})

for df in [train_data,test_data]:
    df["visitId"] = df["Client Id"].apply(lambda x: x.split('.', 1)[1]).astype(str) 
    
cat_cols = ['Revenue','Sessions','Avg. Session Duration', 'Bounce Rate', 'Transactions', 'Goal Conversion Rate',
           'visitId']

# did label encoding on the external data set
for col in cat_cols:
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(train_data[col].values.astype('str')) + list(test_data[col].values.astype('str')))
    train_data[col] = lbl.transform(list(train_data[col].values.astype('str')))
    test_data[col] = lbl.transform(list(test_data[col].values.astype('str')))

train_df.info()
train_new = train_df.merge(train_data, how="left", on="visitId")     
test_new = test_df.merge(test_data, how="left", on="visitId")

for df in [train_new, test_new]:
    df.drop("Client Id", axis = 1, inplace=True)
    
# imputing Nan values    
for df in [train_new, test_new]:
    df["Sessions"] = df["Sessions"].fillna(0)
    df["Avg. Session Duration"] = df["Avg. Session Duration"].fillna(0)
    df["Bounce Rate"] = df["Bounce Rate"].fillna(0)
    df["Revenue"] = df["Revenue"].fillna(0)
    df["Transactions"] = df["Transactions"].fillna(0)
    df["Goal Conversion Rate"] = df["Goal Conversion Rate"].fillna(0)
    df['trafficSource.adContent'].fillna('N/A', inplace=True)
    df['trafficSource.adwordsClickInfo.slot'].fillna('N/A', inplace=True)
    df['trafficSource.adwordsClickInfo.page'].fillna(0.0, inplace=True)
    df['trafficSource.adwordsClickInfo.isVideoAd'].fillna('N/A', inplace=True)
    df['trafficSource.adwordsClickInfo.adNetworkType'].fillna('N/A', inplace=True)
    df['trafficSource.adwordsClickInfo.gclId'].fillna('N/A', inplace=True)
    df['trafficSource.isTrueDirect'].fillna('N/A', inplace=True)
    df['trafficSource.referralPath'].fillna('N/A', inplace=True)
    df['trafficSource.keyword'].fillna('N/A', inplace=True)
    df['totals.bounces'].fillna(0.0, inplace=True)
    df['totals.newVisits'].fillna(0.0, inplace=True)
    df['totals.pageviews'].fillna(0.0, inplace=True)

del train_df
del test_df
train_df = train_new
test_df = test_new
del train_new
del test_new


# In[42]:


train_df.info()


# <font size=8>  </font>
# <font size=8>Task 6 : Predictive Model</font>

# I have used LGBMRegressor with KFold incorporating various factors and adding some more features related to dates.

# In[43]:


def add_time_features(df):
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='ignore')
    df['year'] = df['date'].apply(lambda x: x.year)
    df['month'] = df['date'].apply(lambda x: x.month)
    df['day'] = df['date'].apply(lambda x: x.day)
    df['weekday'] = df['date'].apply(lambda x: x.weekday())
    df['visitStartTime_'] = pd.to_datetime(df['visitStartTime'],unit="s")
    df['visitStartTime_year'] = df['visitStartTime_'].apply(lambda x: x.year)
    df['visitStartTime_month'] = df['visitStartTime_'].apply(lambda x: x.month)
    df['visitStartTime_day'] = df['visitStartTime_'].apply(lambda x: x.day)
    df['visitStartTime_weekday'] = df['visitStartTime_'].apply(lambda x: x.weekday())
    return df

date_features = [#"year","month","day","weekday",'visitStartTime_year',
    "visitStartTime_month","visitStartTime_day","visitStartTime_weekday","visitStartTime_year"]

# add date-time features to train into our model.
train_df = add_time_features(train_df)
test_df = add_time_features(test_df)

# add more features related to hits, pageviews, revenue.
for df in [train_df, test_df]:
    df['hits/pageviews'] = (df["totals.pageviews"]/(df["totals.hits"])).apply(lambda x: 0 if np.isinf(x) else x)
    df['is_high_hits'] = np.logical_or(df["totals.hits"]>5,df["totals.pageviews"]>5).astype(np.int32)
    df["Revenue"] = np.log1p(df["Revenue"])

# drop columns which are insignificant
no_use = ['visitStartTime', "date", "fullVisitorId","visitId", 'trafficSource.referralPath','visitStartTime_',
         'visitStartTime_year','totals.transactionRevenue']
train_df['totals.transactionRevenue'] = np.log1p(train_df['totals.transactionRevenue'])
X = train_df.drop(no_use, axis=1)
y = train_df['totals.transactionRevenue']
X_test = test_df.drop([col for col in no_use if col in test_df.columns], axis=1)


# In[44]:


params = {"objective" : "regression",
          "metric" : "rmse", 
          "min_child_samples": 20, 
          "reg_alpha": 0.033948965191129526, 
          "reg_lambda": 0.06490202783578762,
          "num_leaves" : 34,
          "learning_rate" : 0.019732018807662323,
          "subsample" : 0.876,
          "colsample_bytree" : 0.85,
          "subsample_freq ": 5
         }

# run KFold with LGBMRegressor
n_fold = 5
folds = KFold(n_splits=n_fold, random_state=42)
model = lgb.LGBMRegressor(**params, n_estimators = 10000, nthread = 4, n_jobs = -1)


# In[45]:


# submit our predicted results
prediction = np.zeros(len(test_df))
for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
    print('Fold', fold_n, 'started at', time.ctime())
    X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

    model.fit(X_train, y_train, 
                eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric='rmse',
                verbose=500, early_stopping_rounds=100)
y_pred = model.predict(X_test, num_iteration=model.best_iteration_)
prediction += y_pred
prediction /= n_fold
actual_score = model.best_score_
submission = test_df[['fullVisitorId']].copy()
submission.loc[:, 'PredictedLogRevenue'] = prediction
grouped_test = submission[['fullVisitorId', 'PredictedLogRevenue']].groupby('fullVisitorId').sum().reset_index()
grouped_test.to_csv('submit1.csv',index=False)


# <font size=8>Task 7 : Permutation Test </font>

# In[46]:


import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(model, random_state=1).fit(X, y)
eli5.show_weights(perm, feature_names = X.columns.tolist())


# Above Table shows the importance of each features.

# In[47]:


fig, ax = plt.subplots(figsize=(12,18))
lgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
ax.grid(False)
plt.title("LightGBM - Feature Importance", fontsize=15)
plt.show()


# In[48]:


from sklearn.metrics import mean_squared_error
def score(data, y):
    validation_res = pd.DataFrame(
    {"fullVisitorId": data["fullVisitorId"].values,
     "transactionRevenue": data["totals.transactionRevenue"].values,
     "PredictedLogRevenue": np.expm1(y)})

    validation_res = validation_res.groupby("fullVisitorId")["transactionRevenue", "PredictedLogRevenue"].sum().reset_index()
    return np.sqrt(mean_squared_error(np.log1p(validation_res["transactionRevenue"].values), 
                                     np.log1p(validation_res["predictedRevenue"].values)))


# In[49]:


def RunModel():
    prediction = np.zeros(len(test_df))
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
        print('Fold', fold_n, 'started at', time.ctime())
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        model.fit(X_train, y_train, 
                eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric='rmse',
                verbose=500, early_stopping_rounds=100)
    y_pred = model.predict(X_test, num_iteration=model.best_iteration_)
    prediction += y_pred
    prediction /= n_fold
    actual_score = model.best_score_
    return actual_score


# In[50]:


def Permute(col):
    scorelist = []
    for i in range(3):
        X[col] = np.random.permutation(X[col])
        score = RunModel()
        scorelist.append(score['training']['rmse'])
    return np.mean(scorelist)


# Let's run the permutation test by random shuffling column and compare their rmse with the actual model.

# In[51]:


selected_col = 'visitNumber'
val = Permute(selected_col)


# In[52]:


selected_col = 'totals.pageviews'
val = Permute(selected_col)


# In[53]:


selected_col = 'totals.hits'
val = Permute(selected_col)


# In[54]:


def Permute(col1, col2):
    scorelist = []
    for i in range(3):
        X[col1] = np.random.permutation(X[col1])
        X[col2] = np.random.permutation(X[col2])
        score = RunModel()
        scorelist.append(score['training']['rmse'])
    return np.mean(scorelist)


# In[55]:


col1 = 'totals.hits'
col2 = 'totals.pageviews'
val = Permute(col1, col2)

