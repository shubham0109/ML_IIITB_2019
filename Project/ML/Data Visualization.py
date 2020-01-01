#!/usr/bin/env python
# coding: utf-8

# In[80]:


import numpy as np
import pandas as pd
from datetime import datetime
import calendar
from datetime import timedelta
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


# In[30]:


get_ipython().system('pip3 install geopy')


# In[20]:


train = pd.read_csv("./iiitb2019nyctaxifare/train.csv/train.csv", nrows = 400000)


# In[21]:


train.head()


# In[22]:


train.info()


# In[23]:


train['pickup_datetime']=pd.to_datetime(train['pickup_datetime'],format='%Y-%m-%d %H:%M:%S UTC')
train.head()


# In[24]:


from datetime import datetime
import calendar
from datetime import timedelta
import datetime as dt


# In[25]:


train['pickup_date']= train['pickup_datetime'].dt.date
train['pickup_day']=train['pickup_datetime'].apply(lambda x:x.day)
train['pickup_hour']=train['pickup_datetime'].apply(lambda x:x.hour)
train['pickup_day_of_week']=train['pickup_datetime'].apply(lambda x:calendar.day_name[x.weekday()])
train['pickup_month']=train['pickup_datetime'].apply(lambda x:x.month)
train['pickup_year']=train['pickup_datetime'].apply(lambda x:x.year)


# In[56]:


train.info()


# In[26]:


# removing outliers in latitude and longitude
misplaced_locations = 0
misplaced_locations_index = []

for i, val in enumerate(zip(train.pickup_latitude,train.dropoff_latitude,train.pickup_longitude,train.dropoff_longitude)):
    
    #print(val)
    #break
    
    lat1,lat2,lon1,lon2 = val
    #co_ords1 = (lat1, lon1)
    #co_ords2 = (lat2, lon2)
    
    if lat1 < 40.5 or lat1 > 41.8 or lat2 < 40.5 or lat2 > 41.8 or lon1 < -74.5 or lon1 > -72.8 or lon2 < -74.5 or lon2 > -72.8:
        misplaced_locations += 1
        misplaced_locations_index.append(i)
        
    
print(misplaced_locations)


# In[27]:


train = train.drop(misplaced_locations_index)


# In[28]:


train = train.dropna()


# In[31]:


import geopy.distance
for val in zip(train.pickup_latitude,train.dropoff_latitude,train.pickup_longitude,train.dropoff_longitude):
    
    #print(val)
    #break
    
    lat1,lat2,lon1,lon2 = val
    co_ords1 = (lat1, lon1)
    co_ords2 = (lat2, lon2)
    
    train['distance'] = geopy.distance.distance(co_ords1, co_ords2).km


# In[33]:


train.head(100)


# In[34]:


train = train.drop(train[train['passenger_count'] <= 0].index.tolist())


# In[35]:


train = train.drop(train[train['fare_amount'] <= 0].index.tolist())


# In[36]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[37]:


city_long_border = (-74.03, -73.75)
city_lat_border = (40.63, 40.85)

train.plot(kind='scatter', x='dropoff_longitude', y='dropoff_latitude',
                color='red', 
                s=.02, alpha=.6)
plt.title("Dropoffs")

plt.ylim(city_lat_border)
plt.xlim(city_long_border)


# In[38]:


train.plot(kind='scatter', x='pickup_longitude', y='pickup_latitude',
                color='blue', 
                s=.02, alpha=.6)
plt.title("Pickups")

plt.ylim(city_lat_border)
plt.xlim(city_long_border)


# In[39]:


#calculate trip distance in miles
def distance(lat1, lat2, lon1,lon2):
    p = 0.017453292519943295 # Pi/180
    a = 0.5 - np.cos((lat2 - lat1) * p)/2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a))


# In[40]:


train['trip_distance']=train.apply(lambda row:distance(row['pickup_latitude'],row['dropoff_latitude'],row['pickup_longitude'],row['dropoff_longitude']),axis=1)


# In[41]:


train.head()


# In[42]:


plt.scatter(x=train['trip_distance'],y=train['fare_amount'])
plt.xlabel("Trip Distance")
plt.ylabel("Fare Amount")
plt.title("Trip Distance vs Fare Amount")


# In[43]:


trips_year=train.groupby(['pickup_year'])['key'].count().reset_index().rename(columns={'key':'Num_Trips'})
trips_year.head()
sns.barplot(x='pickup_year',y='Num_Trips',data=trips_year)


# In[44]:


trips_year_fareamount=train.groupby(['pickup_year'])['fare_amount'].mean().reset_index().rename(columns={'fare_amount':'avg_fare_amount'})


# In[45]:


sns.barplot(x='pickup_year',y='avg_fare_amount',data=trips_year_fareamount).set_title("Avg Fare Amount over Years")


# In[46]:


def groupandplot(data,groupby_key,value,aggregate='mean'):
    plt.figure(figsize=(16,10))
    agg_data=data.groupby([groupby_key])[value].agg(aggregate).reset_index().rename(columns={value:aggregate+'_'+value})
    plt.subplot(1,2,1)
    count_data=train.groupby([groupby_key])['key'].count().reset_index().rename(columns={'key':'Num_Trips'})
    sns.barplot(x=groupby_key,y='Num_Trips',data=count_data).set_title("Number of Trips vs "+groupby_key)
    
    plt.subplot(1,2,2)
    sns.barplot(x=groupby_key,y=aggregate+'_'+value,data=agg_data).set_title(aggregate+'_'+value+" vs "+groupby_key)


# In[47]:


groupandplot(train,'pickup_month','fare_amount')


# In[48]:


groupandplot(train,'pickup_day_of_week','fare_amount')


# In[49]:


groupandplot(train,'pickup_hour','fare_amount')


# In[50]:


# Let us encode day of the week to numbers
def encodeDays(day_of_week):
    day_dict={'Sunday':0,'Monday':1,'Tuesday':2,'Wednesday':3,'Thursday':4,'Friday':5,'Saturday':6}
    return day_dict[day_of_week]


# In[51]:


train['pickup_day_of_week']=train['pickup_day_of_week'].apply(lambda x:encodeDays(x))


# In[52]:


groupandplot(train,'passenger_count','fare_amount')


# In[53]:


groupandplot(train,'passenger_count','fare_amount')


# In[54]:


train.to_csv("train_cleaned.csv",index=False)


# In[58]:


train.shape


# In[60]:


train.info(0)


# In[63]:


train = train.drop(columns=['key','pickup_datetime','distance','pickup_date'])


# In[64]:


train.info()


# In[70]:


def processDataForModelling(data,target,is_train=True,split=0.3):
    data_1=data
    # One hot Encoding
    data_1=pd.get_dummies(data_1)
    if is_train==True:
        X=data_1.drop([target],axis=1)
        y=data_1[target]
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=split,random_state=123)
        
        print("Shape of Training Features",X_train.shape)
        print("Shape of Validation Features ",X_test.shape)
        
        return X_train, X_test, y_train, y_test
    else:
        print ("Shape of Test Data",data_1.shape)
        return data_1


# In[73]:


X_train, X_test, y_train, y_test=processDataForModelling(train,'fare_amount',is_train=True,split=0.2)


# In[74]:


avg_fare=round(np.mean(y_train),2)
avg_fare


# In[77]:


# Baseline Model
baseline_pred=np.repeat(avg_fare,y_test.shape[0])
baseline_rmse=np.sqrt(mean_squared_error(baseline_pred, y_test))
print("Basline RMSE of Validation data :",baseline_rmse)


# In[78]:


# Linear Regression Model
lm = LinearRegression()
lm.fit(X_train,y_train)
y_pred=np.round(lm.predict(X_test),2)
lm_rmse=np.sqrt(mean_squared_error(y_pred, y_test))
print("RMSE for Linear Regression is ",lm_rmse)


# In[81]:


# Random Forest Model
rf = RandomForestRegressor(n_estimators = 100, random_state = 883,n_jobs=-1)
rf.fit(X_train,y_train)


# In[82]:


rf_pred= rf.predict(X_test)
rf_rmse=np.sqrt(mean_squared_error(rf_pred, y_test))
print("RMSE for Random Forest is ",rf_rmse)


# In[ ]:




