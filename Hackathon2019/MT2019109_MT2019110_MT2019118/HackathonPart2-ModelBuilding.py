#!/usr/bin/env python
# coding: utf-8

# In[44]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.metrics import accuracy_score
import pickle


# In[2]:


data = pd.read_csv("dataset.csv")


# In[3]:


data = data.fillna(data.mean())


# In[4]:


data.isna().sum()


# In[8]:


data = data.drop(['Parameter'], axis=1)


# In[9]:


months = data.columns[2:14]
months


# In[10]:


df1 = data[['SUBDIVISION', months[0], months[1], months[2], months[3]]]


# In[11]:


df1


# In[13]:


df1.columns = np.array(['SUBDIVISION','x1','x2','x3','x4'])


# In[14]:


df1.head()


# In[16]:


for k in range(1,9):
    df2 = data[['SUBDIVISION',months[k],months[k+1],months[k+2],months[k+3]]]
    df2.columns = np.array(['SUBDIVISION', 'x1','x2','x3','x4'])
    df1 = df1.append(df2)
df1.index = range(df1.shape[0])


# In[19]:


df1.info()


# In[20]:


df1.drop('SUBDIVISION', axis=1,inplace=True)


# In[21]:


msk = np.random.rand(len(df1)) < 0.8
df_train = df1[msk]
df_test = df1[~msk]


# In[22]:


df_train.index = range(df_train.shape[0])
df_test.index = range(df_test.shape[0])


# In[48]:


reg = linear_model.LinearRegression()
reg.fit(df_train.drop('x4',axis=1),df_train['x4'])
pickle_out = open("m1_lr.pickle","wb")
pickle.dump(reg, pickle_out)
pickle_out.close()
predicted_values = reg.predict(df_train.drop('x4',axis=1))
residuals = predicted_values-df_train['x4'].values
print('MAD (Training Data): ' + str(np.mean(np.abs(residuals))))


# In[49]:


predicted_values = reg.predict(df_test.drop('x4',axis=1))
residuals = predicted_values-df_test['x4'].values
print('MAD (Test Data): ' + str(np.mean(np.abs(residuals))))


# In[38]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

division_data = np.asarray(data[['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL',
       'AUG', 'SEP', 'OCT', 'NOV', 'DEC']])

X = None; y = None
for i in range(division_data.shape[1]-3):
    if X is None:
        X = division_data[:, i:i+3]
        y = division_data[:, i+3]
    else:
        X = np.concatenate((X, division_data[:, i:i+3]), axis=0)
        y = np.concatenate((y, division_data[:, i+3]), axis=0)
        
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=2)


# In[35]:


print(msk)


# In[36]:


len(msk)


# In[43]:


from sklearn import linear_model

# linear model
reg = linear_model.ElasticNet(alpha=0.5)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
print ("Mean Absolute Error: ", mean_absolute_error(y_test, y_pred))


# In[46]:


from sklearn.svm import SVR

# SVM model
clf = SVR(gamma='auto', C=0.1, epsilon=0.2)
clf.fit(X_train, y_train) 
pickle_out = open("m1_svr.pickle","wb")
pickle.dump(clf, pickle_out)
pickle_out.close()
y_pred = clf.predict(X_test)
print(mean_absolute_error(y_test, y_pred))


# In[50]:


predicted_values = clf.predict(df_test.drop('x4',axis=1))
residuals = predicted_values-df_test['x4'].values
print('MAD SVR (Test Data): ' + str(np.mean(np.abs(residuals))))


# In[ ]:




