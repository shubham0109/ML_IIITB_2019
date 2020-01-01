#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


data = pd.read_csv("dataset.csv")


# In[5]:


data.describe()


# In[6]:


data.info()


# In[7]:


# check if null values
data.isna().sum()


# In[10]:


# fill the null values with mean
data = data.fillna(data.mean())
data.isna().sum()


# In[15]:


data.columns


# In[12]:


data.head(100)


# In[14]:


data.hist(figsize=(20,20))


# In[17]:


data[['YEAR','JF', 'MAM',
       'JJAS', 'OND']].groupby("YEAR").sum().plot(figsize=(13,8));


# In[19]:


data[['SUBDIVISION', 'JF', 'MAM',
       'JJAS', 'OND']].groupby("SUBDIVISION").sum().plot.barh(stacked=True,figsize=(16,8));


# June July August September gets the most rainfall
# Also rainfall in Kerala, Goa, Coastal areas are much higher
# Rajasthan, Delhi, Haryana are much drier

# In[22]:


plt.figure(figsize=(15,7))
sns.heatmap(data[['JF','MAM','JJAS','OND','ANNUAL']].corr(),annot=True)
plt.show()


# In[23]:


plt.figure(figsize=(11,4))
sns.heatmap(data[['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC','ANNUAL']].corr(),annot=True)
plt.show()


# In[26]:


subdivs = data['SUBDIVISION'].unique()
num_of_subdivs = subdivs.size
print('Subdivs: ' + str(num_of_subdivs))
subdivs


# In[29]:


fig = plt.figure(figsize=(16,8))
ax = fig.add_subplot(111)
data.groupby('SUBDIVISION').mean().sort_values(by='ANNUAL', ascending=False)['ANNUAL'].plot('bar', color='b',width=0.3,title='Subdivision wise Average Annual Rainfall', fontsize=20)
plt.xticks(rotation = 90)
plt.ylabel('Average Annual Rainfall (mm)')
ax.title.set_fontsize(30)
ax.xaxis.label.set_fontsize(20)
ax.yaxis.label.set_fontsize(20)
print(data.groupby('SUBDIVISION').mean().sort_values(by='ANNUAL', ascending=False)['ANNUAL'][[0,1,2]])
print(data.groupby('SUBDIVISION').mean().sort_values(by='ANNUAL', ascending=False)['ANNUAL'][[33,34,35]])


# In[33]:


months = data.columns[2:14]
fig = plt.figure(figsize=(18,10))
ax = fig.add_subplot(111)
xlbls = data['SUBDIVISION'].unique()
xlbls.sort()
dfg = data.groupby("SUBDIVISION").mean()
dfg.plot.line(title='Overall Rainfall in Each Month of Year', ax=ax,fontsize=20)
plt.xticks(np.linspace(0,35,36,endpoint=True),xlbls)
plt.xticks(  rotation = 90)
plt.ylabel('Rainfall (mm)')
plt.legend(loc='upper right', fontsize = 'xx-large')
ax.title.set_fontsize(30)
ax.xaxis.label.set_fontsize(20)
ax.yaxis.label.set_fontsize(20)

dfg = dfg.mean(axis=0)
print('Max: ' + str(dfg.max()) + ' ocurred in ' + str(dfg.loc[dfg == dfg.max()].index.values[0:]))
print('Max: ' + str(dfg.min()) + ' ocurred in ' + str(dfg.loc[dfg == dfg.min()].index.values[0:]))
print('Mean: ' + str(dfg.mean()))


# In[ ]:




