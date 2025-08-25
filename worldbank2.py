#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import  matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv("Worldbank_data.csv", skiprows=4)


# In[3]:


year='2022'


# In[6]:


df=df[["Country Name", year]].dropna()
df[year]= df[year] / 1e6


# In[7]:


top10= df.sort_values(by=year, ascending=False).head(10)


# In[10]:


plt.figure(figsize=(10,6))
plt.bar(top10["Country Name"],top10[year], color="maroon")
plt.title(f'Top 10 Most Populous Countries({year})')
plt.xlabel('Country')
plt.ylabel('Population (in millions)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[ ]:




