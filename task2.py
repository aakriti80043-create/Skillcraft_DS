#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[10]:


df=pd.read_csv("train.csv")


# In[11]:


df.head()


# In[12]:


df.drop(columns=['Cabin'],inplace=True)
df['Age'].fillna(df['Age'].median(),inplace=True)
df['Embarked'].fillna(df['Embarked'].mode() [0],inplace=True)


# In[13]:


sns.set(style ="darkgrid")


# In[14]:


plt.figure(figsize=(6,4))
sns.countplot(x='Survived',data=df)
plt.title('Survial Count')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.show()


# In[16]:


plt.figure(figsize=(6,4))
sns.countplot(x='Sex',hue='Survived',data=df)
plt.title('Survival by Gender')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.show()


# In[20]:


plt.figure(figsize=(6,4))
sns.countplot(x='Pclass',hue='Survived',data=df)
plt.title('Survival by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Count')
plt.show()


# In[24]:


plt.figure(figsize=(8,4))
sns.histplot(data=df,x='Age',hue='Survived',kde=True, bins=30, element="step")
plt.title('Age Distribution by Survival')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()


# In[25]:


plt.figure(figsize=(6,6))
sns.heatmap(df.corr(numeric_only= True), annot=True, cmap='coolwarm',fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()


# In[ ]:




