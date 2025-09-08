#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


import warnings
warnings.filterwarnings('ignore')


get_ipython().run_line_magic('matplotlib', 'inline')


# In[19]:


df=pd.read_csv('bank-additional.csv' ,delimiter=';')
df.rename(columns={'y':'deposit'}, inplace=True)
df.head()


# In[20]:


df.hist(figsize=(10,10),color='skyblue')
plt.show


# In[21]:


from sklearn.model_selection import train_test_split
print(4119*0.25)


# In[22]:


from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
df_encoded =df.apply(lb.fit_transform)
df_encoded


# In[23]:


x = df_encoded.drop('deposit' ,axis=1)
y= df_encoded['deposit']
print(x.shape)
print(y.shape)
print(type(x))
print(type(y))


# In[24]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=1)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[26]:


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

def eval_model(y_test,y_pred):
    acc= accuracy_score(y_test,y_pred)
    print('Accuracy_Score' ,acc)
    cm = confusion_matrix(y_test,y_pred)
    print('Confusion Matrix\n',cm)
    print('Classification Report\n',classification_report(y_test,y_pred))

def mscore(model):
    train_score = model.score(x_train,y_train)
    test_score = model.score(x_test,y_test)
    print('Training Score',train_score)
    print('Testing Score',test_score)

from sklearn.tree import DecisionTreeClassifier

dt= DecisionTreeClassifier(criterion='gini',max_depth=5,min_samples_split=10)
dt.fit(x_train,y_train)
# In[28]:


from sklearn.tree import plot_tree


# In[29]:


cn= ['no','yes']
fn = x_train.columns
print(fn)
print(cn)


# In[30]:


plot_tree(dt,class_names=cn,filled=True)
plt.show()


# In[31]:


dt1=DecisionTreeClassifier(criterion='entropy',max_depth=4, min_samples_split=15)
dt1.fit(x_train,y_train)


# In[32]:


plt.figure(figsize=(15,15))
plot_tree(dt1,class_names=cn,filled=True)
plt.show()


# In[ ]:




