#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import itertools 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# In[2]:


# reading the data
df=pd.read_csv(r'C:\Users\vipul\Downloads\news.csv')


# In[3]:


df.shape
df.head()


# In[4]:


#putting labels
labels=df.label
labels.head()


# In[5]:


#splitting dataset
x_train,x_test,y_train,y_test=train_test_split(df['text'],labels,test_size=0.2,random_state=7)


# In[6]:


#Initialize a TFidfVectorizer
tfidf_vectorizer=TfidfVectorizer(stop_words='english',max_df=0.7)


# In[7]:


#Fit and Transform train set, transform test set
tfidf_train=tfidf_vectorizer.fit_transform(x_train)
tfidf_test=tfidf_vectorizer.transform(x_test)


# In[8]:


#Initiate a passiveaggressive classifier
pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)


# In[9]:


#predict on test set and calculate accuracy
y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2) }%')


# In[10]:


# Build confusion matrix
confusion_matrix(y_test,y_pred,labels=['FAKE','REAL'])


# In[ ]:




