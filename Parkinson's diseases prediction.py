#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score


# In[3]:


parkinsons_data = pd.read_csv('parkinsons.csv')


# In[4]:


parkinsons_data.head()


# In[5]:


parkinsons_data.describe()


# In[6]:


parkinsons_data['status'].value_counts()


# In[7]:


parkinsons_data.groupby('status').mean()


# In[8]:


X = parkinsons_data.drop(columns=['name','status'], axis=1)
Y = parkinsons_data['status']


# In[9]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)


# In[10]:


print(X.shape, X_train.shape, X_test.shape)


# In[11]:


scaler = StandardScaler()


# In[12]:


scaler.fit(X_train)


# In[13]:


X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)


# In[14]:


model = svm.SVC(kernel='linear')


# In[15]:


model.fit(X_train, Y_train)


# In[16]:


X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)


# In[17]:


print('Accuracy score of training data : ', training_data_accuracy)


# In[18]:


X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)


# In[19]:


print('Accuracy score of test data : ', test_data_accuracy)


# In[ ]:


input_data = (197.07600,206.89600,192.05500,0.00289,0.00001,0.00166,0.00168,0.00498,0.01098,0.09700,0.00563,0.00680,0.00802,0.01689,0.00339,26.77500,0.422229,0.741367,-7.348300,0.177551,1.743867,0.085569)

# changing input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)


# In[ ]:


# standardize the data


# In[ ]:





# In[ ]:


std_data = scaler.transform(input_data_reshaped)

prediction = model.predict(std_data)


# In[20]:


print(prediction)


if (prediction[0] == 0):
  print("The Person does not have Parkinsons Disease")

else:
  print("The Person has Parkinsons")


# In[ ]:




