#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import csv
import tensorflow as tf    
tf.compat.v1.disable_v2_behavior() # <-- HERE !
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from keras.utils import np_utils
from tensorflow.keras import optimizers
from tensorflow import keras
from keras.losses import BinaryCrossentropy
get_ipython().run_line_magic('matplotlib', 'inline')
import shap
shap.initjs()


# In[2]:


roi_df = pd.read_csv("/Users/Akhila/pyens/UoR_PhD/999.Final_Results/06.SHAP/00.Dataset/MRI_rois_filtered.csv")
roi_df


# In[3]:


label_quality = LabelEncoder()
roi_df['SD002'] = label_quality.fit_transform(roi_df['SD002'])
roi_df


# In[4]:


labelencoder = LabelEncoder()
roi_df['SD003'] = labelencoder.fit_transform(roi_df['SD003'])
roi_df


# In[5]:


features=roi_df.copy()
features.info()
features


# In[6]:


#Drop category labels, age and gender labels for classification
y_train = features["SD003"]
x_train = features.drop(["SD002",'SD003','SD004'], axis = 1)


# In[7]:


#Normalise the data between 0 and 1
sc=MinMaxScaler()
x_train_scaled = sc.fit_transform(x_train)


# In[8]:


callback = tf.keras.callbacks.EarlyStopping(monitor='loss',mode='min',patience=3,restore_best_weights=True)


# In[9]:


shap_results = []

for x in range(300):
    print(x)
    
    model = Sequential()    
    model.add(Dense(265, input_dim=265, activation='relu'))
    model.add(Dense(150, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(75, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(30, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss= 'binary_crossentropy',optimizer='adam',metrics= ['accuracy'])
    
    model.fit(x_train_scaled, y_train, epochs=150,verbose=1,callbacks=[callback])
    
    print("Processing for shapley values")

    explainer = shap.DeepExplainer(model,x_train_scaled)
    
    shap_values_class = explainer.shap_values(x_train_scaled)
    
    vals = np.abs(shap_values_class[0])
    
    shap_results.append(sum(vals))


# In[10]:


results = pd.DataFrame(shap_results,columns = x_train.columns)
results


# In[11]:


results.to_csv("265Features_shap_MF.csv")


# In[12]:


results.to_excel("265Features_shap_MF.xlsx")

