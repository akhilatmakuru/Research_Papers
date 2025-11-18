#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


# In[2]:


roi_df = pd.read_csv("/Users/Akhila/pyens/UoR_PhD/999.Final_Results/07.Feature_Selection_Methodologies/00.Dataset/MRI_rois_allfeatures.csv")
roi_df


# In[3]:


labelencoder = LabelEncoder()
roi_df['SD002'] = labelencoder.fit_transform(roi_df['SD002'])
roi_df


# In[4]:


labelencoder = LabelEncoder()
roi_df['SD003'] = labelencoder.fit_transform(roi_df['SD003'])
roi_df


# In[5]:


features = pd.read_excel("/Users/Akhila/pyens/UoR_PhD/999.Final_Results/07.Feature_Selection_Methodologies/01.Method_Correlation/all_decreasingorderedfeatures_80.xlsx")
features


# In[6]:


selected_df = features[features[0].str.contains('Keep')]


# In[7]:


len(selected_df)


# In[8]:


names = selected_df['Unnamed: 0'].values


# In[9]:


X = roi_df[names].copy()
X


# In[10]:


#Normalise the data between 0 and 1
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_scaled = sc.fit_transform(X)


# In[11]:


y = roi_df[["SD003"]].copy()
y


# In[12]:


y_np = y.to_numpy()


# In[13]:


from sklearn.model_selection import StratifiedKFold 

skf = StratifiedKFold(n_splits= 10, shuffle=True, random_state=1) 
score_list = [] 

for iterator in range(1, 11):
    for train_index, test_index in skf.split(X_scaled, y_np): 
        x_train_fold, x_test_fold = X_scaled[train_index], X_scaled[test_index] 
        y_train_fold, y_test_fold = y_np[train_index], y_np[test_index] 

        clf = LogisticRegression(max_iter=1000).fit(x_train_fold,y_train_fold.reshape(-1,))
        clf.predict(x_test_fold)
        clf.predict_proba(x_test_fold)
        score = clf.score(x_test_fold, y_test_fold.reshape(-1,))
        #print(score)
        score_list.append(score)


# In[14]:


len(score_list)


# In[15]:


np.mean(score_list)


# In[16]:


np.std(score_list)


# In[17]:


skf = StratifiedKFold(n_splits= 10, shuffle=True, random_state=1) 
score_list = [] 

for iterator in range(1, 11):
    for train_index, test_index in skf.split(X_scaled, y_np): 
        x_train_fold, x_test_fold = X_scaled[train_index], X_scaled[test_index] 
        y_train_fold, y_test_fold = y_np[train_index], y_np[test_index] 
        
        clf = SVC(kernel="linear")

        clf.fit(x_train_fold,y_train_fold.reshape(-1,))
        clf.predict(x_test_fold)
        score = clf.score(x_test_fold, y_test_fold.reshape(-1,))
        #print(score)
        score_list.append(score)


# In[18]:


len(score_list)


# In[19]:


np.mean(score_list)


# In[20]:


np.std(score_list)


# In[21]:


import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from keras.utils import np_utils
from tensorflow.keras import optimizers
from tensorflow import keras


# In[22]:


skf = StratifiedKFold(n_splits= 10, shuffle=True, random_state=1) 
score_list = [] 
accuracy_val = []
loss_val = []

inputlayer_nn = len(selected_df)
hidden_l1 = round(inputlayer_nn/2)
hidden_l2 = round(hidden_l1/2)

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=3,restore_best_weights=True)

for iterator in range(1, 11):
    for train_index, test_index in skf.split(X_scaled, y_np): 
        x_train_fold, x_test_fold = X_scaled[train_index], X_scaled[test_index] 
        y_train_fold, y_test_fold = y_np[train_index], y_np[test_index] 
        
        model = Sequential()
        model.add(Dense(inputlayer_nn, input_dim=inputlayer_nn, activation='relu'))
        model.add(Dense(hidden_l1, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(hidden_l2, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss= 'binary_crossentropy', optimizer= 'rmsprop', metrics=['accuracy'])

        history=model.fit(x_train_fold, y_train_fold, epochs=200, validation_data=(x_test_fold, y_test_fold),verbose=1,callbacks=[callback])

        #Print out the statistics from the classifier
        number_of_epochs_it_ran = len(history.history['loss'])
        epoch_num = number_of_epochs_it_ran + 1

        accuprint = (history.history['accuracy'][-1])
        accuracy_val.append(accuprint)

        lossprint = (history.history['loss'][-1])
        loss_val.append(lossprint)

        val = {accuprint,lossprint}


# In[23]:


accuracy_val


# In[24]:


np.mean(accuracy_val)


# In[25]:


np.std(accuracy_val)


# In[26]:


loss_val


# In[ ]:




