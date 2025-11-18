#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import skbio
from sklearn_extra.cluster import KMedoids 
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples, silhouette_score


# In[4]:


roi_df = pd.read_csv("/Users/Akhila/pyens/02.UoR_PhD/999.Final_Results/07.Feature_Selection_Methodologies/00.Dataset/MRI_rois_filtered.csv")
roi_df


# In[5]:


df = roi_df.drop(["SD002","SD004"], axis= 1)
df


# In[6]:


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
df['SD003'] = labelencoder.fit_transform(df['SD003'])
df


# In[7]:


X = df[['FS106','FS125','FS191','FS143']]
X


# In[8]:


#Normalise the data between 0 and 1
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_scaled = sc.fit_transform(X)


# In[9]:


y = df['SD003']
y


# In[10]:


y_np = y.to_numpy()


# In[11]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


# In[12]:


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


# In[13]:


len(score_list)


# In[14]:


np.mean(score_list)


# In[15]:


np.std(score_list)


# In[19]:


from sklearn.model_selection import StratifiedKFold 

skf = StratifiedKFold(n_splits= 10, shuffle=True, random_state=1) 
score_list = [] 

for iterator in range(1, 11):
    for train_index, test_index in skf.split(X_scaled, y_np): 
        x_train_fold, x_test_fold = X_scaled[train_index], X_scaled[test_index] 
        y_train_fold, y_test_fold = y_np[train_index], y_np[test_index] 

        clf = SVC(probability=True).fit(x_train_fold,y_train_fold.reshape(-1,))
        clf.predict(x_test_fold)
        clf.predict_proba(x_test_fold)
        score = clf.score(x_test_fold, y_test_fold.reshape(-1,))
        #print(score)
        score_list.append(score)


# In[20]:


len(score_list)


# In[21]:


np.mean(score_list)


# In[22]:


np.std(score_list)


# In[27]:


skf = StratifiedKFold(n_splits= 10, shuffle=True, random_state=1) 
score_list = [] 
accuracy_val = []
loss_val = []

inputlayer_nn = X_scaled.shape[1]
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


# In[32]:


len(accuracy_val)


# In[33]:


np.mean(accuracy_val)


# In[34]:


np.std(accuracy_val)


# In[25]:


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from keras.utils import np_utils
from tensorflow.keras import optimizers
from tensorflow import keras


# In[26]:


import numpy as np
import pandas as pd
import skbio
from sklearn_extra.cluster import KMedoids 
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import squareform
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from scipy.linalg import eigh
from sklearn.manifold import MDS


# In[ ]:




