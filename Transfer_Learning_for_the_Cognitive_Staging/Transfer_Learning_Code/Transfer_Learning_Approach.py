#!/usr/bin/env python
# coding: utf-8

# # Gridsearch Model of Step 03 - Regression followed by categorization
# 
# **********
# #### Author: Akhila Atmakuru
# #### Supervisor: Giuseppe di Fatta
# #### University of Reading
# **********
# ### Models: 
# #### Regression followed by categorization:
# #### Classification:
# ### Data:
# ### Results: 
# *********

# ## Necessary Imports 
# The necessary imports include Pandas and NumPy for data manipulation, Matplotlib and Seaborn for data visualization, and scikit-learn for machine learning models and evaluation metrics. These libraries provide essential tools for loading, processing, visualizing, and analyzing data, as well as building and evaluating machine learning models in Python.

# In[1]:


import csv
from sklearn.metrics import mean_absolute_error, accuracy_score, mean_squared_error
from imblearn.under_sampling import RandomUnderSampler
from keras.utils import to_categorical
from sklearn.model_selection import KFold

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from keras.utils import np_utils
from tensorflow.keras import optimizers
from keras.losses import BinaryCrossentropy
from tensorflow.keras.layers import Concatenate
from keras.wrappers.scikit_learn import KerasClassifier
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import optimizers
from sklearn.utils import shuffle

from tensorflow.keras import regularizers
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.layers import BatchNormalization

from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.manifold import TSNE


# ## Data
# 
# ## First Dataset

# In[3]:


mci_age_data = pd.read_csv("/Users/Akhila/pyens/02.UoR_PhD/11.HealthPredictor_Approach/00.Dataset/MRI_rois_alldiseases.csv")
mci_age_data


# In[4]:


mci_age_data["SD003"].value_counts()


# In[5]:


mci_age_data = mci_age_data.loc[(mci_age_data['SD003'] == 'MCI')|(mci_age_data['SD003'] == 'EMCI')|(mci_age_data['SD003'] == 'LMCI') ]
mci_age_data


# In[6]:


# Assuming ground_truth_labels is your true labels
mci_age_data["SD004"] = np.round(mci_age_data["SD004"], decimals=0)
mci_age_y = mci_age_data["SD004"]
mci_age_y


# In[7]:


mci_age_data = mci_age_data.drop(["SD002","SD003","SD004","MMSE","Stage"], axis = 1)
mci_age_data


# In[8]:


mci_age_data_train, mci_age_data_test,mci_age_y_train, mci_age_y_test = train_test_split(mci_age_data, mci_age_y, test_size=0.30, random_state=42)


# In[9]:


sc = StandardScaler()
mci_age_data_train_scaled = sc.fit_transform(mci_age_data_train)
mci_age_data_test_scaled = sc.fit_transform(mci_age_data_test)


# ## Second Dataset

# In[10]:


ad_data = pd.read_csv("/Users/Akhila/pyens/02.UoR_PhD/11.HealthPredictor_Approach/00.Dataset/MRI_rois_alldiseases.csv")
ad_data


# In[11]:


ad_data = ad_data.loc[(ad_data['Stage'] == 'MILD')|(ad_data['Stage'] == 'MODERATE') ]
ad_data


# In[12]:


# Assuming ground_truth_labels is your true labels
ad_c_y = ad_data["Stage"]
ad_c_y


# In[13]:


rus = RandomUnderSampler(random_state=42)
ad_data, ad_c_y = rus.fit_resample(ad_data, ad_c_y)


# In[14]:


# Assuming ground_truth_labels is your true labels
ad_mmse_y = ad_data["MMSE"]
ad_mmse_y


# In[15]:


# Initialize the LabelEncoder
label_encoder = LabelEncoder()

ad_c_y = label_encoder.fit_transform(ad_c_y)
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("Label Mapping:")
for mapping in label_mapping.items():
    print(f" {mapping}")


# In[16]:


# Assuming ground_truth_labels is your true labels
ad_data["SD003"].value_counts()


# In[17]:


ad_data = ad_data.drop(["SD002","SD003","SD004","MMSE","Stage"], axis = 1)
ad_data


# In[18]:


ad_data_train, ad_data_test,ad_c_y_train, ad_c_y_test,ad_mmse_y_train, ad_mmse_y_test = train_test_split(ad_data, ad_c_y, ad_mmse_y, test_size=0.40, random_state=42)


# In[19]:


sc = StandardScaler()
ad_data_train_scaled = sc.fit_transform(ad_data_train)
ad_data_test_scaled = sc.fit_transform(ad_data_test)


# ## Regression - Finalized 

# In[20]:


callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',mode='min',patience=3,restore_best_weights=True)

model = Sequential()
model.add(Dense(401, input_dim=401, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(10, activation='relu'))
model.add(Dense(1,activation='relu'))

model.compile(loss= 'mean_squared_error',optimizer='rmsprop',metrics=['mean_absolute_error'])

model.fit(mci_age_data_train_scaled, mci_age_y_train, epochs=1000,validation_split=0.3,verbose=1,callbacks=[callback])

pred_y = model.predict(mci_age_data_test_scaled)

# Evaluate using regression metrics 
mae = mean_absolute_error(mci_age_y_test, pred_y)
print(f"Mean Absolute Error: {mae}")


# ## Transfer Learning

# In[21]:


# Extract the encoder part from the binary classification model
encoder_model = Model(inputs=model.input, outputs=model.layers[-8].output)

# Freeze the layers of the encoder
for layer in encoder_model.layers:
    layer.trainable = False


# In[22]:


model.layers


# In[23]:


model.layers[-8]


# ## Auto-Encoder

# ## Final autoencoder structure after Gridsearch

# In[24]:


import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

# Set the parameters
batch_size = 64
input_dim = 401
encoding_dim = 60
transfer_learning_neurons = 50

# Define the autoencoder architecture
# Input Layer
input_layer = Input(shape=(input_dim,))

# Encoder layers with Batch Normalization
#encoded = Dense(300, activation='relu')(input_layer)
#encoded = BatchNormalization()(encoded)
encoded = Dense(200, activation='relu')(input_layer)
encoded = BatchNormalization()(encoded)
encoded = Dense(100, activation='relu')(encoded)
encoded = BatchNormalization()(encoded)
encoded = Dense(encoding_dim, activation='relu')(encoded)

# Transfer-learned layers (replace with your encoder_model function)
transfer_learned_layers = Dense(transfer_learning_neurons, activation='relu')(input_layer)

# Concatenate the transfer-learned layers with the encoder output
combined_encoded = Concatenate()([encoded, transfer_learned_layers])

# Decoder layers with Batch Normalization
decoded = BatchNormalization()(combined_encoded)
decoded = Dense(100, activation='relu')(decoded)
decoded = BatchNormalization()(decoded)
decoded = Dense(200, activation='relu')(decoded)
#decoded = BatchNormalization()(decoded)
#decoded = Dense(300, activation='relu')(decoded)
decoded = Dense(input_dim, activation='relu')(decoded)

# Create the autoencoder model
autoencoder = Model(input_layer, decoded)

# Compute Mean Squared Error (MSE) loss
mse_loss = tf.keras.losses.mean_squared_error(input_layer, decoded)
autoencoder.add_loss(mse_loss)

# Compile the model with a lower learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
autoencoder.compile(optimizer='rmsprop')

# Print a summary of the model architecture
autoencoder.summary()

# Early Stopping
early_stopping = EarlyStopping(monitor='loss', mode='min', patience=5, restore_best_weights=True)

# Train the autoencoder on your data
autoencoder.fit(mci_age_data_train_scaled, mci_age_data_train_scaled, epochs=1000, batch_size=batch_size, verbose=1, callbacks=[early_stopping])


# ## Regression followed by caregorization

# In[25]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, confusion_matrix
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

# Initialize lists to store metrics
mae_list = []
rmse_list = []
mse_list = []
accuracy_list = []
conf_matrix_list = []

# Number of iterations
n_iterations = 10  # You can change this number as per your requirement

for i in range(n_iterations):
    # Step 1: Extract the encoder part from the trained autoencoder
    trained_encoder = Model(inputs=autoencoder.input, outputs=encoded)

    # Early Stopping
    early_stopping = EarlyStopping(monitor='loss', mode='min', patience=10, restore_best_weights=True)
    
    # Step 2: Build a classifier on top of the extracted encoder
    regression_input = Input(shape=(encoding_dim,))
    hidden_layer1 = Dense(240, activation='relu')(regression_input)  
    dropout1 = Dropout(0.3)(hidden_layer1) 
    hidden_layer2 = Dense(144, activation='relu')(dropout1)      
    dropout2 = Dropout(0.3)(hidden_layer2) 
    #hidden_layer5 = Dense(10, activation='relu')(dropout2)   

    # Output layer with linear activation
    regression_layer = Dense(1, activation='relu')(dropout2)
    regression_model = Model(inputs=regression_input, outputs=regression_layer)

    # Compile the model with mean squared error loss
    regression_model.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['mean_absolute_error'])

    # Train the model
    regression_model.fit(trained_encoder.predict(ad_data_train_scaled), ad_mmse_y_train, epochs=500, batch_size=batch_size, verbose=0, callbacks=[early_stopping])

    # Predict on test data
    predictions_test = regression_model.predict(trained_encoder.predict(ad_data_test_scaled))

    # Round to the nearest integer
    rounded_predictions = np.round(predictions_test).astype(int)

    # Calculate Mean Absolute Error (MAE)
    mae = mean_absolute_error(ad_mmse_y_test, predictions_test)
    mae_list.append(mae)

    # Calculate Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mean_squared_error(ad_mmse_y_test, predictions_test))
    rmse_list.append(rmse)

    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(ad_mmse_y_test, predictions_test)
    mse_list.append(mse)

    # Initialize a new array with zeros
    predicted_classes = np.zeros_like(predictions_test)

    # Apply conditions and set values accordingly
    predicted_classes[(predictions_test >= 15) & (predictions_test <= 20)] = 1
    predicted_classes[(predictions_test >= 21) & (predictions_test <= 30)] = 0

    rounded_predicted_classes = np.round(predicted_classes).astype(int)

    # Compare predicted clusters with actual labels
    accuracy = accuracy_score(ad_c_y_test, rounded_predicted_classes)
    accuracy_list.append(accuracy)
    
    # Create a confusion matrix
    conf_matrix = confusion_matrix(ad_c_y_test, rounded_predicted_classes)
    conf_matrix_list.append(conf_matrix)

    # Print progress
    print(f"Iteration {i+1}/{n_iterations} - MAE: {mae}, RMSE: {rmse}, MSE: {mse}, Accuracy: {accuracy}")

# Print average metrics
print(f"Average MAE: {np.mean(mae_list)}")
print(f"Average RMSE: {np.mean(rmse_list)}")
print(f"Average MSE: {np.mean(mse_list)}")
print(f"Average Accuracy: {np.mean(accuracy_list)}")


# In[33]:


# Print average metrics
print(f"Average MAE: {np.mean(mae_list)}")
print(f"Standard Dev. MAE: {np.std(mae_list)}")
print(f"Average RMSE: {np.mean(rmse_list)}")
print(f"Average MSE: {np.mean(mse_list)}")
print(f"Average Accuracy: {np.mean(accuracy_list)}")
print(f"Standard Dev. Accuracy: {np.std(accuracy_list)}")

