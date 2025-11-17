#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import skbio
from sklearn_extra.cluster import KMedoids 
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples, silhouette_score


# In[2]:


roi_df = pd.read_csv("/Users/Akhila/pyens/UoR_PhD/999.Final_Results/07.Feature_Selection_Methodologies/00.Dataset/MRI_rois_allfeatures.csv")
roi_df


# In[3]:


df = roi_df.drop(["SD002","SD004"], axis= 1)
df


# In[4]:


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
df['SD003'] = labelencoder.fit_transform(df['SD003'])
df


# In[5]:


y_train = df["SD003"]
y_train


# In[6]:


df = df.drop(["SD003"], axis= 1)
df


# In[7]:


corr_matrix = df.corr()
corr_matrix


# In[8]:


corr_df = pd.DataFrame(corr_matrix)
corr_df


# In[9]:


abs_corr = corr_df.abs()
abs_corr


# In[10]:


dist = abs_corr-1
dist


# In[11]:


x_train = dist.copy()
x_train


# In[12]:


x_train_df = x_train.copy()
x_train_df


# In[13]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(x_train_df)
x_scaled = scaler.transform(x_train_df)
x_scaled


# In[14]:


my_pcoa = skbio.stats.ordination.pcoa(x_train_df.values)


# In[15]:


pcoa = my_pcoa.samples[['PC1', 'PC2']]
pcoa


# In[16]:


sse = {} 
sw = []

for k in range(2, 11):
    kmedo = KMedoids(n_clusters=k, max_iter=10, random_state=1).fit(pcoa)
    sse[k] = kmedo.inertia_
    y_kmed = kmedo.fit_predict(pcoa)
    silhouette_avg = silhouette_score(pcoa, y_kmed,metric='euclidean')
    sw.append(silhouette_avg)
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()), 'bx-')
plt.title('Elbow Method')
plt.xlabel("Number of cluster")
plt.ylabel("Inertia Score")
plt.show()

plt.plot(range(2, 11), sw)
plt.title('Silhoute Score')
plt.xlabel('Number of clusters')
plt.ylabel('Silhoute Score')      #within cluster sum of squares
plt.show()


# In[17]:


featurenameslst = []
col_names = x_train.columns
for i in range(1,11):
    kmedo = KMedoids(n_clusters = i, random_state=1)
    kmedo.fit(pcoa)
    array_centers = kmedo.cluster_centers_
    for index in kmedo.medoid_indices_:
        label = kmedo.labels_[index]
        number = index - 1 
        medoidpc1 = pcoa.iloc[number,0]
        medoidpc2 = pcoa.iloc[number,1]
        featurename = col_names[number]
        featurenameslst.append(featurename)
        print(f'{label:<5}  {medoidpc1}   {medoidpc2}        {number:<10}   {featurename}')      
    print("********************************************")


# In[18]:


res = np.array(featurenameslst)
res


# In[19]:


resss = pd.DataFrame(featurenameslst)
resss


# In[20]:


listunique = resss[0].value_counts()


# In[21]:


listunique


# In[ ]:




