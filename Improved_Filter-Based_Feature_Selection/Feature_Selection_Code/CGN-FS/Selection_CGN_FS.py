#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

#A Threshold is used to correlation values.
threshold = 0.95

#Sorting of the features based on the count column.
sorting_order = 'decreasing'


# In[2]:


roi_df = pd.read_csv("/Users/Akhila/pyens/UoR_PhD/999.Final_Results/07.Feature_Selection_Methodologies/00.Dataset/MRI_rois_allfeatures.csv")
roi_df


# In[3]:


df = roi_df.drop(["SD002","SD003","SD004"], axis= 1)
df


# In[4]:


corr_matrix = df.corr()
corr_matrix


# In[5]:


corr_df = pd.DataFrame(corr_matrix)
corr_df


# In[6]:


abs_corr = corr_df.abs()
abs_corr


# In[7]:


np.fill_diagonal(abs_corr.values, 'NaN')


# In[8]:


abs_corr


# In[9]:


if sorting_order == 'random':
    neighbour_ouput = abs_corr.gt(threshold).apply(lambda x: x.index[x].tolist(), axis=1)
    feature_names = abs_corr.index


# In[10]:


abs_corr['Count'] = (abs_corr>threshold).sum()
abs_corr


# In[11]:


abs_corr['Sum'] = abs_corr.sum(axis=1)
abs_corr


# In[12]:


abs_corr


# In[13]:


if sorting_order == 'decreasing':
    sorted_df = abs_corr.sort_values(by=['Count'], ascending=False)
    removed_sorted = sorted_df.drop(["Count","Sum"], axis= 1)
    neighbour_ouput = removed_sorted.gt(threshold).apply(lambda x: x.index[x].tolist(), axis=1)
    feature_names = removed_sorted.index
elif sorting_order == 'decreasing':
    sorted_df = abs_corr.sort_values(by=['Count'], ascending=True)
    removed_sorted = sorted_df.drop(["Count","Sum"], axis= 1)
    neighbour_ouput = removed_sorted.gt(threshold).apply(lambda x: x.index[x].tolist(), axis=1)
    feature_names = removed_sorted.index


# In[14]:


keyList = feature_names
myDict = {key: 'Keep' for key in keyList}
myDict


# In[15]:


for j,i in zip(feature_names,neighbour_ouput):
    if (i == ''):
        continue 
    else: 
        print('******')
        print('feature number')
        print(j)
        print('******')
        print('features list')
        for each in i:
            print('starting')
            print(each) 
            print('searching')
            if (str(each) in feature_names):
                myDict.update({str(each): 'Remove'})
                print('remove')
            print('ending')
        print('over')
        print('******')


# In[16]:


myDict


# In[17]:


df = pd.DataFrame(data=myDict, index=[0])
df


# In[18]:


transdf = df.T
transdf


# In[19]:


transdf.value_counts()


# In[20]:


if sorting_order == 'random': 
    transdf.to_excel('all_randomorderedfeatures_95.xlsx')
elif sorting_order == 'decreasing':
    transdf.to_excel('all_decreasingorderedfeatures.xlsx')
elif sorting_order == 'increasing':
    transdf.to_excel('all_increasingorderedfeatures.xlsx')

