#!/usr/bin/env python
# coding: utf-8

# In[1]:


import SALib
from SALib.sample import saltelli,morris,latin,ff,finite_diff
from SALib.sample import morris
from SALib.analyze import sobol,dgsm,rbd_fast,delta,ff,dgsm
from SALib.analyze import fast
from SALib.sample import fast_sampler
from SALib.analyze import morris
from SALib.plotting.bar import plot as barplot
from SALib.plotting.morris import *
from SALib.sample import saltelli
from SALib.analyze import sobol
from SALib.test_functions import Ishigami


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from keras.utils import np_utils
from tensorflow.keras import optimizers
from tensorflow import keras
import pandas as pd
import numpy as np
from keras.losses import BinaryCrossentropy
get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


roi_df = pd.read_csv("/Users/Akhila/pyens/UoR_PhD/999.Final_Results/05.SALIB/00.Dataset/MRI_rois_filtered.csv")
roi_df


# In[8]:


label_quality = LabelEncoder()
roi_df['SD002'] = label_quality.fit_transform(roi_df['SD002'])
roi_df


# In[9]:


label_quality = LabelEncoder()
roi_df['SD003'] = label_quality.fit_transform(roi_df['SD003'])
roi_df


# In[10]:


features=roi_df.copy()
features.info()
features


# In[11]:


col_names = features.columns.values.tolist()
col_names


# In[13]:


means = features.mean()
means


# In[14]:


stds = features.std()
stds


# In[15]:


from operator import itemgetter
FS003_mean,FS004_mean,FS005_mean,FS006_mean,FS007_mean,FS008_mean,FS009_mean,FS010_mean,FS011_mean,FS012_mean,FS013_mean,FS014_mean,FS015_mean,FS016_mean,FS017_mean,FS018_mean,FS019_mean,FS020_mean,FS021_mean,FS022_mean,FS023_mean,FS024_mean,FS025_mean,FS026_mean,FS027_mean,FS028_mean,FS029_mean,FS030_mean,FS031_mean,FS032_mean,FS033_mean,FS034_mean,FS035_mean,FS036_mean,FS037_mean,FS106_mean,FS107_mean,FS108_mean,FS109_mean,FS110_mean,FS111_mean,FS112_mean,FS113_mean,FS114_mean,FS115_mean,FS116_mean,FS117_mean,FS118_mean,FS119_mean,FS120_mean,FS121_mean,FS122_mean,FS123_mean,FS124_mean,FS125_mean,FS126_mean,FS127_mean,FS128_mean,FS129_mean,FS130_mean,FS131_mean,FS132_mean,FS133_mean,FS134_mean,FS135_mean,FS136_mean,FS137_mean,FS138_mean,FS139_mean,FS140_mean,FS141_mean,FS142_mean,FS143_mean,FS144_mean,FS145_mean,FS146_mean,FS147_mean,FS148_mean,FS149_mean,FS150_mean,FS151_mean,FS152_mean,FS153_mean,FS154_mean,FS155_mean,FS156_mean,FS157_mean,FS158_mean,FS159_mean,FS160_mean,FS161_mean,FS162_mean,FS163_mean,FS164_mean,FS165_mean,FS166_mean,FS167_mean,FS168_mean,FS169_mean,FS170_mean,FS171_mean,FS172_mean,FS173_mean,FS174_mean,FS175_mean,FS176_mean,FS177_mean,FS178_mean,FS179_mean,FS180_mean,FS181_mean,FS182_mean,FS183_mean,FS184_mean,FS185_mean,FS186_mean,FS187_mean,FS188_mean,FS189_mean,FS190_mean,FS191_mean,FS192_mean,FS193_mean,FS194_mean,FS195_mean,FS196_mean,FS197_mean,FS198_mean,FS199_mean,FS200_mean,FS201_mean,FS202_mean,FS203_mean,FS204_mean,FS205_mean,FS206_mean,FS207_mean,FS276_mean,FS277_mean,FS278_mean,FS279_mean,FS280_mean,FS281_mean,FS282_mean,FS283_mean,FS284_mean,FS285_mean,FS286_mean,FS287_mean,FS288_mean,FS289_mean,FS290_mean,FS291_mean,FS292_mean,FS293_mean,FS294_mean,FS295_mean,FS296_mean,FS297_mean,FS298_mean,FS299_mean,FS300_mean,FS301_mean,FS302_mean,FS303_mean,FS304_mean,FS305_mean,FS306_mean,FS307_mean,FS308_mean,FS309_mean,FS310_mean,FS311_mean,FS312_mean,FS313_mean,FS314_mean,FS315_mean,FS316_mean,FS317_mean,FS318_mean,FS319_mean,FS320_mean,FS321_mean,FS322_mean,FS323_mean,FS324_mean,FS325_mean,FS326_mean,FS327_mean,FS328_mean,FS329_mean,FS330_mean,FS331_mean,FS332_mean,FS333_mean,FS334_mean,FS335_mean,FS336_mean,FS337_mean,FS338_mean,FS339_mean,FS340_mean,FS341_mean,FS342_mean,FS343_mean,FS344_mean,FS345_mean,FS346_mean,FS347_mean,FS348_mean,FS349_mean,FS350_mean,FS351_mean,FS352_mean,FS353_mean,FS354_mean,FS355_mean,FS356_mean,FS357_mean,FS358_mean,FS359_mean,FS360_mean,FS361_mean,FS362_mean,FS363_mean,FS364_mean,FS365_mean,FS366_mean,FS367_mean,FS368_mean,FS369_mean,FS370_mean,FS371_mean,FS372_mean,FS373_mean,FS374_mean,FS375_mean,FS376_mean,FS377_mean,FS378_mean,FS379_mean,FS380_mean,FS381_mean,FS382_mean,FS383_mean,FS384_mean,FS385_mean,FS386_mean,FS387_mean,FS388_mean,FS389_mean,FS390_mean,FS391_mean,FS392_mean,FS393_mean,FS394_mean,FS395_mean,FS396_mean,FS397_mean,FS398_mean,FS399_mean,FS400_mean,FS401_mean,FS402_mean,FS403_mean = itemgetter(3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,244,245,246,247,248,249,250,251,252,253,254,255,256,257,258,259,260,261,262,263,264,265,266,267)(means)


# In[16]:


FS003_mean


# In[17]:


from operator import itemgetter
FS003_std,FS004_std,FS005_std,FS006_std,FS007_std,FS008_std,FS009_std,FS010_std,FS011_std,FS012_std,FS013_std,FS014_std,FS015_std,FS016_std,FS017_std,FS018_std,FS019_std,FS020_std,FS021_std,FS022_std,FS023_std,FS024_std,FS025_std,FS026_std,FS027_std,FS028_std,FS029_std,FS030_std,FS031_std,FS032_std,FS033_std,FS034_std,FS035_std,FS036_std,FS037_std,FS106_std,FS107_std,FS108_std,FS109_std,FS110_std,FS111_std,FS112_std,FS113_std,FS114_std,FS115_std,FS116_std,FS117_std,FS118_std,FS119_std,FS120_std,FS121_std,FS122_std,FS123_std,FS124_std,FS125_std,FS126_std,FS127_std,FS128_std,FS129_std,FS130_std,FS131_std,FS132_std,FS133_std,FS134_std,FS135_std,FS136_std,FS137_std,FS138_std,FS139_std,FS140_std,FS141_std,FS142_std,FS143_std,FS144_std,FS145_std,FS146_std,FS147_std,FS148_std,FS149_std,FS150_std,FS151_std,FS152_std,FS153_std,FS154_std,FS155_std,FS156_std,FS157_std,FS158_std,FS159_std,FS160_std,FS161_std,FS162_std,FS163_std,FS164_std,FS165_std,FS166_std,FS167_std,FS168_std,FS169_std,FS170_std,FS171_std,FS172_std,FS173_std,FS174_std,FS175_std,FS176_std,FS177_std,FS178_std,FS179_std,FS180_std,FS181_std,FS182_std,FS183_std,FS184_std,FS185_std,FS186_std,FS187_std,FS188_std,FS189_std,FS190_std,FS191_std,FS192_std,FS193_std,FS194_std,FS195_std,FS196_std,FS197_std,FS198_std,FS199_std,FS200_std,FS201_std,FS202_std,FS203_std,FS204_std,FS205_std,FS206_std,FS207_std,FS276_std,FS277_std,FS278_std,FS279_std,FS280_std,FS281_std,FS282_std,FS283_std,FS284_std,FS285_std,FS286_std,FS287_std,FS288_std,FS289_std,FS290_std,FS291_std,FS292_std,FS293_std,FS294_std,FS295_std,FS296_std,FS297_std,FS298_std,FS299_std,FS300_std,FS301_std,FS302_std,FS303_std,FS304_std,FS305_std,FS306_std,FS307_std,FS308_std,FS309_std,FS310_std,FS311_std,FS312_std,FS313_std,FS314_std,FS315_std,FS316_std,FS317_std,FS318_std,FS319_std,FS320_std,FS321_std,FS322_std,FS323_std,FS324_std,FS325_std,FS326_std,FS327_std,FS328_std,FS329_std,FS330_std,FS331_std,FS332_std,FS333_std,FS334_std,FS335_std,FS336_std,FS337_std,FS338_std,FS339_std,FS340_std,FS341_std,FS342_std,FS343_std,FS344_std,FS345_std,FS346_std,FS347_std,FS348_std,FS349_std,FS350_std,FS351_std,FS352_std,FS353_std,FS354_std,FS355_std,FS356_std,FS357_std,FS358_std,FS359_std,FS360_std,FS361_std,FS362_std,FS363_std,FS364_std,FS365_std,FS366_std,FS367_std,FS368_std,FS369_std,FS370_std,FS371_std,FS372_std,FS373_std,FS374_std,FS375_std,FS376_std,FS377_std,FS378_std,FS379_std,FS380_std,FS381_std,FS382_std,FS383_std,FS384_std,FS385_std,FS386_std,FS387_std,FS388_std,FS389_std,FS390_std,FS391_std,FS392_std,FS393_std,FS394_std,FS395_std,FS396_std,FS397_std,FS398_std,FS399_std,FS400_std,FS401_std,FS402_std,FS403_std = itemgetter(3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,244,245,246,247,248,249,250,251,252,253,254,255,256,257,258,259,260,261,262,263,264,265,266,267)(stds)


# In[18]:


FS003_std


# In[20]:


df_features = features
df_features


# In[21]:


x_train = df_features.drop(['SD002','SD003','SD004'], axis = 1)
x_train


# In[22]:


y_train= df_features["SD003"]


# In[23]:


#Normalise the data between 0 and 1
sc=MinMaxScaler()
x_train_scaled = sc.fit_transform(x_train)


# In[24]:


problem_ABA_features = {
    'num_vars': 265,
    'names': [ 'FS003','FS004','FS005','FS006','FS007','FS008','FS009','FS010','FS011','FS012','FS013','FS014','FS015','FS016','FS017','FS018','FS019','FS020','FS021','FS022','FS023','FS024','FS025','FS026','FS027','FS028',
              'FS029','FS030','FS031','FS032','FS033','FS034','FS035','FS036','FS037','FS106','FS107','FS108',
              'FS109','FS110','FS111','FS112','FS113','FS114','FS115','FS116','FS117','FS118','FS119','FS120','FS121','FS122','FS123','FS124','FS125','FS126','FS127','FS128','FS129','FS130','FS131','FS132','FS133','FS134','FS135',
              'FS136','FS137','FS138','FS139','FS140','FS141','FS142','FS143','FS144','FS145','FS146','FS147','FS148','FS149','FS150','FS151','FS152','FS153','FS154','FS155','FS156','FS157','FS158','FS159','FS160','FS161','FS162',
              'FS163','FS164','FS165','FS166','FS167','FS168','FS169','FS170','FS171','FS172','FS173','FS174','FS175','FS176','FS177','FS178','FS179','FS180','FS181','FS182','FS183','FS184','FS185','FS186','FS187','FS188','FS189',
              'FS190','FS191','FS192','FS193','FS194','FS195','FS196','FS197','FS198','FS199','FS200','FS201','FS202','FS203','FS204','FS205','FS206','FS207','FS276','FS277','FS278','FS279','FS280','FS281','FS282','FS283','FS284','FS285','FS286','FS287','FS288','FS289','FS290','FS291','FS292','FS293','FS294','FS295','FS296','FS297',
              'FS298','FS299','FS300','FS301','FS302','FS303','FS304','FS305','FS306','FS307','FS308','FS309','FS310','FS311','FS312','FS313','FS314','FS315','FS316','FS317','FS318','FS319','FS320','FS321','FS322','FS323','FS324',
              'FS325','FS326','FS327','FS328','FS329','FS330','FS331','FS332','FS333','FS334','FS335','FS336','FS337','FS338','FS339','FS340','FS341','FS342','FS343','FS344','FS345','FS346','FS347','FS348','FS349','FS350','FS351',
              'FS352','FS353','FS354','FS355','FS356','FS357','FS358','FS359','FS360','FS361','FS362','FS363','FS364','FS365','FS366','FS367','FS368','FS369','FS370','FS371','FS372','FS373','FS374','FS375','FS376','FS377','FS378',
              'FS379','FS380','FS381','FS382','FS383','FS384','FS385','FS386','FS387','FS388','FS389','FS390','FS391','FS392','FS393','FS394','FS395','FS396','FS397','FS398','FS399','FS400','FS401','FS402','FS403'],
    'bounds': [[FS003_mean,FS003_std],[FS004_mean,FS004_std],[FS005_mean,FS005_std],[FS006_mean,FS006_std],[FS007_mean,FS007_std],[FS008_mean,FS008_std],[FS009_mean,FS009_std],[FS010_mean,FS010_std],[FS011_mean,FS011_std],[FS012_mean,FS012_std],[FS013_mean,FS013_std],
               [FS014_mean,FS014_std],[FS015_mean,FS015_std],[FS016_mean,FS016_std],[FS017_mean,FS017_std],[FS018_mean,FS018_std],[FS019_mean,FS019_std],[FS020_mean,FS020_std],[FS021_mean,FS021_std],[FS022_mean,FS022_std],[FS023_mean,FS023_std],[FS024_mean,FS024_std],
               [FS025_mean,FS025_std],[FS026_mean,FS026_std],[FS027_mean,FS027_std],[FS028_mean,FS028_std],[FS029_mean,FS029_std],[FS030_mean,FS030_std],[FS031_mean,FS031_std],[FS032_mean,FS032_std],[FS033_mean,FS033_std],[FS034_mean,FS034_std],[FS035_mean,FS035_std],
               [FS036_mean,FS036_std],[FS037_mean,FS037_std],[FS106_mean,FS106_std],[FS107_mean,FS107_std],[FS108_mean,FS108_std],[FS109_mean,FS109_std],[FS110_mean,FS110_std],[FS111_mean,FS111_std],[FS112_mean,FS112_std],
               [FS113_mean,FS113_std],[FS114_mean,FS114_std],[FS115_mean,FS115_std],[FS116_mean,FS116_std],[FS117_mean,FS117_std],[FS118_mean,FS118_std],[FS119_mean,FS119_std],[FS120_mean,FS120_std],[FS121_mean,FS121_std],[FS122_mean,FS122_std],[FS123_mean,FS123_std],
               [FS124_mean,FS124_std],[FS125_mean,FS125_std],[FS126_mean,FS126_std],[FS127_mean,FS127_std],[FS128_mean,FS128_std],[FS129_mean,FS129_std],[FS130_mean,FS130_std],[FS131_mean,FS131_std],[FS132_mean,FS132_std],[FS133_mean,FS133_std],[FS134_mean,FS134_std],
               [FS135_mean,FS135_std],[FS136_mean,FS136_std],[FS137_mean,FS137_std],[FS138_mean,FS138_std],[FS139_mean,FS139_std],[FS140_mean,FS140_std],[FS141_mean,FS141_std],[FS142_mean,FS142_std],[FS143_mean,FS143_std],[FS144_mean,FS144_std],[FS145_mean,FS145_std],
               [FS146_mean,FS146_std],[FS147_mean,FS147_std],[FS148_mean,FS148_std],[FS149_mean,FS149_std],[FS150_mean,FS150_std],[FS151_mean,FS151_std],[FS152_mean,FS152_std],[FS153_mean,FS153_std],[FS154_mean,FS154_std],[FS155_mean,FS155_std],[FS156_mean,FS156_std],
               [FS157_mean,FS157_std],[FS158_mean,FS158_std],[FS159_mean,FS159_std],[FS160_mean,FS160_std],[FS161_mean,FS161_std],[FS162_mean,FS162_std],[FS163_mean,FS163_std],[FS164_mean,FS164_std],[FS165_mean,FS165_std],[FS166_mean,FS166_std],[FS167_mean,FS167_std],
               [FS168_mean,FS168_std],[FS169_mean,FS169_std],[FS170_mean,FS170_std],[FS171_mean,FS171_std],[FS172_mean,FS172_std],[FS173_mean,FS173_std],[FS174_mean,FS174_std],[FS175_mean,FS175_std],[FS176_mean,FS176_std],[FS177_mean,FS177_std],[FS178_mean,FS178_std],
               [FS179_mean,FS179_std],[FS180_mean,FS180_std],[FS181_mean,FS181_std],[FS182_mean,FS182_std],[FS183_mean,FS183_std],[FS184_mean,FS184_std],[FS185_mean,FS185_std],[FS186_mean,FS186_std],[FS187_mean,FS187_std],[FS188_mean,FS188_std],[FS189_mean,FS189_std],
               [FS190_mean,FS190_std],[FS191_mean,FS191_std],[FS192_mean,FS192_std],[FS193_mean,FS193_std],[FS194_mean,FS194_std],[FS195_mean,FS195_std],[FS196_mean,FS196_std],[FS197_mean,FS197_std],[FS198_mean,FS198_std],[FS199_mean,FS199_std],[FS200_mean,FS200_std],
               [FS201_mean,FS201_std],[FS202_mean,FS202_std],[FS203_mean,FS203_std],[FS204_mean,FS204_std],[FS205_mean,FS205_std],[FS206_mean,FS206_std],[FS207_mean,FS207_std],[FS276_mean,FS276_std],[FS277_mean,FS277_std],
               [FS278_mean,FS278_std],[FS279_mean,FS279_std],[FS280_mean,FS280_std],[FS281_mean,FS281_std],[FS282_mean,FS282_std],[FS283_mean,FS283_std],[FS284_mean,FS284_std],[FS285_mean,FS285_std],[FS286_mean,FS286_std],[FS287_mean,FS287_std],[FS288_mean,FS288_std],
               [FS289_mean,FS289_std],[FS290_mean,FS290_std],[FS291_mean,FS291_std],[FS292_mean,FS292_std],[FS293_mean,FS293_std],[FS294_mean,FS294_std],[FS295_mean,FS295_std],[FS296_mean,FS296_std],[FS297_mean,FS297_std],[FS298_mean,FS298_std],[FS299_mean,FS299_std],
               [FS300_mean,FS300_std],[FS301_mean,FS301_std],[FS302_mean,FS302_std],[FS303_mean,FS303_std],[FS304_mean,FS304_std],[FS305_mean,FS305_std],[FS306_mean,FS306_std],[FS307_mean,FS307_std],[FS308_mean,FS308_std],[FS309_mean,FS309_std],[FS310_mean,FS310_std],
               [FS311_mean,FS311_std],[FS312_mean,FS312_std],[FS313_mean,FS313_std],[FS314_mean,FS314_std],[FS315_mean,FS315_std],[FS316_mean,FS316_std],[FS317_mean,FS317_std],[FS318_mean,FS318_std],[FS319_mean,FS319_std],[FS320_mean,FS320_std],[FS321_mean,FS321_std],
               [FS322_mean,FS322_std],[FS323_mean,FS323_std],[FS324_mean,FS324_std],[FS325_mean,FS325_std],[FS326_mean,FS326_std],[FS327_mean,FS327_std],[FS328_mean,FS328_std],[FS329_mean,FS329_std],[FS330_mean,FS330_std],[FS331_mean,FS331_std],[FS332_mean,FS332_std],
               [FS333_mean,FS333_std],[FS334_mean,FS334_std],[FS335_mean,FS335_std],[FS336_mean,FS336_std],[FS337_mean,FS337_std],[FS338_mean,FS338_std],[FS339_mean,FS339_std],[FS340_mean,FS340_std],[FS341_mean,FS341_std],[FS342_mean,FS342_std],[FS343_mean,FS343_std],
               [FS344_mean,FS344_std],[FS345_mean,FS345_std],[FS346_mean,FS346_std],[FS347_mean,FS347_std],[FS348_mean,FS348_std],[FS349_mean,FS349_std],[FS350_mean,FS350_std],[FS351_mean,FS351_std],[FS352_mean,FS352_std],[FS353_mean,FS353_std],[FS354_mean,FS354_std],
               [FS355_mean,FS355_std],[FS356_mean,FS356_std],[FS357_mean,FS357_std],[FS358_mean,FS358_std],[FS359_mean,FS359_std],[FS360_mean,FS360_std],[FS361_mean,FS361_std],[FS362_mean,FS362_std],[FS363_mean,FS363_std],[FS364_mean,FS364_std],[FS365_mean,FS365_std],
               [FS366_mean,FS366_std],[FS367_mean,FS367_std],[FS368_mean,FS368_std],[FS369_mean,FS369_std],[FS370_mean,FS370_std],[FS371_mean,FS371_std],[FS372_mean,FS372_std],[FS373_mean,FS373_std],[FS374_mean,FS374_std],[FS375_mean,FS375_std],[FS376_mean,FS376_std],
               [FS377_mean,FS377_std],[FS378_mean,FS378_std],[FS379_mean,FS379_std],[FS380_mean,FS380_std],[FS381_mean,FS381_std],[FS382_mean,FS382_std],[FS383_mean,FS383_std],[FS384_mean,FS384_std],[FS385_mean,FS385_std],[FS386_mean,FS386_std],[FS387_mean,FS387_std],
               [FS388_mean,FS388_std],[FS389_mean,FS389_std],[FS390_mean,FS390_std],[FS391_mean,FS391_std],[FS392_mean,FS392_std],[FS393_mean,FS393_std],[FS394_mean,FS394_std],[FS395_mean,FS395_std],[FS396_mean,FS396_std],[FS397_mean,FS397_std],[FS398_mean,FS398_std],
               [FS399_mean,FS399_std],[FS400_mean,FS400_std],[FS401_mean,FS401_std],[FS402_mean,FS402_std],[FS403_mean,FS403_std ]
],
    'dists': ['norm', 'norm','norm','norm','norm', 'norm','norm','norm','norm','norm','norm', 'norm','norm','norm','norm', 'norm','norm','norm','norm','norm','norm', 'norm','norm','norm','norm', 'norm','norm','norm','norm','norm','norm', 'norm','norm','norm','norm', 'norm','norm',
              'norm','norm','norm','norm', 'norm','norm','norm','norm', 'norm','norm','norm','norm','norm','norm', 'norm','norm','norm','norm', 'norm','norm','norm','norm','norm','norm', 'norm','norm','norm','norm', 'norm','norm','norm','norm','norm','norm', 'norm','norm','norm','norm',
              'norm','norm','norm','norm','norm','norm', 'norm','norm','norm','norm', 'norm','norm','norm','norm','norm','norm', 'norm','norm','norm','norm', 'norm','norm','norm','norm','norm','norm', 'norm','norm','norm','norm', 'norm','norm','norm','norm','norm','norm', 'norm','norm',
              'norm','norm', 'norm','norm','norm','norm','norm','norm', 'norm','norm','norm','norm', 'norm','norm','norm','norm','norm','norm', 'norm','norm','norm','norm', 'norm','norm','norm','norm','norm','norm', 'norm','norm','norm','norm', 'norm','norm','norm','norm','norm','norm', 
              'norm','norm', 'norm','norm','norm','norm','norm','norm', 'norm','norm','norm','norm', 'norm','norm','norm','norm','norm','norm', 'norm','norm','norm','norm', 'norm','norm','norm','norm','norm','norm', 'norm','norm','norm','norm', 'norm','norm','norm','norm','norm','norm', 
              'norm','norm','norm','norm', 'norm','norm','norm','norm','norm','norm', 'norm','norm','norm','norm', 'norm','norm','norm','norm','norm','norm', 'norm','norm','norm','norm', 'norm','norm','norm','norm','norm','norm', 'norm','norm','norm','norm', 'norm','norm','norm','norm',
              'norm','norm','norm', 'norm','norm','norm','norm', 'norm','norm','norm','norm','norm','norm', 'norm','norm','norm','norm', 'norm','norm','norm','norm','norm','norm','norm','norm','norm','norm','norm','norm','norm','norm','norm','norm','norm','norm','norm','norm','norm']
}

# Generate samples

sample_size = 256
param_values_ABA_features = fast_sampler.sample(problem_ABA_features, sample_size)


# In[25]:


param_values_ABA_features.shape


# In[26]:


param_values_ABA_features_PD = pd.DataFrame(param_values_ABA_features)
param_values_ABA_features_PD


# In[27]:


x_test_satellisample =  param_values_ABA_features_PD
x_test_satellisample


# In[28]:


#Normalise the data between 0 and 1
sc=MinMaxScaler()
x_test_satelliscaled = sc.fit_transform(x_test_satellisample)


# In[29]:


callback = tf.keras.callbacks.EarlyStopping(monitor='loss',mode='min',patience=3,restore_best_weights=True)


# In[30]:


results_list = []
final_results_list = []

for x in range(300):
    
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
    
    print(x)
    print("training Starts")
    
    model.fit(x_train_scaled, y_train, epochs=150,verbose=1,callbacks=[callback])
    
    print("training ends")
    
    print("testing Starts")
    pred_y = (model.predict(x_test_satelliscaled) > 0.5).astype("int64")
    print("testing ends")
    
    lst1 = np.unique(pred_y)
    print("Output list : ", lst1)
    print("reshaping")
    shape = len(pred_y)
    pred_y_array = pred_y.reshape(shape,)
    
    print("SA analysis")
    Si = fast.analyze(problem_ABA_features,pred_y_array, print_to_console=False)
   
    SALib_100_results = Si
    
    results_list = SALib_100_results.items()
    
    final_results_list.append(results_list)


# In[31]:


resultdf = pd.DataFrame(final_results_list) 
resultdf


# In[32]:


resultdf.to_csv("265Features_fast.csv")


# In[ ]:




