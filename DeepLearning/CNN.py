
# coding: utf-8

# ### 使用tensorflow构建GoogLeNet构建技术分析因子的模式识别

# ### 今天偶然看到RiceQuant支持tensorflow，就把之前写的一个帖子搬到线上了。

# - GoogLeNet, 2014年ILSVRC挑战赛冠军,这个model证明了一件事：用更多的卷积，更深的层次可以得到更好的结构。（当然，它并没有证明浅的层次不能达到这样的效果）   
# 
# - 通过使用 NiN（Network-in-network）结构拓宽卷积网络的宽度和深度，其中将稀疏矩阵合并成稠密矩阵的方法和路径具有相当的工程价值。   
# 
# - 本帖使用这个NiN结构的复合滤波器对 HS300ETF 进行技术分析因子预测。并通过叠加不同指数，尝试寻找‘指数轮动’可能存在的相关关系。  
# - 本帖使用论文里面三个softmax设计降低过拟合可能

# In[34]:


import matplotlib.image as mpimg


# ### NiN结构的Inception module, GoogleLeNet核心卷积模块，一个拓宽宽度的滤波器 ,相当于一个高度非线性的滤波器

# In[35]:


image = mpimg.imread("jpb.png")
plt.figure(figsize=(14,6))
plt.axis("off")
plt.imshow(image)
plt.show()


# ### GoogleLeNet 拓扑结构图，可以看到GoogleLeNet在LeNet网络结构上面大量使用Inception_unit滤波器拓宽加深LeNet网络，Going deeper with convolutions论文中Inception_unit滤波器将稀疏矩阵合并成稠密矩阵的方法和路径具有相当的工程价值

# In[36]:


image = mpimg.imread("jpa.png")
plt.figure(figsize=(16,46))
plt.axis("off")
plt.imshow(image)
plt.show()


# In[1]:


import numpy as np
import pandas as pd
import tensorflow as tf
import Technical_Analysis_Factor_Package_Library as TAF
import datetime
import matplotlib.pylab as plt
#import seaborn as sns
get_ipython().magic('matplotlib inline')

HS300 = pd.read_csv('HS300.csv')
HS300.rename(columns={'Unnamed: 0':'tradingPeriod', 'TotalVolumeTraded':'turnoverVol',
                     'HighPx':'highestPrice', 'OpeningPx':'openPrice','TotalTurnover':'turnoverValue',
                     'LowPx':'lowestPrice', 'ClosingPx':'closePrice'}, inplace= True)
HS300 = TAF.Technical_Analysis_Factor_Normalization(inputdata= HS300, rolling=16*40, Tdropna= True)

Dtmp = pd.read_csv('HS300D.csv')
Dtmp.rename(columns={'Unnamed: 0':'tradingPeriod', 'TotalVolumeTraded':'turnoverVol',
                     'HighPx':'highestPrice', 'OpeningPx':'openPrice','TotalTurnover':'turnoverValue',
                     'LowPx':'lowestPrice', 'ClosingPx':'closePrice'}, inplace= True)

# 预测14天涨跌
Dtmp['actual_future_rate_of_return'] = Dtmp.closePrice.shift(-14)/Dtmp.closePrice - 1.0
Dtmp = Dtmp.dropna()
Dtmp = Dtmp[-200:]
Dtmp['Direction_Label'] = 0
Dtmp.actual_future_rate_of_return.describe()
Dtmp.loc[Dtmp.actual_future_rate_of_return>0.025,'Direction_Label'] = 1
Dtmp.loc[Dtmp.actual_future_rate_of_return<-0.01,'Direction_Label'] = -1
Dtmp.reset_index(drop= True , inplace= True)

start = Dtmp.tradingPeriod.values[0]
end = Dtmp.tradingPeriod.values[-1]
end = datetime.datetime.strptime(end,'%Y-%m-%d') # 将STR转换为datetime
end = end + datetime.timedelta(days=1) # 增加一天
end = end.strftime('%Y-%m-%d')

fac_HS300 = HS300.ix[(HS300.tradingPeriod.values>start) & (HS300.tradingPeriod.values<end)].reset_index(drop=True)
fac_list = TAF.get_Multifactor_list(fac_HS300)
fac_list = fac_list[:56]

fe = 56 # 回溯日期
tmp_HS300 = np.zeros((1,fe*16*56))
for i in np.arange(fe,int(len(fac_HS300)/16)):
    tmp = fac_HS300.ix[16*(i-fe):16*i-1][fac_list]
    tmp = np.array(tmp).ravel(order='C').transpose()
    tmp_HS300 = np.vstack((tmp_HS300,tmp))
tmp_HS300 = np.delete(tmp_HS300,0,axis=0)


# ### GoogLeNet用于图像识别，如果我们将技术分析因子投影到0~255范围内合成图像，则可以强假设技术分析因子合成图片使用图片分类进行涨跌预测。

# - 技术分析因子数值波动

# In[2]:


o = HS300.ATR14[:300]
p = HS300.ADX[:300]
q = HS300.DX[:300]
fig = plt.figure(figsize=(14,6))
plt.plot(p)
plt.plot(o)
plt.plot(q)


# - 多种技术分析因子数值在Y轴并列之后使用颜色表示因子数值大小

# In[3]:


import matplotlib.pyplot as plt
import matplotlib.image as mping
plt.figure(figsize=(6,6))
shpig = tmp_HS300[1]
shpig = shpig.reshape(224,224)
shpig +=4
shpig *=26
plt.axis("off")
plt.imshow(shpig)
plt.show()


# ### 数据读取部分，我这个模型是在本地建立的，懒得改代码了就直接把数据上传到平台上面了

# In[4]:


HS300 = pd.read_csv('HS300.csv')
SHCI  = pd.read_csv('SHCI.csv')
ZZ500  = pd.read_csv('ZZ500.csv')
CYBZ  = pd.read_csv('CYBZ.csv')
HS300.rename(columns={'Unnamed: 0':'tradingPeriod', 'TotalVolumeTraded':'turnoverVol',
                     'HighPx':'highestPrice', 'OpeningPx':'openPrice','TotalTurnover':'turnoverValue',
                     'LowPx':'lowestPrice', 'ClosingPx':'closePrice'}, inplace= True)
HS300 = TAF.Technical_Analysis_Factor_Normalization(inputdata= HS300, rolling=16*40, Tdropna= True)

SHCI.rename(columns={'Unnamed: 0':'tradingPeriod', 'TotalVolumeTraded':'turnoverVol',
                     'HighPx':'highestPrice', 'OpeningPx':'openPrice','TotalTurnover':'turnoverValue',
                     'LowPx':'lowestPrice', 'ClosingPx':'closePrice'}, inplace= True)
SHCI = TAF.Technical_Analysis_Factor_Normalization(inputdata= SHCI, rolling=16*40, Tdropna= True)

ZZ500.rename(columns={'Unnamed: 0':'tradingPeriod', 'TotalVolumeTraded':'turnoverVol',
                     'HighPx':'highestPrice', 'OpeningPx':'openPrice','TotalTurnover':'turnoverValue',
                     'LowPx':'lowestPrice', 'ClosingPx':'closePrice'}, inplace= True)
ZZ500 = TAF.Technical_Analysis_Factor_Normalization(inputdata= ZZ500, rolling=16*40, Tdropna= True)

CYBZ.rename(columns={'Unnamed: 0':'tradingPeriod', 'TotalVolumeTraded':'turnoverVol',
                     'HighPx':'highestPrice', 'OpeningPx':'openPrice','TotalTurnover':'turnoverValue',
                     'LowPx':'lowestPrice', 'ClosingPx':'closePrice'}, inplace= True)
CYBZ = TAF.Technical_Analysis_Factor_Normalization(inputdata= CYBZ, rolling=16*40, Tdropna= True)

Dtmp = pd.read_csv('HS300D.csv')
Dtmp.rename(columns={'Unnamed: 0':'tradingPeriod', 'TotalVolumeTraded':'turnoverVol',
                     'HighPx':'highestPrice', 'OpeningPx':'openPrice','TotalTurnover':'turnoverValue',
                     'LowPx':'lowestPrice', 'ClosingPx':'closePrice'}, inplace= True)
# 预测14天涨跌
Dtmp['actual_future_rate_of_return'] = Dtmp.closePrice.shift(-14)/Dtmp.closePrice - 1.0
Dtmp = Dtmp.dropna()
Dtmp = Dtmp[-200:]
Dtmp['Direction_Label'] = 0
Dtmp.actual_future_rate_of_return.describe()
Dtmp.loc[Dtmp.actual_future_rate_of_return>0.025,'Direction_Label'] = 1
Dtmp.loc[Dtmp.actual_future_rate_of_return<-0.01,'Direction_Label'] = -1
Dtmp.reset_index(drop= True , inplace= True)

start = Dtmp.tradingPeriod.values[0]
end = Dtmp.tradingPeriod.values[-1]
end = datetime.datetime.strptime(end,'%Y-%m-%d') # 将STR转换为datetime
end = end + datetime.timedelta(days=1) # 增加一天
end = end.strftime('%Y-%m-%d')

fac_HS300 = HS300.ix[(HS300.tradingPeriod.values>start) & (HS300.tradingPeriod.values<end)].reset_index(drop=True)
fac_ZZ500 = HS300.ix[(ZZ500.tradingPeriod.values>start) & (ZZ500.tradingPeriod.values<end)].reset_index(drop=True)
fac_SHCI = HS300.ix[(SHCI.tradingPeriod.values>start) & (SHCI.tradingPeriod.values<end)].reset_index(drop=True)
fac_CYBZ = HS300.ix[(CYBZ.tradingPeriod.values>start) & (CYBZ.tradingPeriod.values<end)].reset_index(drop=True)
fac_list = TAF.get_Multifactor_list(fac_HS300)
fac_list = fac_list[:56]

def overs(fac_HS300):
    fe = 56 # 回溯日期
    tmp_HS300 = np.zeros((1,fe*16*56))
    for i in np.arange(fe,1+int(len(fac_HS300)/16)):
        tmp = fac_HS300.ix[16*(i-fe):16*i-1][fac_list]
        tmp = np.array(tmp).ravel(order='C').transpose()
        tmp_HS300 = np.vstack((tmp_HS300,tmp))
    tmp_HS300 = np.delete(tmp_HS300,0,axis=0)
    return tmp_HS300 

fac_HS300 = overs(fac_HS300)
fac_CYBZ = overs(fac_CYBZ)
fac_SHCI = overs(fac_SHCI)
fac_ZZ500 = overs(fac_ZZ500)

fac_HS300 = fac_HS300.reshape(145,50176)
fac = np.array(fac_HS300)

ret = Dtmp.iloc[-145:,-1]
ret = np.array(ret)
wid = np.zeros((145,3))
wid[np.arange(145),ret] = 1
ret = wid


# ### GoogLeNet 拓扑结构构建部分

# In[5]:


def inception_unit(inception_in, weights, biases):
    
    # Conv 1x1+S1
    inception_1x1_S1 = tf.nn.conv2d(inception_in, weights['inception_1x1_S1'], strides=[1,1,1,1], padding='SAME')
    inception_1x1_S1 = tf.nn.bias_add(inception_1x1_S1, biases['inception_1x1_S1'])
    inception_1x1_S1 = tf.nn.relu(inception_1x1_S1)
    # Conv 3x3+S1
    inception_3x3_S1_reduce = tf.nn.conv2d(inception_in, weights['inception_3x3_S1_reduce'], strides=[1,1,1,1], padding='SAME')
    inception_3x3_S1_reduce = tf.nn.bias_add(inception_3x3_S1_reduce, biases['inception_3x3_S1_reduce'])
    inception_3x3_S1_reduce = tf.nn.relu(inception_3x3_S1_reduce)
    inception_3x3_S1 = tf.nn.conv2d(inception_3x3_S1_reduce, weights['inception_3x3_S1'], strides=[1,1,1,1], padding='SAME')
    inception_3x3_S1 = tf.nn.bias_add(inception_3x3_S1, biases['inception_3x3_S1'])
    inception_3x3_S1 = tf.nn.relu(inception_3x3_S1)
    # Conv 5x5+S1
    inception_5x5_S1_reduce = tf.nn.conv2d(inception_in, weights['inception_5x5_S1_reduce'], strides=[1,1,1,1], padding='SAME')
    inception_5x5_S1_reduce = tf.nn.bias_add(inception_5x5_S1_reduce, biases['inception_5x5_S1_reduce'])
    inception_5x5_S1_reduce = tf.nn.relu(inception_5x5_S1_reduce)
    inception_5x5_S1 = tf.nn.conv2d(inception_5x5_S1_reduce, weights['inception_5x5_S1'], strides=[1,1,1,1], padding='SAME')
    inception_5x5_S1 = tf.nn.bias_add(inception_5x5_S1, biases['inception_5x5_S1'])
    inception_5x5_S1 = tf.nn.relu(inception_5x5_S1)
    # MaxPool
    inception_MaxPool = tf.nn.max_pool(inception_in, ksize=[1,3,3,1], strides=[1,1,1,1], padding='SAME')
    inception_MaxPool = tf.nn.conv2d(inception_MaxPool, weights['inception_MaxPool'], strides=[1,1,1,1], padding='SAME')
    inception_MaxPool = tf.nn.bias_add(inception_MaxPool, biases['inception_MaxPool'])
    inception_MaxPool = tf.nn.relu(inception_MaxPool)
    # Concat
    #tf.concat(concat_dim, values, name='concat')
    #concat_dim是tensor连接的方向（维度），values是要连接的tensor链表，name是操作名。cancat_dim维度可以不一样，其他维度的尺寸必须一样。
    inception_out = tf.concat(concat_dim=3, values=[inception_1x1_S1, inception_3x3_S1, inception_5x5_S1, inception_MaxPool])
    return inception_out

weights = {
    'conv1_7x7_S2': tf.Variable(tf.random_normal([7,7,1,64])),
    'conv2_1x1_S1': tf.Variable(tf.random_normal([1,1,64,64])),
    'conv2_3x3_S1': tf.Variable(tf.random_normal([3,3,64,192])),
    'FC2': tf.Variable(tf.random_normal([7*7*832, 3])),
    'test': tf.Variable(tf.random_normal([7*7*1024,3]))
    
}

biases = {
    'conv1_7x7_S2': tf.Variable(tf.random_normal([64])),
    'conv2_1x1_S1': tf.Variable(tf.random_normal([64])),
    'conv2_3x3_S1': tf.Variable(tf.random_normal([192])),
    'FC2': tf.Variable(tf.random_normal([3])),
    'test': tf.Variable(tf.random_normal([3]))
    
}


conv_W_3a = {
    'inception_1x1_S1': tf.Variable(tf.random_normal([1,1,192,64])),
    'inception_3x3_S1_reduce': tf.Variable(tf.random_normal([1,1,192,96])),
    'inception_3x3_S1': tf.Variable(tf.random_normal([1,1,96,128])),
    'inception_5x5_S1_reduce': tf.Variable(tf.random_normal([1,1,192,16])),
    'inception_5x5_S1': tf.Variable(tf.random_normal([5,5,16,32])),
    'inception_MaxPool': tf.Variable(tf.random_normal([1,1,192,32]))
    
}

conv_B_3a = {
    'inception_1x1_S1': tf.Variable(tf.random_normal([64])),
    'inception_3x3_S1_reduce': tf.Variable(tf.random_normal([96])),
    'inception_3x3_S1': tf.Variable(tf.random_normal([128])),
    'inception_5x5_S1_reduce': tf.Variable(tf.random_normal([16])),
    'inception_5x5_S1': tf.Variable(tf.random_normal([32])),
    'inception_MaxPool': tf.Variable(tf.random_normal([32]))
}

conv_W_3b = {
    'inception_1x1_S1': tf.Variable(tf.random_normal([1,1,256,128])),
    'inception_3x3_S1_reduce': tf.Variable(tf.random_normal([1,1,256,128])),
    'inception_3x3_S1': tf.Variable(tf.random_normal([1,1,128,192])),
    'inception_5x5_S1_reduce': tf.Variable(tf.random_normal([1,1,256,32])),
    'inception_5x5_S1': tf.Variable(tf.random_normal([5,5,32,96])),
    'inception_MaxPool': tf.Variable(tf.random_normal([1,1,256,64]))

}

conv_B_3b = {
    'inception_1x1_S1': tf.Variable(tf.random_normal([128])),
    'inception_3x3_S1_reduce': tf.Variable(tf.random_normal([128])),
    'inception_3x3_S1': tf.Variable(tf.random_normal([192])),
    'inception_5x5_S1_reduce': tf.Variable(tf.random_normal([32])),
    'inception_5x5_S1': tf.Variable(tf.random_normal([96])),
    'inception_MaxPool': tf.Variable(tf.random_normal([64]))
}

conv_W_4a = {
    'inception_1x1_S1': tf.Variable(tf.random_normal([1,1,480,192])),
    'inception_3x3_S1_reduce': tf.Variable(tf.random_normal([1,1,480,96])),
    'inception_3x3_S1': tf.Variable(tf.random_normal([1,1,96,208])),
    'inception_5x5_S1_reduce': tf.Variable(tf.random_normal([1,1,480,16])),
    'inception_5x5_S1': tf.Variable(tf.random_normal([5,5,16,48])),
    'inception_MaxPool': tf.Variable(tf.random_normal([1,1,480,64]))

}

conv_B_4a = {
    'inception_1x1_S1': tf.Variable(tf.random_normal([192])),
    'inception_3x3_S1_reduce': tf.Variable(tf.random_normal([96])),
    'inception_3x3_S1': tf.Variable(tf.random_normal([208])),
    'inception_5x5_S1_reduce': tf.Variable(tf.random_normal([16])),
    'inception_5x5_S1': tf.Variable(tf.random_normal([48])),
    'inception_MaxPool': tf.Variable(tf.random_normal([64]))
}


conv_W_4b = {
    'inception_1x1_S1': tf.Variable(tf.random_normal([1,1,512,160])),
    'inception_3x3_S1_reduce': tf.Variable(tf.random_normal([1,1,512,112])),
    'inception_3x3_S1': tf.Variable(tf.random_normal([1,1,112,224])),
    'inception_5x5_S1_reduce': tf.Variable(tf.random_normal([1,1,512,24])),
    'inception_5x5_S1': tf.Variable(tf.random_normal([5,5,24,64])),
    'inception_MaxPool': tf.Variable(tf.random_normal([1,1,512,64]))

}

conv_B_4b = {
    'inception_1x1_S1': tf.Variable(tf.random_normal([160])),
    'inception_3x3_S1_reduce': tf.Variable(tf.random_normal([112])),
    'inception_3x3_S1': tf.Variable(tf.random_normal([224])),
    'inception_5x5_S1_reduce': tf.Variable(tf.random_normal([24])),
    'inception_5x5_S1': tf.Variable(tf.random_normal([64])),
    'inception_MaxPool': tf.Variable(tf.random_normal([64]))
}

conv_W_4c = {
    'inception_1x1_S1': tf.Variable(tf.random_normal([1,1,512,128])),
    'inception_3x3_S1_reduce': tf.Variable(tf.random_normal([1,1,512,128])),
    'inception_3x3_S1': tf.Variable(tf.random_normal([1,1,128,256])),
    'inception_5x5_S1_reduce': tf.Variable(tf.random_normal([1,1,512,24])),
    'inception_5x5_S1': tf.Variable(tf.random_normal([5,5,24,64])),
    'inception_MaxPool': tf.Variable(tf.random_normal([1,1,512,64]))

}

conv_B_4c = {
    'inception_1x1_S1': tf.Variable(tf.random_normal([128])),
    'inception_3x3_S1_reduce': tf.Variable(tf.random_normal([128])),
    'inception_3x3_S1': tf.Variable(tf.random_normal([256])),
    'inception_5x5_S1_reduce': tf.Variable(tf.random_normal([24])),
    'inception_5x5_S1': tf.Variable(tf.random_normal([64])),
    'inception_MaxPool': tf.Variable(tf.random_normal([64]))
}

conv_W_4d = {
    'inception_1x1_S1': tf.Variable(tf.random_normal([1,1,512,112])),
    'inception_3x3_S1_reduce': tf.Variable(tf.random_normal([1,1,512,144])),
    'inception_3x3_S1': tf.Variable(tf.random_normal([1,1,144,288])),
    'inception_5x5_S1_reduce': tf.Variable(tf.random_normal([1,1,512,32])),
    'inception_5x5_S1': tf.Variable(tf.random_normal([5,5,32,64])),
    'inception_MaxPool': tf.Variable(tf.random_normal([1,1,512,64]))

}

conv_B_4d = {
    'inception_1x1_S1': tf.Variable(tf.random_normal([112])),
    'inception_3x3_S1_reduce': tf.Variable(tf.random_normal([144])),
    'inception_3x3_S1': tf.Variable(tf.random_normal([288])),
    'inception_5x5_S1_reduce': tf.Variable(tf.random_normal([32])),
    'inception_5x5_S1': tf.Variable(tf.random_normal([64])),
    'inception_MaxPool': tf.Variable(tf.random_normal([64]))
}

conv_W_4e = {
    'inception_1x1_S1': tf.Variable(tf.random_normal([1,1,528,256])),
    'inception_3x3_S1_reduce': tf.Variable(tf.random_normal([1,1,528,160])),
    'inception_3x3_S1': tf.Variable(tf.random_normal([1,1,160,320])),
    'inception_5x5_S1_reduce': tf.Variable(tf.random_normal([1,1,528,32])),
    'inception_5x5_S1': tf.Variable(tf.random_normal([5,5,32,128])),
    'inception_MaxPool': tf.Variable(tf.random_normal([1,1,528,128]))

}

conv_B_4e = {
    'inception_1x1_S1': tf.Variable(tf.random_normal([256])),
    'inception_3x3_S1_reduce': tf.Variable(tf.random_normal([160])),
    'inception_3x3_S1': tf.Variable(tf.random_normal([320])),
    'inception_5x5_S1_reduce': tf.Variable(tf.random_normal([32])),
    'inception_5x5_S1': tf.Variable(tf.random_normal([128])),
    'inception_MaxPool': tf.Variable(tf.random_normal([128]))
}

conv_W_5a = {
    'inception_1x1_S1': tf.Variable(tf.random_normal([1,1,832,256])),
    'inception_3x3_S1_reduce': tf.Variable(tf.random_normal([1,1,832,160])),
    'inception_3x3_S1': tf.Variable(tf.random_normal([1,1,160,320])),
    'inception_5x5_S1_reduce': tf.Variable(tf.random_normal([1,1,832,32])),
    'inception_5x5_S1': tf.Variable(tf.random_normal([5,5,32,128])),
    'inception_MaxPool': tf.Variable(tf.random_normal([1,1,832,128]))

}

conv_B_5a = {
    'inception_1x1_S1': tf.Variable(tf.random_normal([256])),
    'inception_3x3_S1_reduce': tf.Variable(tf.random_normal([160])),
    'inception_3x3_S1': tf.Variable(tf.random_normal([320])),
    'inception_5x5_S1_reduce': tf.Variable(tf.random_normal([32])),
    'inception_5x5_S1': tf.Variable(tf.random_normal([128])),
    'inception_MaxPool': tf.Variable(tf.random_normal([128]))
}

conv_W_5b = {
    'inception_1x1_S1': tf.Variable(tf.random_normal([1,1,832,384])),
    'inception_3x3_S1_reduce': tf.Variable(tf.random_normal([1,1,832,192])),
    'inception_3x3_S1': tf.Variable(tf.random_normal([1,1,192,384])),
    'inception_5x5_S1_reduce': tf.Variable(tf.random_normal([1,1,832,48])),
    'inception_5x5_S1': tf.Variable(tf.random_normal([5,5,48,128])),
    'inception_MaxPool': tf.Variable(tf.random_normal([1,1,832,128]))

}

conv_B_5b = {
    'inception_1x1_S1': tf.Variable(tf.random_normal([384])),
    'inception_3x3_S1_reduce': tf.Variable(tf.random_normal([192])),
    'inception_3x3_S1': tf.Variable(tf.random_normal([384])),
    'inception_5x5_S1_reduce': tf.Variable(tf.random_normal([48])),
    'inception_5x5_S1': tf.Variable(tf.random_normal([128])),
    'inception_MaxPool': tf.Variable(tf.random_normal([128]))
}

softmax0_W = {
    'Conv': tf.Variable(tf.random_normal([1,1,512,128])),
    'FC1' : tf.Variable(tf.random_normal([4*4*128,1024])),
    'FC2' : tf.Variable(tf.random_normal([1024,3]))
}

softmax0_b ={
    'Conv': tf.Variable(tf.random_normal([128])),
    'FC1' : tf.Variable(tf.random_normal([1024])),
    'FC2' : tf.Variable(tf.random_normal([3]))
}

softmax1_W = {
    'Conv': tf.Variable(tf.random_normal([1,1,528,128]),name='Stest'),
    'FC1' : tf.Variable(tf.random_normal([4*4*128,1024])),
    'FC2' : tf.Variable(tf.random_normal([1024,3]))
}

softmax1_b ={
    'Conv': tf.Variable(tf.random_normal([128])),
    'FC1' : tf.Variable(tf.random_normal([1024])),
    'FC2' : tf.Variable(tf.random_normal([3]))
}   


def GoogLeNet_C1(x, weights, bias, conv_W_3a, conv_B_3a, conv_W_3b,conv_B_3b,conv_W_4a,conv_B_4a,conv_W_4b,
                                      conv_B_4b,conv_W_4c,conv_B_4c,conv_W_4d,conv_B_4d,conv_W_4e,conv_B_4e,conv_W_5a,conv_B_5a,
                                      conv_W_5b,conv_B_5b, softmax0_W ,softmax0_b, softmax1_W ,softmax1_b, dropout=0.8):
    x = tf.reshape(x, shape=[-1,224,224,1])
    
    # layer 1 
    x = tf.nn.conv2d(x, weights['conv1_7x7_S2'], strides= [1,2,2,1], padding='SAME')
    x = tf.nn.bias_add(x, bias['conv1_7x7_S2'])
    x = tf.nn.relu(x)
    x = tf.nn.max_pool(x, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')
    x = tf.nn.local_response_normalization(x, depth_radius=5/2.0, bias=2.0, alpha=1e-4, beta= 0.75)
    
    # layer 2
    x = tf.nn.conv2d(x, weights['conv2_1x1_S1'], strides=[1,1,1,1], padding='SAME')
    x = tf.nn.bias_add(x, biases['conv2_1x1_S1'])
    x = tf.nn.relu(x)
    x = tf.nn.conv2d(x, weights['conv2_3x3_S1'], strides=[1,1,1,1], padding='SAME')
    x = tf.nn.bias_add(x, biases['conv2_3x3_S1'])
    x = tf.nn.relu(x)
    x = tf.nn.local_response_normalization(x, depth_radius=5/2.0, bias=2.0, alpha=1e-4, beta= 0.75)
    x = tf.nn.max_pool(x, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')
    
    # layer 3
    inception_3a = inception_unit(inception_in=x, weights=conv_W_3a, biases=conv_B_3a)
    inception_3b = inception_unit(inception_3a, weights=conv_W_3b, biases=conv_B_3b)
    
    # 池化层
    x = inception_3b
    x = tf.nn.max_pool(x, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME' )

    # inception 4
    inception_4a = inception_unit(inception_in=x, weights=conv_W_4a, biases=conv_B_4a)
    # 引出第一条分支
    softmax0 = inception_4a
    inception_4b = inception_unit(inception_4a, weights=conv_W_4b, biases=conv_B_4b)    
    inception_4c = inception_unit(inception_4b, weights=conv_W_4c, biases=conv_B_4c)
    inception_4d = inception_unit(inception_4c, weights=conv_W_4d, biases=conv_B_4d)
    # 引出第二条分支
    softmax1 = inception_4d
    inception_4e = inception_unit(inception_4d, weights=conv_W_4e, biases=conv_B_4e)

    # 池化
    x = inception_4e
    x = tf.nn.max_pool(x, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME' )

    # inception 5
    x = inception_unit(x, weights=conv_W_5a, biases=conv_B_5a)
    inception_5b = inception_unit(x, weights=conv_W_5a, biases=conv_B_5a)
    #print (inception_5b.get_shape().as_list())
    
    
    # softmax 0
    softmax0 = tf.nn.avg_pool(softmax0, ksize=[1,5,5,1], strides=[1,3,3,1], padding='VALID')
    softmax1 = tf.nn.avg_pool(softmax1, ksize=[1,5,5,1], strides=[1,3,3,1], padding='VALID')
    softmax2 = tf.nn.avg_pool(inception_5b, ksize=[1,7,7,1], strides=[1,1,1,1], padding='SAME')
    #print (softmax2.get_shape().as_list())
    
    
    softmax0 = tf.nn.conv2d(softmax0, softmax0_W['Conv'], strides=[1,1,1,1], padding='SAME')
    softmax0 = tf.nn.bias_add(softmax0, softmax0_b['Conv'])
    softmax0 = tf.nn.relu(softmax0)
    
    softmax0 = tf.reshape(softmax0, [-1,softmax0_W['FC1'].get_shape().as_list()[0]])
    softmax0 = tf.add(tf.matmul(softmax0, softmax0_W['FC1']), softmax0_b['FC1'])
    softmax0 = tf.nn.relu(softmax0)
    
    softmax0 = tf.nn.dropout(softmax0,keep_prob= 0.3)
    
    softmax0 = tf.reshape(softmax0, [-1,softmax0_W['FC2'].get_shape().as_list()[0]])
    softmax0 = tf.add(tf.matmul(softmax0, softmax0_W['FC2']), softmax0_b['FC2'])
    
    
    # softmax1
    softmax1 = tf.nn.conv2d(softmax1, softmax1_W['Conv'], strides=[1,1,1,1], padding='SAME')
    softmax1 = tf.nn.bias_add(softmax1, softmax1_b['Conv'])
    softmax1 = tf.nn.relu(softmax1)
    
    softmax1 = tf.reshape(softmax1, [-1,softmax1_W['FC1'].get_shape().as_list()[0]])
    softmax1 = tf.add(tf.matmul(softmax1, softmax1_W['FC1']), softmax1_b['FC1'])
    softmax1 = tf.nn.relu(softmax1)
    
    softmax1 = tf.nn.dropout(softmax1,keep_prob= 0.3)
    
    softmax1 = tf.reshape(softmax1, [-1,softmax1_W['FC2'].get_shape().as_list()[0]])
    softmax1 = tf.add(tf.matmul(softmax1, softmax1_W['FC2']), softmax1_b['FC2'])
    
    # softmax 2
    softmax2 = tf.nn.dropout(softmax2, keep_prob = dropout)
    softmax2 = tf.reshape(softmax2, [-1,weights['FC2'].get_shape().as_list()[0]])
    softmax2 = tf.nn.bias_add(tf.matmul(softmax2,weights['FC2']),biases['FC2'])
    
    return softmax0, softmax1, softmax2


# ### 设置参数并训练

# In[6]:


# Parameters
learning_rate = 0.001 # 学习速率，
training_iters = 36 # 训练次数
batch_size = 32 # 每次计算数量 批次大小
display_step = 10 # 显示步长

dropout = 0.8 # Dropout, probability to keep units

# tensorflow 图 Graph 输入 input，这里的占位符均为输入
x = tf.placeholder(tf.float32,[None, 224*224])
y = tf.placeholder(tf.float32, [None,3])
keep_prob = tf.placeholder(tf.float32)

Predict0,Predict1,Predict2 = GoogLeNet_C1(x,weights,biases,conv_W_3a, conv_B_3a,conv_W_3b,conv_B_3b,conv_W_4a,conv_B_4a,conv_W_4b,
                                      conv_B_4b,conv_W_4c,conv_B_4c,conv_W_4d,conv_B_4d,conv_W_4e,conv_B_4e,conv_W_5a,conv_B_5a,
                                      conv_W_5b,conv_B_5b,softmax0_W,softmax0_b,softmax1_W ,softmax1_b,dropout=keep_prob)

cost = tf.reduce_mean(0.3*tf.nn.softmax_cross_entropy_with_logits(Predict0,y) + 
                      0.3*tf.nn.softmax_cross_entropy_with_logits(Predict1,y) +
                      tf.nn.softmax_cross_entropy_with_logits(Predict2,y)
                     )

params = tf.trainable_variables()
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
gvs = optimizer.compute_gradients(cost, params)
ysss = []
for i, (grid,var) in enumerate(gvs):
    if grid != None:
        grid = tf.clip_by_value(grid,-1.,1.)
        gvs[i] = (grid,var)
optimizer = optimizer.apply_gradients(gvs)
correct_pred = tf.equal(tf.arg_max(Predict2,1), tf.arg_max(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
PredictTest = tf.nn.softmax(Predict2)
init = tf.initialize_all_variables()


# In[7]:


get_ipython().run_cell_magic('time', '', 'with tf.Session() as sess:\n    sess.run(init)\n    for tr in range(training_iters):\n        for i in range(int(len(fac)/batch_size)):\n            batch_x = fac[i*batch_size:(i+1)*batch_size]\n            batch_y = ret[i*batch_size:(i+1)*batch_size]\n            sess.run(optimizer,feed_dict={x:batch_x,y:batch_y,keep_prob:dropout})\n        loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,y: batch_y, keep_prob:1.})\n        print("Iter " + str(tr*batch_size) + ", Minibatch Loss= " + \\\n              "{:.26f}".format(loss) + ", Training Accuracy= " + \\\n              "{:.26f}".format(acc))\n    print("Optimization Finished!") \n    tmp_out = sess.run(PredictTest, feed_dict={x:fac[-60:], keep_prob:1.})    \n    sess.close()')


# In[33]:


ptmp = Dtmp[-60:]
fig = plt.figure(figsize=(14,6))
plt.plot(ptmp.actual_future_rate_of_return,'gray')
for r,i in enumerate(ptmp.index):
    if tmp_out[r][0] ==1.:
        plt.scatter(i,ptmp.ix[i]['actual_future_rate_of_return'] ,color='g') 
    elif tmp_out[r][1] ==1.:
        plt.scatter(i,ptmp.ix[i]['actual_future_rate_of_return'] ,color='r')
plt.xlim(140,200)
plt.show()       


# ### 回测，我不太熟悉RiceQuant 回测平台，回测有空再写
