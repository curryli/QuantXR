
# coding: utf-8

# # HMM在股票市场中的应用 #

# 我们假设隐藏状态数量是6，即假设股市的状态有6种，虽然我们并不知道每种状态到底是什么，但是通过后面的图我们可以看出那种状态下市场是上涨的，哪种是震荡的，哪种是下跌的。可观测的特征状态我们选择了3个指标进行标示，进行预测的时候假设假设所有的特征向量的状态服从高斯分布，这样就可以使用<font color=#0099ff> hmmlearn </font>这个包中的<font color=#0099ff> GaussianHMM </font>进行预测了。下面我会逐步解释。

# 首先导入必要的包：

# In[1]:


from hmmlearn.hmm import GaussianHMM
import numpy as np
from matplotlib import cm, pyplot as plt
import matplotlib.dates as dates
import pandas as pd
import datetime


# 测试时间从2005年1月1日到2015年12月31日，拿到每日沪深300的各种交易数据。

# In[2]:


beginDate = '2005-01-01'
endDate = '2015-12-31'
n = 6 #6个隐藏状态
data = get_price('CSI300.INDX',start_date=beginDate, end_date=endDate,frequency='1d')
data.to_csv("sorted_1000.csv", sep=' ')
data[0:9]


# 拿到每日成交量和收盘价的数据。

# In[3]:


volume = data['TotalVolumeTraded']
close = data['ClosingPx']


# 计算每日最高最低价格的对数差值，作为特征状态的一个指标。

# In[4]:


logDel = np.log(np.array(data['HighPx'])) - np.log(np.array(data['LowPx']))
logDel


# 计算每5日的指数对数收益差，作为特征状态的一个指标。

# In[5]:


logRet_1 = np.array(np.diff(np.log(close)))#这个作为后面计算收益使用
logRet_5 = np.log(np.array(close[5:])) - np.log(np.array(close[:-5]))
logRet_5


# 计算每5日的指数成交量的对数差，作为特征状态的一个指标。

# In[6]:


logVol_5 = np.log(np.array(volume[5:])) - np.log(np.array(volume[:-5]))
logVol_5


# 由于计算中出现了以5天为单位的计算，所以要调整特征指标的长度。

# In[7]:


logDel = logDel[5:]
logRet_1 = logRet_1[4:]
close = close[5:]
Date = pd.to_datetime(data.index[5:])


# 把我们的特征状态合并在一起。

# In[8]:


A = np.column_stack([logDel,logRet_5,logVol_5])
A


# 下面运用<font color=#0099ff> hmmlearn </font>这个包中的<font color=#0099ff> GaussianHMM </font>进行预测。

# In[9]:


model = GaussianHMM(n_components= n, covariance_type="full", n_iter=2000).fit([A])
hidden_states = model.predict(A)
hidden_states


# 关于<font color=#0099ff> covariance_type </font>的参数有下面四种：

# spherical：是指在每个马尔可夫隐含状态下，可观察态向量的所有特性分量使用相同的方差值。对应协方差矩阵的非对角为0，对角值相等，即球面特性。这是最简单的高斯分布PDF。
# ![image](https://pic1.zhimg.com/dc016d706d98137f95a3d26d82def9bc_b.jpg)
# diag：是指在每个马尔可夫隐含状态下，可观察态向量使用对角协方差矩阵。对应协方差矩阵非对角为0，对角值不相等。diag是hmmlearn里面的默认类型。
# ![image](https://pic2.zhimg.com/d5416d8179b5a4f2ea67ab6c2afee0c1_b.jpg)
# full：是指在每个马尔可夫隐含状态下，可观察态向量使用完全协方差矩阵。对应的协方差矩阵里面的元素都是不为零。
# ![image](https://pic4.zhimg.com/0911027beaeae675a4219d70ce2b0bdb_b.jpg)
# tied：是指所有的马尔可夫隐含状态使用相同的完全协方差矩阵。
# 
# 这四种PDF类型里面，spherical, diag和full代表三种不同的高斯分布概率密度函数，而tied则可以看作是GaussianHMM和GMMHMM的特有实现。其中，full是最强大的，但是需要足够多的数据来做合理的参数估计；spherical是最简单的，通常用在数据不足或者硬件平台性能有限的情况之下；而diag则是这两者一个折中。在使用的时候，需要根据可观察态向量不同特性的相关性来选择合适的类型。
# 
# 转自知乎用户Aubrey Li

# 我们把每个预测的状态用不同颜色标注在指数曲线上看一下结果。

# In[10]:


plt.figure(figsize=(25, 18)) 
for i in range(model.n_components):
    pos = (hidden_states==i)
    plt.plot_date(Date[pos],close[pos],'o',label='hidden state %d'%i,lw=2)
    plt.legend(loc="left")


# 从图中可以比较明显的看出绿色的隐藏状态代表指数大幅上涨，浅蓝色和黄色的隐藏状态代表指数下跌。

# 为了更直观的表现不同的隐藏状态分别对应了什么，我们采取获得隐藏状态结果后第二天进行买入的操作，这样可以看出每种隐藏状态代表了什么。

# In[11]:


res = pd.DataFrame({'Date':Date,'logRet_1':logRet_1,'state':hidden_states}).set_index('Date')
plt.figure(figsize=(25, 18)) 
for i in range(model.n_components):
    pos = (hidden_states==i)
    pos = np.append(0,pos[:-1])#第二天进行买入操作
    df = res.logRet_1
    res['state_ret%s'%i] = df.multiply(pos)
    plt.plot_date(Date,np.exp(res['state_ret%s'%i].cumsum()),'-',label='hidden state %d'%i)
    plt.legend(loc="left")


# 可以看到，隐藏状态1是一个明显的大牛市阶段，隐藏状态0是一个缓慢上涨的阶段(可能对应反弹)，隐藏状态3和5可以分别对应震荡下跌的大幅下跌。其他的两个隐藏状态并不是很明确。由于股指期货可以做空，我们可以进行如下操作：当处于状态0和1时第二天做多，当处于状态3和5第二天做空，其余状态则不持有。

# In[12]:


long = (hidden_states==0) + (hidden_states == 1) #做多
short = (hidden_states==3) + (hidden_states == 5)  #做空
long = np.append(0,long[:-1]) #第二天才能操作
short = np.append(0,short[:-1]) #第二天才能操作


# 收益曲线图如下：

# In[13]:


res['ret'] =  df.multiply(long) - df.multiply(short)  
plt.plot_date(Date,np.exp(res['ret'].cumsum()),'r-')


# 可以看到效果还是很不错的。但事实上该结果是有些问题的。真实操作时，我们并没有未来的信息来训练模型。不过可以考虑用历史数据进行训练，再对之后的数据进行预测。
