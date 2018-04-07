
# coding: utf-8

# # `导读`
# ***
# ### 本帖主要介绍了因子IC的三种计算方法，并提出了基于因子IC的因子权重优化模型，以策略的方式对比了因子权重优化前后的结果,最后简单介绍了协方差阵优化的方法。

# # **一.前言**
# - ### 多因子模型是一种常用的选股模型，其构建方法一般分为回归法和打分法两类。
# - ### 打分法是指选用若干能够对股票收益产生预测作用的因子，之后根据股票的每个因子值在截面上的相对位置给出股票在该因子上的得分，然后按照一定的权重将每个股票的各个因子得分相加从而得到该股票的最终得分并按照该得分对股票进行排序，筛选，构造投资组合。
# - ### 在打分法的模型构建中，各因子上的权重比例设定非常关键，若干好的因子在不好的权重配置下很可能会有不好的结果。

# In[ ]:

#加载要用的包
import numpy as np
import pandas as pd
import scipy.stats as st
import numpy.linalg as nlg


# - ### 举个例子来说明确定因子权重的重要性：
# 	
#     ##### 下面是四只股票在两类因子上的当期得分和下期收益率：

# In[ ]:

lizi = pd.DataFrame()
lizi['因子A'] = [9,3,8,5]
lizi['因子B'] = [6,7,9,2]
lizi['下期收益率'] = [0.01,0.009,0.008,0.006]
lizi.index = ['股票a','股票b','股票c','股票d']
r1,_ = st.pearsonr(lizi['因子A'],lizi['下期收益率'])
r2,_ = st.pearsonr(lizi['因子B'],lizi['下期收益率'])
print lizi.to_html()
print '因子A得分与下期收益率的相关系数为：'+str(r1)
print '因子B得分与下期收益率的相关系数为：'+str(r2)


# - ##### 若按分组收益来看，因子A得分较高的两只股票(a,c)相对得分较低的两只股票(b,d)有0.003的超额收益，因子B得分较高的b，c相对得分较低的a，d有0.001的超额收益。A与B均有很好的收益预测能力，然而若是以因子A，B得分的平均分作为股票的最终得分，则最终得分最高的是股票c，收益为0.008，低于四只股票的平均收益0.00825
# - ### 下面我们介绍一种基于因子IC的因子模型：
# 
# # **二.因子IC及优化模型**
# ## **1.因子IC**
# 
# - ### 我们首先要介绍一下因子IC(Information Coefficient)是什么.
# - ### 定义可以参考：[Information Coefficient](http://www.investopedia.com/terms/i/information-coefficient.asp#axzz22Qau39ZX)
# - ### 简单来说，因子在某一期的IC指的是该期因子对股票的下期收益的预测值和股票下期的实际回报值在横截面上的相关系数，即：
# $$IC_A = correlation(\vec{f_A},\vec{r})$$
# 	其中，$IC_A$代表因子A在该期的IC值，$\vec{f_A}$为该期因子A对股票下期收益率的预测向量，$\vec{r}$为股票下期实际收益率向量
# - ### 那么，问题就来了，因子对股票下期收益率的预测是怎么算的呢？最先在脑海里浮现的就是用回归的方法来做预测嘛，不过这个预测选取的模型不一样，岂不是结果就不一样了。然后我又去找了下资料，发现有一种叫做原始IC或者传统IC的计算方法：
# 
# # **`传统IC`**
# 
# - ### 直接用该期因子的值，和股票下期收益率算相关系数，就是因子IC值，即把上面的$\vec{f_A}$换为该期因子值向量
# - ### 这样也很好理解，直接反映了因子的预测能力.IC越高，就表明该因子在该期对股票收益的预测能力越强。
# - ### 下面举个例子算一下：计算沪深300成分股在20160809这一天因子'PE'的IC值

# In[ ]:

universe = set_universe('HS300','20160809')  #20160809这一天沪深300成分股secID
PEdata = DataAPI.MktStockFactorsOneDayGet(tradeDate='20160809',secID=universe,field='secID,PE').set_index('secID')   #20160809因子PE的值
Return = DataAPI.MktEqudAdjGet(tradeDate='20160810',secID=universe,field='secID,preClosePrice,closePrice').set_index('secID')  #20160810股票的数据
Return = pd.concat([PEdata,Return],axis=1)
Return['return'] = Return['closePrice']/Return['preClosePrice']-1     #20160810这一天股票的收益率向量
ic,_ = st.pearsonr(Return['return'],Return['PE'])
ic


# # `定义IC`
# - ### 我也按照定义的方式来计算一下IC的值，也就是用该期的股票收益率对上期因子值作回归，并用该期因子值预测下期股票收益率，取下期预测的收益率和下期实际收益率的相关系数
# - ### 还是用上面的例子来计算：

# In[ ]:

universe = set_universe('HS300','20160809')  #20160809这一天沪深300成分股secID
PEdata1 = DataAPI.MktStockFactorsOneDayGet(tradeDate='20160808',secID=universe,field='secID,PE').set_index('secID')   #20160808因子PE的值
Return1 = DataAPI.MktEqudAdjGet(tradeDate='20160809',secID=universe,field='secID,preClosePrice,closePrice').set_index('secID')  #20160809股票的数据
Return1 = pd.concat([PEdata1,Return1],axis=1)
Return1['return'] = Return1['closePrice']/Return1['preClosePrice']-1     #20160809这一天股票的收益率向量

#20160809股票的收益率向量对20160808因子PE的值作回归：
from sklearn import linear_model
x=np.zeros((1,len(Return1)))
x[0]=PEdata1['PE']
x=np.mat(x).T
y=Return1['return']
y=np.mat(y).T
regr = linear_model.LinearRegression()
regr.fit(x,y)

#用20160809因子PE的值来预测20160810的股票收益率
PEdata2 = DataAPI.MktStockFactorsOneDayGet(tradeDate='20160809',secID=universe,field='secID,PE').set_index('secID')   #20160809因子PE的值
X=np.zeros((1,len(Return1)))
X[0]=PEdata2['PE']
Y=regr.predict(np.mat(X).T)

#取20160810的股票收益率预测值和20160810的股票收益率实际值的相关系数作为因子PE在20160809的IC值
Return2 = DataAPI.MktEqudAdjGet(tradeDate='20160810',secID=universe,field='secID,preClosePrice,closePrice').set_index('secID')  #20160810股票的数据
Return2['return'] = Return2['closePrice']/Return2['preClosePrice']-1     #20160810这一天股票的收益率向量
ic,_ = st.pearsonr(Return2['return'],Y.reshape(Y.shape[0],))
ic


# # `研报IC`
# - ### 下面是研报里对因子IC的计算方法：
# > ### 1.对因子进行标准化，去极值处理(横截面上)；
# > ### 2.调整行业和市值影响：因子在大小盘，不同行业的股票上的值会有明显的差异，比如市盈率因子较低的股票会集中于银行等板块.调整的方法就是，把每个因子对市值和行业哑变量做回归，取残差作为因子的一个替代；（参见行业中性化）
# > ### 3.残差正交化调整：因子间存在较强同质性时，使用施密特正交化方法对上一步得到的残差做正交化处理，用得到的正交化残差作为因子对股票收益率的一个预测。
# - ### 前两步在优矿上可以通过standardize，neutralize，winsorize方便地实现，具体请参考魏老师的大作[破解Alpha对冲策略](https://uqer.io/community/share/55ff6ce9f9f06c597265ef04)
# - ### 我们只需要在有多个因子时，多作第三步即可。
# - ### 不妨先看看一个因子时的IC值，还是用上面的例子，计算沪深300成分股在20160809这一天因子'PE'的IC值：

# In[ ]:

universe = set_universe('HS300','20160809')  #20160809这一天沪深300成分股secID
PEdata = DataAPI.MktStockFactorsOneDayGet(tradeDate='20160809',secID=universe,field='ticker,PE').set_index('ticker')   #20160809因子PE的值
signal_PE = standardize(neutralize(winsorize(PEdata['PE'].dropna().to_dict()),'20160809'))   #去极值，标准化，中性化
PEdata['PE'][signal_PE.keys()] = signal_PE.values()
Return = DataAPI.MktEqudAdjGet(tradeDate='20160810',secID=universe,field='ticker,preClosePrice,closePrice').set_index('ticker')  #20160810股票的数据
Return = pd.concat([PEdata,Return],axis=1)
Return['return'] = Return['closePrice']/Return['preClosePrice']-1     #20160810这一天股票的收益率向量
ic,_ = st.pearsonr(Return['return'],Return['PE'])
ic


# - ### 可以多加入一个变量，看看正交化之后的结果，比如计算沪深300成分股在20160809这一天因子'PE'和因子'PB'的IC值
# - ### 关于施密特正交化的内容，不作过多介绍

# In[ ]:

#施密特正交化函数，输入n个向量的dataframe，输出n个向量的dataframe
def Schmidt(data):
    output = pd.DataFrame()
    mat = np.mat(data)
    output[0] = np.array(mat[:,0].reshape(len(data),))[0]
    for i in range(1,data.shape[1]):
        tmp = np.zeros(len(data))
        for j in range(i):
            up = np.array((mat[:,i].reshape(1,len(data)))*(np.mat(output[j]).reshape(len(data),1)))[0][0]
            down = np.array((np.mat(output[j]).reshape(1,len(data)))*(np.mat(output[j]).reshape(len(data),1)))[0][0]
            tmp = tmp+up*1.0/down*(np.array(output[j]))
        output[i] = np.array(mat[:,i].reshape(len(data),))[0]-np.array(tmp)
    output.index = data.index
    output.columns = data.columns
    return output


# In[ ]:

universe = set_universe('HS300','20160809')  #20160809这一天沪深300成分股secID
PEdata = DataAPI.MktStockFactorsOneDayGet(tradeDate='20160809',secID=universe,field='ticker,PE,PB').set_index('ticker')   #20160809因子PE的值
signal_PE = standardize(neutralize(winsorize(PEdata['PE'].dropna().to_dict()),'20160809'))   #去极值，标准化，中性化
signal_PB = standardize(neutralize(winsorize(PEdata['PB'].dropna().to_dict()),'20160809'))   #去极值，标准化，中性化

PEdata['PE'][signal_PE.keys()] = signal_PE.values()
PEdata['PB'][signal_PB.keys()] = signal_PB.values()
PEdata = Schmidt(PEdata)                                  #施密特正交化

Return = DataAPI.MktEqudAdjGet(tradeDate='20160810',secID=universe,field='ticker,preClosePrice,closePrice').set_index('ticker')  #20160810股票的数据
Return = pd.concat([PEdata,Return],axis=1)
Return['return'] = Return['closePrice']*1.0/Return['preClosePrice']-1     #20160810这一天股票的收益率向量
icPE,_ = st.pearsonr(Return['return'],Return['PE'])
icPB,_ = st.pearsonr(Return['return'],Return['PB'])
print '因子PE的IC值为：'+str(icPE)
print '因子PB的IC值为：'+str(icPB)


# - ### 以上给出了三种方法来计算因子IC，我们可以根据自己使用因子的方法来选取或设计可行的办法来计算因子IC。
# - ### 本帖主要重于讲述方法和过程，不妨以第三种IC的计算方法为例进行下面的阐述：

# ## **2.优化模型**
# - ### 因子的IR(信息比)值为因子IC的均值和因子IC的标准差的比值，IR值越高越好
# - ### 假设有M个因子，其IC的均值向量为$\vec{IC} = (\overline{IC_1},\overline{IC_2},\cdots,\overline{IC_M})^T$,IC的协方差矩阵为：$\Sigma$.如果各因子权重向量为$\vec{v}=(\overline{V_1},\overline{V_2},\cdots,\overline{V_M})^T$,则复合因子的IR值为：
# $$IR = \frac{v^T*\vec{IC}}{\sqrt{v^T* \Sigma*v}}$$
# - ### 我们的目标就是要最大化复合因子的IR，一方面，我们可以通过python的优化函数求解，另一方面，我们可以直接求出一个解析解：
# - ### 求导以后，可以得到以上最值问题的解,因子的`最优权重向量`为：
# $$v^* = \delta *\Sigma^{-1}*\vec{IC}$$
# - #### 其中$\delta$为任意正数，可以用来调整最后的权重之和为1

# - ### 同样地，我们举例子来计算一下，考虑到基本面因子其实变化并没有那么快，而技术类因子周期又不能太长，不妨以周度数据为研究对象：

# In[ ]:

# 构建日期列表，取每周最后一个交易日
data=DataAPI.TradeCalGet(exchangeCD=u"XSHG",beginDate=u"20140101",endDate='20160901',field=['calendarDate','isWeekEnd','isMonthEnd'],pandas="1")
data = data[data['isWeekEnd'] == 1]
date_list = map(lambda x: x[0:4]+x[5:7]+x[8:10], data['calendarDate'].values.tolist())


# In[ ]:

print date_list


# - ### 下面需要确定用哪几个因子，我顺手选取了`'PE','ROE','RSI','NetProfitGrowRate'`这四个因子
# - ### 为了避免数据量过大，仍以沪深300成分股为研究对象

# In[ ]:

factor_names = ['PE','ROE','RSI','NetProfitGrowRate']


# In[ ]:

#定义一个计算因子IC的函数，输入当期日期(注意必须是每周最后一个交易日)，就可以得到因子IC值的dataframe，列名为四个因子
def get_currentIC(currentdate):
    factor_api_field = ['ticker'] + factor_names
    nextdate = date_list[date_list.index(currentdate)+1]   #要取下期的收益率序列
    factordata = DataAPI.MktStockFactorsOneDayGet(tradeDate=currentdate,secID=set_universe('HS300',currentdate),field=factor_api_field).set_index('ticker') #获取因子数据
    for i in range(len(factor_names)):
        signal = standardize(neutralize(winsorize(factordata[factor_names[i]].dropna().to_dict()),currentdate)) #去极值，标准化，中性化
        factordata[factor_names[i]][signal.keys()] = signal.values()
    factordata = factordata.dropna()
    factordata = Schmidt(factordata)                              #施密特正交化
    Return = DataAPI.MktEquwAdjGet(secID=set_universe('HS300',currentdate),beginDate=nextdate,endDate=nextdate,field=u"ticker,return").set_index('ticker')
    Return = pd.concat([factordata,Return],axis=1).dropna()
    IC = pd.DataFrame()
    for i in range(len(factor_names)):
        ic, p_value = st.pearsonr(Return[factor_names[i]],Return["return"])
        IC[factor_names[i]] = np.array([ic])                             #计算IC值，存入dataframe里
    return IC


# - ### 试一下这个函数是否work

# In[ ]:

get_currentIC('20160812')


# - ### 这里需要注意IC的定义以避免未来函数，当期的IC的计算其实是用到了下期的收益数据，所以使用的时候，其实是要再往前推一期才不会用到未来数据.
# - ### 下面按照前面的优化模型来计算当期因子最优的权重配比，这里又会出现一个参数，就是选取前几期的因子IC值比较准确.我暂时选取前8期的值来计算，感兴趣的可以自己改这个参数进行计算,这里已经考虑了避免未来函数的因素，也就是说，get_bestweight函数得到的是根据当前日期及之前的数据得到的最优的权重
# - ### 对于因子权重而言，就不考虑权重大于0以及和为1了，因为有些因子的作用是反向的，权重是负的也是有道理的

# In[ ]:

N = 8 #取前几期的因子IC值来计算当前的最优权重
def get_bestweight(currentdate):   #传入当前日期，得到当前日期及之前8期的数据所得到的最优权重
    date = [date_list[date_list.index(currentdate)-i-1] for i in range(N)]  #取前8期日期
    IC = pd.DataFrame()
    for i in range(N):
        ic = get_currentIC(date[i])    #计算每个日期的IC值
        IC = pd.concat([IC,ic],axis=0)
    mat = np.mat(IC.cov())                     #按照公式计算最优权重
    mat = nlg.inv(mat)
    weight = mat*np.mat(IC.mean()).reshape(len(mat),1)
    weight = np.array(weight.reshape(len(weight),))[0]
    return weight                          #返回最优权重值


# In[ ]:

#举个例子算一下
get_bestweight('20160812')


# - ### 为了写策略方便，我们不妨把研究时间内的权重都算出来，以后直接在这个dataframe里调用即可

# In[ ]:

Weight = pd.DataFrame()    #用于存权重序列，index为日期
for i in range(30,136):       
    weight = get_bestweight(date_list[i])    #计算权重
    tmp = pd.DataFrame(index=[date_list[i]],columns=factor_names, data=0) 
    for j in range(4):
        tmp.ix[date_list[i],j] = weight[j]
    Weight = pd.concat([Weight,tmp],axis=0)    #存在dataframe里


# In[ ]:

pd.concat([Weight.head(10),Weight.tail(10)],keys = ['前十行','后十行'],axis=0)   #来看看我们的结果


# - ### 下面我们试一下基于因子IC的多因子模型是否work，可能我选取的因子和处理方法还有很多细节值得优化，所以结果不见得太好
# 
# # **三.基于因子IC的四因子模型**
# - ### 研究时间范围：20160128-20160815
# - ### 股票仍然是沪深３００成分股
# - ### 每周第一个交易日调仓
# 
# ## **1.因子等权重求和**
# 
# - ### 选股逻辑就是按照上一个周最后一个交易日的四个因子数据，按照等权重对因子值加权求和，选出得分最高的３０只股票

# In[ ]:

# 导入包
from CAL.PyCAL import *
import numpy as np
import pandas as pd


start = '2016-01-28'                       # 回测起始时间
end = '2016-08-15'     # 回测结束时间
universe = set_universe('HS300')    # 股票池
benchmark = 'HS300'                       # 策略参考标准
capital_base = 10000000                     # 起始资金
freq = 'd'                              # 策略类型，'d'表示日间策略使用日线回测，'m'表示日内策略使用分钟线回测
refresh_rate = 1                          # 调仓频率

# 日期处理相关
cal = Calendar('China.SSE')
period = Period('-1B')
factor_api_field = ['secID','ticker'] + factor_names

def initialize(account):                   # 初始化虚拟账户状态
    pass

def handle_data(account):                  # 每个交易日的买入卖出指令
    
    today = account.current_date
    today = Date.fromDateTime(account.current_date)  # 向前移动一个工作日
    yesterday = cal.advanceDate(today, period)
    yesterday = yesterday.toDateTime().strftime('%Y%m%d')
    if yesterday in date_list:
        
        factordata = DataAPI.MktStockFactorsOneDayGet(tradeDate=yesterday,secID=set_universe('HS300',yesterday),field=factor_api_field).set_index('ticker') #获取因子数据
        for i in range(len(factor_names)):
            signal = standardize(neutralize(winsorize(factordata[factor_names[i]].dropna().to_dict()),yesterday)) #去极值，标准化，中性化
            factordata[factor_names[i]][signal.keys()] = signal.values()
        factordata = factordata.dropna()
        factordata = factordata.set_index('secID')
        factordata = Schmidt(factordata)        #施密特正交化
        factordata['total_score'] = np.dot(factordata, np.array([0.25 for i in range(4)]))    #因子值等权求和
        factordata.sort(['total_score'],ascending=False,inplace=True)                  #排序
        factordata = factordata[:30]
        
        # 先卖出
        sell_list = account.security_position
        for stk in sell_list:
            order_to(stk, 0)

        # 再买入
        buy_list = list(set(factordata.index).intersection(set(account.universe)))
        total_money = account.reference_portfolio_value
        prices = account.reference_price 
        for stk in buy_list:
            if np.isnan(prices[stk]) or prices[stk] == 0:  # 停牌或是还没有上市等原因不能交易
                continue
            order(stk, int(total_money / len(buy_list) / prices[stk] /100)*100)
    else:
        return


# ## **2.因子权重优化**
# - ### 选股逻辑就是按照上一个周最后一个交易日的四个因子数据，按照我们计算出的上个周最后一个交易日的因子权重加权求和，选出得分最高的３０只股票

# In[ ]:

# 导入包
from CAL.PyCAL import *
import numpy as np
import pandas as pd


start = '2016-01-28'                       # 回测起始时间
end = '2016-08-15'     # 回测结束时间
universe = set_universe('HS300')    # 股票池
benchmark = 'HS300'                       # 策略参考标准
capital_base = 10000000                     # 起始资金
freq = 'd'                              # 策略类型，'d'表示日间策略使用日线回测，'m'表示日内策略使用分钟线回测
refresh_rate = 1                          # 调仓频率

# 日期处理相关
cal = Calendar('China.SSE')
period = Period('-1B')
factor_api_field = ['secID','ticker'] + factor_names

def initialize(account):                   # 初始化虚拟账户状态
    pass

def handle_data(account):                  # 每个交易日的买入卖出指令
    
    today = account.current_date
    today = Date.fromDateTime(account.current_date)  # 向前移动一个工作日
    yesterday = cal.advanceDate(today, period)
    yesterday = yesterday.toDateTime().strftime('%Y%m%d')
    if yesterday in date_list:
        
        factordata = DataAPI.MktStockFactorsOneDayGet(tradeDate=yesterday,secID=set_universe('HS300',yesterday),field=factor_api_field).set_index('ticker') #获取因子数据
        for i in range(len(factor_names)):
            signal = standardize(neutralize(winsorize(factordata[factor_names[i]].dropna().to_dict()),yesterday)) #去极值，标准化，中性化
            factordata[factor_names[i]][signal.keys()] = signal.values()
        factordata = factordata.dropna()
        factordata = factordata.set_index('secID')
        factordata = Schmidt(factordata)               #施密特正交化
        factordata['total_score'] = np.dot(factordata, np.array(Weight.ix[yesterday,:]))         #按照优化后的权重求和
        factordata.sort(['total_score'],ascending=False,inplace=True)                     #排序
        factordata = factordata[:30]
        
        # 先卖出
        sell_list = account.security_position
        for stk in sell_list:
            order_to(stk, 0)

        # 再买入
        buy_list = list(set(factordata.index).intersection(set(account.universe)))
        total_money = account.reference_portfolio_value
        prices = account.reference_price 
        for stk in buy_list:
            if np.isnan(prices[stk]) or prices[stk] == 0:  # 停牌或是还没有上市等原因不能交易
                continue
            order(stk, int(total_money / len(buy_list) / prices[stk] /100)*100)
    else:
        return


# - ## 可以看到，因子权重优化后的结果比原来好得多，某种程度上，因子权重也体现因子的影响到底是正向的还是负向的。

# # **四.协方差阵的收缩估计**
# - ### 我们在计算因子间的协方差阵时，其实是使用的协方差阵的无偏估计--样本协方差阵$ \hat{\Sigma}$。那么问题就来了，按照目前的节奏，优矿上有几百个因子，我们的协方差阵就不一定可逆了。即便它是可逆的，样本协方差阵的逆矩阵也不是协方差阵逆矩阵的无偏估计了，如下：
# $$E(\hat{\Sigma}) = \frac{T}{T-M-2} \Sigma^{-1}$$
# - #### 其中M指因子个数，T指样本期，即取前几期的数据
# - ### 基于以上问题，有人提出了一种压缩估计得方法。它的基本思想使用一个方差小但偏差大的协方差矩阵估计量$\hat \Phi$作为目标估计量，和样本协方差矩阵做一个调和，牺牲部分偏差来获得更稳健的估计量:
# $$\hat \Sigma_{shrink} = \lambda \Phi + (1-\lambda)*\hat{\Sigma}$$
# - ### 参数$\lambda$通过最小化估计量的二次偏差得到，至于估计量$\hat \Phi$的选择上。有以下三种形式可供参考：
# 
# 	> ### 1) 单参数形式，可以表示为方差乘以一个单位矩阵
# 
# 	> ### 2) CAPM单因子结构化模型估计
# 
# 	> ### 3) 平均相关系数形式
# - ### 其中第二种形式只适用于股票
# - ### 该方法可运用于许多涉及需要估计协方差阵的地方，比如均值方差模型。关于这部分内容，这里不再赘述，有机会发一贴专门讲这个问题
# 
# ***
# 参考文献：《安信证券－多因子系列报告之一：基于因子IC的多因子模型》

# In[ ]:



