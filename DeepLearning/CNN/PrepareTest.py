# coding: utf-8
import matplotlib.image as mpimg


import numpy as np
import pandas as pd
import TAF
import datetime
import matplotlib.pylab as plt
#import seaborn as sns


fe = 56 # 回溯日期

HS300_15m = pd.read_csv('HS300_15m.csv')
#print HS300_15m.head(5)

DF_taf = TAF.get_factors(HS300_15m.tradeTime,
                HS300_15m.open,
                HS300_15m.close,
                HS300_15m.high,
                HS300_15m.low,
                HS300_15m.volume,
                rolling = 16*fe,
                drop=False,
                normalization=True)

#print(DF_taf.head(5))

HS300_1d = pd.read_csv('HS300_1d.csv')
HS300_1d['actual_future_rate_of_return'] = HS300_1d.close.shift(-14)/HS300_1d.close - 1.0
#print HS300_1d.head(5)
HS300_1d = HS300_1d.dropna()
HS300_1d = HS300_1d[-200:]
HS300_1d['Direction_Label'] = 0
HS300_1d.actual_future_rate_of_return.describe()
HS300_1d.loc[HS300_1d.actual_future_rate_of_return>0.025,'Direction_Label'] = 1
HS300_1d.loc[HS300_1d.actual_future_rate_of_return<-0.01,'Direction_Label'] = -1
HS300_1d.reset_index(drop= True, inplace= True)
#print(HS300_1d.tail(5))

start = HS300_1d.tradeTime.values[0]
end = HS300_1d.tradeTime.values[-1]
print start, end
end = datetime.datetime.strptime(end,'%Y-%m-%d') # 将STR转换为datetime
end = end + datetime.timedelta(days=1) # 增加一天
end = end.strftime('%Y-%m-%d')

fac_list = DF_taf.columns
fac_HS300 = DF_taf.ix[(DF_taf.index.values>start) & (DF_taf.index.values<end)]#.reset_index(drop=True)
print(len(fac_HS300))

fac_size = len(fac_list)

tmp_HS300 = np.zeros((1,fe*16*fac_size))

for i in np.arange(fe,int(len(fac_HS300)/16)):
    tmp = fac_HS300.ix[16*(i-fe):16*i][fac_list]
    tmp = np.array(tmp).ravel(order='C').transpose()
    tmp_HS300 = np.vstack((tmp_HS300,tmp))
#tmp_HS300 = np.delete(tmp_HS300,0,axis=0)
print(tmp_HS300.shape)


# # - 多种技术分析因子数值在Y轴并列之后使用颜色表示因子数值大小
# import matplotlib.pyplot as plt
# import matplotlib.image as mping
# plt.figure(figsize=(6,6))
# shpig = tmp_HS300[1]
# shpig = shpig.reshape(224,232)
# # shpig +=4
# # shpig *=26
# plt.axis("off")
# plt.imshow(shpig)
# plt.show()

ret = HS300_1d.iloc[-145:,-1]
ret = np.array(ret)
wid = np.zeros((145,3))
wid[np.arange(145),ret] = 1
ret = wid
print(ret.shape)