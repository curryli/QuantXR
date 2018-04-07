
# coding: utf-8

# # 1. 引言
# ### 机器学习在量化中的应用点:
# 1、挖掘新因子，例如从非结构化的新闻、研报、公告等文本数据中获取情感、主题热度等量化指标。
# 2、对传统的基本面因子进行预测，例如[深度学习模型对基本面因子的预测应用研究](https://uqer.datayes.com/report/20180301)。利用高频数据对公司的基本面情况进行预测。
# 3、对因子进行合成，例如[循环神经网络在因子合成的应用初探](https://uqer.datayes.com/report/20180202)。传统的多因子模型如何加权，等权、IC、IC_IR加权。

# 深度学习在图像、语音方面获得了很大的成功，但在量化方面，目前看来还有些难度。对比下图象识别，图中有没有目标物体是确定的，构成目标的因素是确定的。且随着网络层数的增加，模型能学到更复杂的图形特征。但在金融数据中构成目标的因素是不确定的，不同时点的归因完全不一样。当然我们不求能完全解释，深度学习在图像识别能做到98%、99%，但在量化上胜率能到60%已经谢天谢地了。
# ![不确定](http://odqb0lggi.bkt.clouddn.com/548d8a7df9f06c45e7073e61/1e3875ac-33e7-11e8-a6f8-0242ac140002)

# ### 出发点
# 1、彼得林奇的PEG指标，两个基本因子的组合。考量了市场预期与实际情况是否相符，为什么能想到？。
# 2、按照传统有效因子要有可解释性，避免数据挖掘。但线性因子效用在不断降低，需要更深度的发现。
# 3、[神经网络可以拟合任意功能的函数](http://neuralnetworksanddeeplearning.com/chap4.html)，能否实现在高维空间中发现有效因子？下面我们介绍深度挖掘工具TensorFlow工具，借此来寻找圣杯。

# # 2. 使用TensorFlow构建神经网络
# TensorFlow字面上的意思是张量在流动，那什么是张量，张量又在哪流动呢?Google对TensorFlow的定位其实不仅仅是一个深度学习的库，而是一个基于数据流图的数值计算框架。在数据流图中节点表示操作，边表示在节点间相互联系的多为数组即张量，只要把这个数据流图画好，张量就可以在图上“流动”起来。
# 
# 下面分为两部分
# - 1、使用TensorFlow基础函数搭建神经网络模型
# - 2、使用TensorFlow高级API：Estimator搭建网络

# ## 2.1 使用TensorFlow底层函数构建网络
# ### 2.1.1 数据准备
# 我们将展示如何用TensorFlow画一个简单LR和多层神经网络模型流程图并进行效果回测。画图前我们得明确输入输出是啥，从多因子模型角度来说我们的输入是一堆因子然后希望得到alpha信号，从机器学习模型的角度来说输入就是一堆特征输出是分类或回归结果。本文中我们的输入是优矿提供的因子，基于我们对这些因子有预测能力的假设，我们目标是通过因子判断下一个调仓周期中股票是涨是跌。具体而言，我们取每月末的每只股票的因子为输入特征，以下个月股票是涨（值为1）是跌（值为0）为输出标签。一般而言我们需要分训练集、测试集。训练集用来训练我们的模型，测试集用来验证我们的模型效果如何，是否过拟合导致模型在测试集上的效果大打折扣。

# In[ ]:

# 生成样本
import pandas  as pd
import numpy as np
import os
FACTOR_LIST=[b'Beta60', b'OperatingRevenueGrowRate', b'NetProfitGrowRate', b'NetCashFlowGrowRate', b'NetProfitGrowRate5Y',
           b'TVSTD20',
           b'TVSTD6', b'TVMA20', b'TVMA6', b'BLEV', b'MLEV', b'CashToCurrentLiability', b'CurrentRatio', b'REC',
           b'DAREC', b'GREC',
           b'DASREV', b'SFY12P', b'LCAP', b'ASSI', b'LFLO', b'TA2EV', b'PEG5Y', b'PE', b'PB', b'PS', b'SalesCostRatio',
           b'PCF', b'CETOP',
           b'TotalProfitGrowRate', b'CTOP', b'MACD', b'DEA', b'DIFF', b'RSI', b'PSY', b'BIAS10', b'ROE', b'ROA',
           b'ROA5', b'ROE5',
           b'DEGM', b'GrossIncomeRatio', b'ROECut', b'NIAPCut', b'CurrentAssetsTRate', b'FixedAssetsTRate', b'FCFF',
           b'FCFE', b'PLRC6',
           b'REVS5', b'REVS10', b'REVS20', b'REVS60', b'HSIGMA', b'HsigmaCNE5', b'ChaikinOscillator',
           b'ChaikinVolatility', b'Aroon',
           b'DDI', b'MTM', b'MTMMA', b'VOL10', b'VOL20', b'VOL5', b'VOL60', b'RealizedVolatility', b'DASTD', b'DDNSR',
           b'Hurst']
# 对横截面的每个因子去极值标准化
def process_data(df):
    df.set_index('secID',inplace=True)
    for col in df.columns:
        df[col] = standardize(winsorize(df[col]))
    return df.reset_index()
# 得到输入特征
def get_features(tradeDate,tickers=''):
    print 'feature day:',tradeDate
    features = DataAPI.MktStockFactorsOneDayGet(tradeDate=tradeDate,secID=tickers,field=['secID']+FACTOR_LIST,pandas="1").fillna(0)
    features = process_data(features)
    return features
# 得到输出标签
def get_labels(monthEndDate):
    print 'label day:',monthEndDate
    labels = DataAPI.MktEqumGet(secID=u"",ticker=u"",monthEndDate=monthEndDate,isOpen=u"",beginDate=u"",endDate=u"",field=u"secID,chgPct",pandas="1")
    labels['label'] = labels['chgPct'].map(lambda x : 1 if (x >0) else 0)
    labels['lable'] = labels['label'].astype(int)
    return labels[['secID','label']]
# 获取不同时间段和投资域的样本
def get_samples(start,end,tickers=''):
    tradeCal = DataAPI.TradeCalGet(exchangeCD=u"XSHG",beginDate=start,endDate=end,field=u"",pandas="1").query('isMonthEnd==1')
    tradeCal['nextMonthEnd'] = tradeCal['calendarDate'].shift(-1)
    tradeCal.dropna(inplace=True)
    samples = []
    for index, row in tradeCal.iterrows():
        features = get_features(row['calendarDate'],tickers)
        labels = get_labels(row['nextMonthEnd'])
        sample = pd.merge(features,labels,how='inner',on='secID')
        samples.append(sample)
    df = pd.concat(samples).fillna(0)
    return df


# In[ ]:

data_path='tensorflow_data'
if not os.path.exists(data_path):
    os.mkdir(data_path)
print 'train_samples\n'
train_samples = get_samples('20141231','20160101')
train_samples.fillna(0).to_csv('tensorflow_data/train.csv',index=False)
print 'test_samples\n'
test_samples = get_samples('20151231','20160301')
test_samples.fillna(0).to_csv('tensorflow_data/test.csv',index=False)


# ### 2.1.2 构建单层神经网络模型
# 准备完输入输出，我们就要开始画图了。

# In[ ]:

import tensorflow as tf

# 定义模型参数
# 学习率
learning_rate = 0.01
# 迭代次数
train_epochs = 5000
# 迭代多少次后打印一些信息
display_step = 200
feature_num_index = -1
# 训练用的特征，将dataframe转为矩阵
train_X = train_samples.ix[:,1:feature_num_index].as_matrix()
# 训练用的标签
train_Y = train_samples.ix[:,-1:].as_matrix()
n_samples = train_X.shape[0]
feature_dim = train_X.shape[1]
print 'samples:',n_samples,'feature_dim:',feature_dim

# 定义一个图
with tf.Graph().as_default() as graph_lr:
    # 定义输入节点
    with tf.name_scope('Input'):
        # 为feed操作创建占位符。我们需要将准备好的训练数据注入到张量中，先明确张量的维度，这种注入操作在TensorFlow中称为feed。
        X = tf.placeholder(tf.float32, [None,feature_dim])
        Y = tf.placeholder(tf.float32, [None,1])

    with tf.name_scope('Inference'):
        # 定义变量权重W和偏置b，这里用的是tf.Variable()来创建表示这是模型中需要训练更新的值。给这些变量加个名字，以便重载模型时能识别。
        W = tf.Variable(tf.random_normal([feature_dim,1]),name='W')
        b = tf.Variable(tf.random_normal([1]),name='b')
        # 得到预测值的操作，这里我们选用的激活函数是sigmod，其余还有relu、relu6、softplus、tanh等
        Y_ = tf.nn.sigmoid(tf.matmul(X,W)+b)

    # 定义损失函数
    with tf.name_scope('Loss'):
    # 损失函数使用的是分类损失函数：交叉熵（crossentropy），代码可以用其提供的函数tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=Y_)
        cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(Y_), reduction_indices=1))
    
    # 定义优化方法
    with tf.name_scope('Train'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

    # 定义评估方法
    with tf.name_scope('Eval'):
        correct_prediction = tf.equal(tf.round(Y_), Y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 初始化各个节点
    init = tf.global_variables_initializer()
    
    # 这里我们需要将我们训练好的模型保存下来，以便回测时使用
    saver = tf.train.Saver([W,b])
    # 启动一个会话来运行我们定义的图，TensorFlow会自动完成优化工作
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(train_epochs):
            sess.run(optimizer, feed_dict={X: train_X, Y: train_Y})
            if (epoch + 1) % display_step == 0:
                epoch_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
                saver.save(sess, 'tensorflow_model_lr/my-model.ckpt')
                print 'epochs:', '%d' % (epoch + 1), 'cost=', '{:.9f}'.format(epoch_cost)
        print 'finish optimize'
        training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
        print "training cost=", training_cost
        test_X, test_Y = test_samples.ix[:,1:feature_num_index].as_matrix(), test_samples.ix[:,-1:].as_matrix()
        print("test accuracy:", sess.run(accuracy, feed_dict={X: test_X, Y: test_Y}))
        prediction = sess.run(Y_, feed_dict={X: test_X})
        P = tf.round(prediction)
        # 查看二分类相关的指标
        print 'P:',P.eval(feed_dict={X: test_X})
        TP = tf.count_nonzero(P * Y, dtype=tf.float32)
        TN = tf.count_nonzero((P - 1) * (Y - 1), dtype=tf.float32)
        FP = tf.count_nonzero(P * (Y - 1), dtype=tf.float32)
        FN = tf.count_nonzero((P - 1) * Y, dtype=tf.float32)
        print 'TP:',TP.eval(feed_dict={X: test_X, Y: test_Y})
        print 'TN:',TN.eval(feed_dict={X: test_X, Y: test_Y})
        print 'FP:',FP.eval(feed_dict={X: test_X, Y: test_Y})
        print 'FN:',FN.eval(feed_dict={X: test_X, Y: test_Y})
        precision =tf.divide(TP, TP + FP)
        recall = tf.divide(TP, TP + FN)
        F1 = 2 * precision * recall / (precision + recall)
        print 'precision:',precision.eval(feed_dict={X: test_X, Y: test_Y}) ,'recall:',recall.eval(feed_dict={X: test_X, Y: test_Y}),'F1:',F1.eval(feed_dict={X: test_X, Y: test_Y})


# ### 2.1.3 模型可视化
# 我们说了半天画图，但其实是在堆代码，那我们的图到底长啥样呢？我们将上面定义的图可视化后如下图所示，图中标识了主要的计算节点，边就是流动的张量，和本文开头介绍的一样。

# ![Alt text](https://static.wmcloud.com/v1/AUTH_datayes/rrp/66eae96b7e1d4056858a14872743ae1a.png "lr model")

# ### 2.1.4  构建多层神经网络模型
# 上面我们定义了一个简单的LR模型，下面我们增加网络的层数。

# In[ ]:

learning_rate = 0.01
train_epochs = 10
display_step = 2

train_X = train_samples.ix[:,1:feature_num_index].as_matrix()
train_Y = train_samples.ix[:,-1:].as_matrix()
n_samples = train_X.shape[0]
feature_dim = train_X.shape[1]
print 'samples:',n_samples,'feature_dim:',feature_dim

with tf.Graph().as_default():
    with tf.name_scope('Input'):
        X = tf.placeholder(tf.float32, [None,feature_dim])
        Y = tf.placeholder(tf.float32, [None,1])

    #这里我们改变下网络结构，增加了2层网络
    with tf.name_scope('Inference'):
        # 定义网络的层数及每层的节点数
        layer1 = feature_dim/2
        layer2 = layer1/2
        W = {  
            "L1":tf.Variable(tf.random_normal([feature_dim, layer1]),name='W_L1'),  
            "L2":tf.Variable(tf.random_normal([layer1, layer2]),name='W_L2'),  
            "output": tf.Variable(tf.random_normal([layer2, 1]),name='W_output')  
        }  

        b = {  
            "L1":tf.Variable(tf.random_normal([layer1]),name='b_L1'),  
            "L2":tf.Variable(tf.random_normal([layer2]),name='b_L2'),  
            "output":tf.Variable(tf.random_normal([1]),name='b_output')  
        }  
 
        net1 = tf.nn.sigmoid(tf.matmul(X, W["L1"]) + b["L1"])
        net2 = tf.nn.sigmoid(tf.matmul(net1, W["L2"]) + b["L2"])
        Y_ = tf.nn.sigmoid(tf.matmul(net2, W["output"]) + b["output"]) 

    with tf.name_scope('Loss'):
        cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(Y_), reduction_indices=1))
    
    with tf.name_scope('Train'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

    with tf.name_scope('Eval'):
        correct_prediction = tf.equal(tf.round(Y_), Y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init = tf.global_variables_initializer()
    
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(train_epochs):
            sess.run(optimizer, feed_dict={X: train_X, Y: train_Y})
            if (epoch + 1) % display_step == 0:
                epoch_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
                saver.save(sess, 'tensorflow_model_mlp/my-model.ckpt')
                print 'epochs:', '%d' % (epoch + 1), 'cost=', '{:.9f}'.format(epoch_cost)
        print 'finish optimize'
        training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
        print "training cost=", training_cost
        test_X, test_Y = test_samples.ix[:,1:feature_num_index].as_matrix(), test_samples.ix[:,-1:].as_matrix()
        print("test accuracy:", sess.run(accuracy, feed_dict={X: test_X, Y: test_Y}))
        prediction = sess.run(Y_, feed_dict={X: test_X})
        P = tf.round(prediction)
        # print 'P:',P.eval(feed_dict={X: test_X})
        TP = tf.count_nonzero(P * Y, dtype=tf.float32)
        TN = tf.count_nonzero((P - 1) * (Y - 1), dtype=tf.float32)
        FP = tf.count_nonzero(P * (Y - 1), dtype=tf.float32)
        FN = tf.count_nonzero((P - 1) * Y, dtype=tf.float32)
        print 'TP:',TP.eval(feed_dict={X: test_X, Y: test_Y})
        print 'TN:',TN.eval(feed_dict={X: test_X, Y: test_Y})
        print 'FP:',FP.eval(feed_dict={X: test_X, Y: test_Y})
        print 'FN:',FN.eval(feed_dict={X: test_X, Y: test_Y})
        precision =tf.divide(TP, TP + FP)
        recall = tf.divide(TP, TP + FN)
        F1 = 2 * precision * recall / (precision + recall)
        print 'precision:',precision.eval(feed_dict={X: test_X, Y: test_Y}) ,'recall:',recall.eval(feed_dict={X: test_X, Y: test_Y}),'F1:',F1.eval(feed_dict={X: test_X, Y: test_Y})


# ### 2.1.5 回测
# 训练完模型后，如果模型的效果不错，我们就可以用模型来进行回测。下面我们将保存的模型载入内存，然后用回测框架进行回测。

# In[ ]:

#在载入模型时需要明确下我们的模型是怎样的，主要涉及到几个训练的变量，我们也只需要把这个图的几个节点明确就行。
import tensorflow as tf
tf.reset_default_graph()
sess = tf.Session()
X = tf.placeholder(tf.float32, [None,feature_dim])
Y = tf.placeholder(tf.float32, [None,1])
# 这里需要将变量的名字与之前训练时的对应，注意训练时这两个变量的name_scope是Inference
W = tf.Variable(tf.zeros([feature_dim,1]),name='Inference/W')
b = tf.Variable(tf.zeros([1]),name='Inference/b')
Y_ = tf.nn.sigmoid(tf.matmul(X,W)+b)
saver = tf.train.Saver()
saver.restore(sess, "tensorflow_model_lr/my-model.ckpt")


# 我们用的是15年的数据进行训练，16年1、2月的数据进行测试的，因此我们回测区间要避开这两个时间段。因为我们在训练测试时会进行调优，如果在测试集上进行回测就会用到未来数据。我们选取模型预测值最大的前10个来构建我们的组合。

# In[ ]:

start = '2016-03-01'                       # 回测起始时间
end = '2017-03-01'                         # 回测结束时间
universe = DynamicUniverse('HS300')        # 证券池，支持股票和基金、期货
benchmark = 'HS300'                        # 策略参考基准
freq = 'd'                                 # 'd'表示使用日频率回测，'m'表示使用分钟频率回测
refresh_rate = Monthly(1)                          # 执行handle_data的时间间隔

accounts = {
    'fantasy_account': AccountConfig(account_type='security', capital_base=10000000)
}
    
def get_signal(features):
    prediction = sess.run(Y_, feed_dict={X: features.ix[:,1:].as_matrix()})
    prediction = pd.DataFrame(prediction,columns=['prob'])
    prediction['prob'] = prediction['prob'].astype(float)
    res = pd.concat([features['secID'],prediction],axis=1).set_index('secID')
    return res['prob'].to_dict()
    
def initialize(context):                   # 初始化策略运行环境
    pass

def handle_data(context):                  # 核心策略逻辑
    account = context.get_account('fantasy_account')
    close_price = account.reference_price
    pre_date = context.previous_date.strftime('%Y%m%d')
    input_features = get_features(pre_date,context.get_universe(asset_type='stock'))
    signals = get_signal(input_features)
    wts = long_only(signals, select_type=0, top_ratio=0.1, weight_type=0, target_date=pre_date)
    # 交易部分
    sell_list = [stk for stk in account.security_position if stk not in wts]
    for stk in sell_list:
        order_to(stk,0)

    c = account.reference_portfolio_value
    change = {}
    for stock, w in wts.iteritems():
        p = account.reference_price[stock]
        if not np.isnan(p) and p > 0:
            change[stock] = int(c * w / p) - account.security_position.get(stock, 0)

    for stock in sorted(change, key=change.get):
        order(stock, change[stock])


# 同样我们尝试下多层网络的效果，下面代码展示如何载入之前训练的MLP模型。

# In[ ]:

tf.reset_default_graph()
X = tf.placeholder(tf.float32, [None,feature_dim])
Y = tf.placeholder(tf.float32, [None,1])
layer1 = feature_dim/2
layer2 = layer1/2
W = {  
    "L1":tf.Variable(tf.random_normal([feature_dim, layer1]),name='Inference/W_L1'),  
    "L2":tf.Variable(tf.random_normal([layer1, layer2]),name='Inference/W_L2'),  
    "output": tf.Variable(tf.random_normal([layer2, 1]),name='Inference/W_output')  
    }  

b = {  
    "L1":tf.Variable(tf.random_normal([layer1]),name='Inference/b_L1'),  
    "L2":tf.Variable(tf.random_normal([layer2]),name='Inference/b_L2'),  
    "output":tf.Variable(tf.random_normal([1]),name='Inference/b_output')  
}  

net1 = tf.nn.sigmoid(tf.matmul(X, W["L1"]) + b["L1"])
net2 = tf.nn.sigmoid(tf.matmul(net1, W["L2"]) + b["L2"])
Y_ = tf.nn.sigmoid(tf.matmul(net2, W["output"]) + b["output"])
sess = tf.Session()
saver = tf.train.Saver()
saver.restore(sess, "tensorflow_model_mlp/my-model.ckpt")


# ## 2.2 使用TensorFlow高级API构建模型
# ### 2.2.1 TensorFlow架构
# 利用底层函数实现网络还是有一定的复杂性，但具体的细节我们都能掌握。如果我们想快速搭建一个网络，该怎么办呢？好在Google在TensorFlow1.3版本提供了Estimator这个高级API，该API提供了即简易又灵活的方式来构建模型。首先它提供了一些现成模型，例如：
# - DNNClassifier
# - DNNRegressor
# - LinearClassifier
# - LinearRegressor
# - DNNLinearCombinedClassifier
# - DNNLinearCombinedRegressor
# 
# 这些模型我们只要改改参数就能用，不用关心底层的具体实现。稍后我们会演示一个具体例子。
# ![tensorflow](http://odqb0lggi.bkt.clouddn.com/548d8a7df9f06c45e7073e61/04cdb2f4-33e6-11e8-a6f8-0242ac140002)
# 一般而言，具体问题需要具体解决，因此，该API还提供了可以自定义的框架。下图是Google内部的模型使用统计。
# ![Google内部](http://odqb0lggi.bkt.clouddn.com/548d8a7df9f06c45e7073e61/937b1b98-33fc-11e8-a6f8-0242ac140002)

# ### 2.2.2 Estimator结构

# init(model_fn=None, model_dir=None, config=None, params=None, feature_engineering_fn=None)
# - model_fn: 模型定义，定义了train, eval, predict的实现
# - model_dir: 日志文件和训练参数的保存目录
# - config: 模型运行的配置
# - params: 模型需要的一些参数
# 
# model_fn(features, labels, mode, params) 
# - features: 样本数据的x 
# - labels: 样本数据的y 
# - mode: 模式 有3种TRAIN/EVAL/PREDICT，根据这个参数，model_fn可以做特定的处理 
# - params: mode_fn需要的其他参数，dict数据结构
# 
# ![estimator](http://odqb0lggi.bkt.clouddn.com/548d8a7df9f06c45e7073e61/8a57619e-340a-11e8-a6f8-0242ac140002)

# ### 2.2.2 载入数据
# 我们还是使用之前生成的数据，这里不再重复，直接读取

# In[ ]:

import pandas as pd
import numpy as np
feature_num_index=-1
train_samples = pd.read_csv("tensorflow_data/train.csv")
test_samples = pd.read_csv("tensorflow_data/test.csv")
feature_cols = list(train_samples.ix[:,1:feature_num_index].columns)
train_X = train_samples.ix[:,1:feature_num_index].as_matrix()
train_Y = train_samples.ix[:,-1:].as_matrix()
n_samples = train_X.shape[0]
feature_dim = train_X.shape[1]
print n_samples,feature_dim


# ### 2.2.2 使用封装好的模型

# In[ ]:

import tensorflow as tf
#获得输入函数对象
def get_input_fn(data_set,batch_size=128,num_epochs=None, shuffle=False):
    return tf.estimator.inputs.numpy_input_fn(
    x={"x": np.array(data_set.ix[:,1:feature_num_index],dtype=np.float32)},
    y=np.array(data_set['label'],dtype=np.float32),
    batch_size=batch_size,
    num_epochs=num_epochs,
    shuffle=shuffle)
train_input_fn = get_input_fn(train_samples,batch_size=128,num_epochs=1,shuffle=True)
test_input_fn = get_input_fn(test_samples,batch_size=128,num_epochs=1)


# In[ ]:

feature_columns = [tf.feature_column.numeric_column('x', shape=np.array(train_samples.ix[:,1:feature_num_index]).shape[1:])]    
dnn = tf.estimator.DNNClassifier(model_dir='dnn_estimator/',
                feature_columns=feature_columns,
                 hidden_units=[feature_dim/2, feature_dim/4, feature_dim/8],
                 activation_fn=tf.nn.sigmoid,
                 dropout=0.2,
                 n_classes=2, 
                 optimizer="Adam")


# In[ ]:

dnn.train(input_fn=train_input_fn)


# In[ ]:

dnn.evaluate(input_fn=test_input_fn)


# ### 2.2.3 自定义模型

# In[ ]:

model_params={'layer1':feature_dim/2,'layer2':feature_dim/4,"learning_rate":0.01}


# In[ ]:

###自定义模型
def my_model_fn(features, labels, mode, params):
    #定义第一层
    first_hidden_layer = tf.layers.dense(features['x'], params['layer1'], use_bias=True, activation=tf.nn.sigmoid)
    #定义第二层
    second_hidden_layer = tf.layers.dense(first_hidden_layer, params['layer2'], use_bias=True, activation=tf.nn.sigmoid)
    #连接到输出层 
    output_layer = tf.layers.dense(second_hidden_layer,1, use_bias=True, activation=tf.nn.sigmoid)
    print first_hidden_layer
    #输出层reshape
    predictions = tf.reshape(output_layer, [-1])
    #对于预测模式需要返回预测结果,需要写在输出层后
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions={"predict": predictions})
    #设置损失函数
    loss = tf.losses.sigmoid_cross_entropy(labels, predictions)
    #设置优化方法
    optimizer = tf.train.GradientDescentOptimizer(
      learning_rate=params["learning_rate"])
    train_op = optimizer.minimize(
      loss=loss, global_step=tf.train.get_global_step())
    
    prediction_label = tf.round(predictions)
    #根据模型不同计算一些评价指标
    def _eval_confusion_matrix(labels, predictions):
        with tf.variable_scope("eval_confusion_matrix"):
            con_matrix = tf.confusion_matrix(labels=labels, predictions=predictions, num_classes=2)
            con_matrix_sum = tf.Variable(tf.zeros(shape=(2,2), dtype=tf.int32),
                                                trainable=False,
                                                name="confusion_matrix_result",
                                                collections=[tf.GraphKeys.LOCAL_VARIABLES])
            update_op = tf.assign_add(con_matrix_sum, con_matrix)
            return tf.convert_to_tensor(con_matrix_sum), update_op
    
    eval_metric_ops = {
    "rmse": tf.metrics.root_mean_squared_error(
      tf.cast(labels, tf.float32), predictions),
    "accuracy": tf.metrics.accuracy(labels,prediction_label),
    "false negatives":tf.metrics.false_negatives(labels,prediction_label),
    "false positives":tf.metrics.false_positives(labels,prediction_label),
    "true positives":tf.metrics.true_positives(labels,prediction_label),
    "true positives":tf.metrics.mean_absolute_error(labels,prediction_label),
    "recall":tf.metrics.recall(labels,prediction_label),
    "conv_matrix": _eval_confusion_matrix(
        labels, prediction_label)
    }

    

    #返回训练和测试需要的结果
    return tf.estimator.EstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=eval_metric_ops)


# In[ ]:

my_nn = tf.estimator.Estimator(model_fn=my_model_fn,params=model_params, model_dir='tensorflow_estimator/')


# In[ ]:

my_nn.train(input_fn=train_input_fn)


# In[ ]:

my_nn.get_variable_names()


# In[ ]:

my_nn.get_variable_value('dense/bias')


# In[ ]:

eval_result = my_nn.evaluate(input_fn=test_input_fn)


# In[ ]:

pd.DataFrame.from_dict(eval_result,orient='index')


# In[ ]:

start = '2016-03-01'                       # 回测起始时间
end = '2017-03-01'                         # 回测结束时间
universe = DynamicUniverse('HS300')        # 证券池，支持股票和基金、期货
benchmark = 'HS300'                        # 策略参考基准
freq = 'd'                                 # 'd'表示使用日频率回测，'m'表示使用分钟频率回测
refresh_rate = Monthly(1)                          # 执行handle_data的时间间隔

accounts = {
    'fantasy_account': AccountConfig(account_type='security', capital_base=10000000)
}
#载入模型
session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
session_config.gpu_options.allow_growth = True
session_config.allow_soft_placement = True
config = tf.estimator.RunConfig(session_config=session_config)
trained_model = tf.estimator.Estimator(model_fn=my_model_fn, model_dir='tensorflow_estimator/',params=model_params,config=config)

# 得到输入特征
def get_features(tradeDate,tickers=''):
    print 'feature day:',tradeDate
    features = DataAPI.MktStockFactorsOneDayGet(tradeDate=tradeDate,secID=tickers,field=['secID']+FACTOR_LIST,pandas="1").fillna(0)
    features = process_data(features)
    return features

def getSignal(predictions,secIDs):
    signal={}
    for pred,secID in zip(predictions,secIDs):
        signal.update({secID:float(pred['predict'])})
    return signal

def initialize(context):                   # 初始化策略运行环境
    pass

def handle_data(context):                  # 核心策略逻辑
    account = context.get_account('fantasy_account')
    close_price = account.reference_price
    pre_date = context.previous_date.strftime('%Y%m%d')
    input_features = get_features(pre_date,context.get_universe(asset_type='stock'))
    input_features['label']=-1
    print input_features.shape
    predict_input_fn = get_input_fn(input_features)
    predictions = trained_model.predict(input_fn=predict_input_fn)
    signals = getSignal(predictions,input_features['secID'])
     # 组合构建                
    wts = long_only(signals, select_type=0, top_ratio=0.1, weight_type=0, target_date=pre_date)
    # 交易部分
    sell_list = [stk for stk in account.security_position if stk not in wts]
    for stk in sell_list:
        order_to(stk,0)

    c = account.reference_portfolio_value
    change = {}
    for stock, w in wts.iteritems():
        p = account.reference_price[stock]
        if not np.isnan(p) and p > 0:
            change[stock] = int(c * w / p) - account.security_position.get(stock, 0)

    for stock in sorted(change, key=change.get):
        order(stock, change[stock])

