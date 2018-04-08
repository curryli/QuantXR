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
#print(start, end)
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
tmp_HS300 = np.delete(tmp_HS300,0,axis=0)
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

ret = HS300_1d.iloc[-144:,-1]
ret = np.array(ret)
wid = np.zeros((144,3))
wid[np.arange(144),ret] = 1
ret = wid
print(ret.shape)


######################################################################################
# 在准备好的输入数据中，分别抽取训练数据和测试数据，按照 70/30 原则来做。
train_test_split = np.random.rand(len(ret)) < 0.70
train_x = tmp_HS300[train_test_split]
train_y = ret[train_test_split]
test_x = tmp_HS300[~train_test_split]
test_y = ret[~train_test_split]


# start tensorflow interactiveSession
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

sess = tf.InteractiveSession()

# weight initialization
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

# convolution
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
# pooling
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


training_epochs = 100
input_height = 224
input_width = 232
num_labels = 3  #3类， 输出3维的onehot
num_channels = 1  #
batch_size = 1
learning_rate = 1e-4   #如果loss出现NAN，或者accuracy固定在一个很小的位置，很可能上来学习率就设的太大了，导致到不了最小点。   这时候尝试一个量级一个量级地降学习率试试


total_batchs = ret.shape[0]//batch_size

# 下面是使用 Tensorflow 创建神经网络的过程。
X = tf.placeholder(tf.float32, shape=[None,input_height,input_width,num_channels])
Y = tf.placeholder(tf.float32, shape=[None,num_labels])

# [filter_height, filter_width, in_channels, out_channels]  [卷积核的高，卷积核的宽，输入通道数，输出通道数]  一般卷积核3*3 5*5 9*9...
f_h_1 = 5
f_w_1 = 5  # 自定义
in_chan_1 = num_channels
out_chan_1 = 32  # 自定义，深度

x_Arr = tf.reshape(X, [-1, input_height,input_width,num_channels])

# first convolutinal layer
w_conv1 = weight_variable([f_h_1, f_w_1, in_chan_1, out_chan_1])
b_conv1 = bias_variable([out_chan_1])


h_conv1 = tf.nn.relu(conv2d(x_Arr, w_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


f_h_2 = 5
f_w_2 = 5  #自定义,比f_w_1小一点
in_chan_2 = out_chan_1  #上一层输出chann数作为当前输入chann数
out_chan_2 = 16 #自定义，深度


# second convolutional layer
w_conv2 = weight_variable([f_h_2, f_w_2, in_chan_2, out_chan_2])
b_conv2 = bias_variable([out_chan_2])

h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

shape = h_pool2.get_shape().as_list()
print(shape[1],shape[2],shape[3])  #经过两次max_pool_1x2


num_out_1 = 1024  #自定义，第一层flattern输出维度   （输入维度就是shape[1] * shape[2] * shape[3]展开）
# densely connected layer
w_fc1 = weight_variable([shape[1] * shape[2] * shape[3], num_out_1])
b_fc1 = bias_variable([num_out_1])

h_pool2_flat = tf.reshape(h_pool2, [-1, shape[1] * shape[2] * shape[3]])
h_fc1 = tf.nn.tanh(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

# dropout
h_fc1_drop = tf.nn.dropout(h_fc1, 0.5)  #保存0.5

# readout layer
w_fc2 = weight_variable([num_out_1, num_labels])   #第二层输出维度与类别数一致
b_fc2 = bias_variable([num_labels])

ylogits = tf.nn.bias_add(tf.matmul(h_fc1_drop, w_fc2) ,b_fc2)

loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=ylogits))

optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)

y_ = tf.nn.softmax(ylogits)
correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# 开始训练
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    # 开始迭代
    for epoch in range(training_epochs):
        for b in range(total_batchs):
            offset = (b * batch_size) % (train_y.shape[0] - batch_size)
            batch_x = train_x[offset:(offset + batch_size), :, :, :]
            #print(batch_x[0])
            batch_y = train_y[offset:(offset + batch_size), :]
            _, c , a = sess.run([optimizer, loss,ylogits], feed_dict={X: batch_x, Y: batch_y})
            #print(a)
        print("Epoch {}: Training Loss = {}, Training Accuracy = {}".format(epoch, c, sess.run(accuracy, feed_dict={X: train_x, Y: train_y})))

    y_p = tf.argmax(y_, 1)
    y_true = np.argmax(test_y, 1)
    final_acc, y_pred = sess.run([accuracy, y_p], feed_dict={X: test_x, Y: test_y})
    print("Testing Accuracy: {}".format(final_acc))
    # 计算模型的 metrics
    print("Precision", precision_score(y_true.tolist(), y_pred.tolist(), average='weighted'))
    print("Recall", recall_score(y_true, y_pred, average='weighted'))
    print("f1_score", f1_score(y_true, y_pred, average='weighted'))
    print("confusion_matrix")
    print(confusion_matrix(y_true, y_pred))