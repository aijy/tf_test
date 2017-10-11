import tensorflow as tf
import numpy as np
import pickle as pl
import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn.preprocessing as prep
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
t1=time.time()

L_all=['EURUSD','GBPUSD','NZDUSD','USDCHF','USDJPY','XAUUSD','EURAUD',
       'EURNZD','EURCAD','GBPAUD','GBPNZD','GBPCAD','GBPCHF','AUDCAD',
       'AUDCHF','AUDJPY','NZDCAD','NZDCHF','NZDJPY','CADCHF','CADJPY','CHFJPY']


swap=pd.read_csv('tickmill_swap.csv',dtype=np.float32)
type=[]
a=pd.read_csv('fx_data_15.csv',dtype=np.float32)
for each in a:
    if swap[each].values[1]>-0.5:
        type.append(-1)
        continue
    if swap[each].values[0]>-0.5:
        type.append(1)
        continue
    type.append(0)
a=a*type
a.to_csv('test.csv')
exit()
l1=['EURUSD','GBPUSD','AUDUSD','NZDUSD','USDCAD','USDCHF','USDJPY','XAUUSD']
l2=['EURUSD','EURGBP','EURAUD','EURNZD','EURCAD','EURCHF','EURJPY','XAUUSD']
l3=['GBPUSD','EURGBP','GBPAUD','GBPNZD','GBPCAD','GBPCHF','GBPJPY','XAUUSD']
l4=['AUDUSD','EURAUD','GBPAUD','AUDNZD','AUDCAD','AUDCHF','AUDJPY','XAUUSD']
l5=['NZDUSD','EURNZD','GBPNZD','AUDNZD','NZDCAD','NZDCHF','NZDJPY','XAUUSD']
l6=['USDCAD','EURCAD','GBPCAD','AUDCAD','NZDCAD','CADCHF','CADJPY','XAUUSD']
l7=['USDCHF','EURCHF','GBPCHF','AUDCHF','NZDCHF','CADCHF','CHFJPY','XAUUSD']
l8=['USDJPY','EURJPY','GBPJPY','AUDJPY','NZDJPY','CADJPY','CHFJPY','XAUUSD']
L=l1

b1=pd.DataFrame(a,columns=L,dtype=np.float32)
base=pd.DataFrame(a,columns=l1,dtype=np.float32)
b=b1
first_value=b[L].values[0]
data_p1=b[L].values-b[L].values[0]
data_p=[]

# l1=['EURUSD','GBPUSD','AUDUSD','NZDUSD','USDCAD','USDCHF','USDJPY']
if L==l1:
    for i in range(len(data_p1)):
        tickvaule_1 = np.array([1, 1, 1, 1, 100000.0 / b1['USDCAD'].values[i], 100000.0 / b1['USDCHF'].values[i],\
                                100000.0 / b1['USDJPY'].values[i],0.1],dtype=np.float32)
        data_p.append(data_p1[i]*np.abs(tickvaule_1))
# l2=['EURUSD','EURGBP','EURAUD','EURNZD','EURCAD','EURCHF','EURJPY']
if L==l2:
    for i in range(len(data_p1)):
        tickvaule_1 = np.array([1, base['EURUSD'].values[i]/b1['EURGBP'].values[i], base['EURUSD'].values[i]/b1['EURAUD'].values[i],\
                                base['EURUSD'].values[i]/b1['EURNZD'].values[i], base['EURUSD'].values[i]/b1['EURCAD'].values[i],\
                                base['EURUSD'].values[i]/b1['EURCHF'].values[i],base['EURUSD'].values[i]/b1['EURJPY'].values[i],1])
        data_p.append(data_p1[i]*np.abs(tickvaule_1))
# a=np.array([ 0.16260192,-0.09447282, 0.09285256,-0.14521897,0.05961967, 0.25467968,-0.13755047,-0.05300393])
# print(tickvaule_1)
# print((data_p1[-1]-data_p1[0])*a)
# bb=np.array([1,1.32,0.81,0.76,0.8,1.05,0.9,1])
# print(np.sum(data_p1[-1]*bb*a))
# print(np.sum(data_p[-1]*a))
# exit()
# l3=['GBPUSD','EURGBP','GBPAUD','GBPNZD','GBPCAD','GBPCHF','GBPJPY']
if L==l3:
    for i in range(len(data_p1)):
        tickvaule_1 = np.array([1, base['EURUSD'].values[i]/b1['EURGBP'].values[i], base['GBPUSD'].values[i]/b1['GBPAUD'].values[i],\
                                base['GBPUSD'].values[i]/b1['GBPNZD'].values[i], base['GBPUSD'].values[i]/b1['GBPCAD'].values[i],\
                                base['GBPUSD'].values[i]/b1['GBPCHF'].values[i],base['GBPUSD'].values[i]/b1['GBPJPY'].values[i],1])
        data_p.append(data_p1[i]*np.abs(tickvaule_1))
# l4=['AUDUSD','EURAUD','GBPAUD','AUDNZD','AUDCAD','AUDCHF','AUDJPY']
if L == l4:
    for i in range(len(data_p1)):
        tickvaule_1 = np.array([1, base['EURUSD'].values[i]/b1['EURAUD'].values[i], base['GBPUSD'].values[i]/b1['GBPAUD'].values[i],\
                                base['AUDUSD'].values[i]/b1['AUDNZD'].values[i], base['AUDUSD'].values[i]/b1['AUDCAD'].values[i],\
                                base['AUDUSD'].values[i]/b1['AUDCHF'].values[i],base['AUDUSD'].values[i]/b1['AUDJPY'].values[i],1])
        data_p.append(data_p1[i]*np.abs(tickvaule_1))

# l5=['NZDUSD','EURNZD','GBPNZD','AUDNZD','NZDCAD','NZDCHF','NZDJPY']
if L == l5:
    for i in range(len(data_p1)):
        tickvaule_1 = np.array([1, base['EURUSD'].values[i]/b1['EURNZD'].values[i], base['GBPUSD'].values[i]/b1['GBPNZD'].values[i],\
                                base['AUDUSD'].values[i]/b1['AUDNZD'].values[i], base['NZDUSD'].values[i]/b1['NZDCAD'].values[i],\
                                base['NZDUSD'].values[i]/b1['NZDCHF'].values[i],base['NZDUSD'].values[i]/b1['NZDJPY'].values[i],1])
        data_p.append(data_p1[i]*np.abs(tickvaule_1))
# l6=['USDCAD','EURCAD','GBPCAD','AUDCAD',''NZDCAD',CADCHF','CADJPY']
if L == l6:
    for i in range(len(data_p1)):
        tickvaule_1 = np.array([100000.0 / b1['USDCAD'].values[i], base['EURUSD'].values[i]/b1['EURCAD'].values[i], base['GBPUSD'].values[i]/b1['GBPCAD'].values[i],\
                                base['AUDUSD'].values[i]/b1['AUDCAD'].values[i], base['NZDUSD'].values[i]/b1['NZDCAD'].values[i], \
                                100000.0*100000 / base['USDCAD'].values[i]/b1['CADCHF'].values[i],100000.0*100000 / base['USDCAD'].values[i]/b1['CADJPY'].values[i],1])
        data_p.append(data_p1[i]*np.abs(tickvaule_1))
# l7=['USDCHF','EURCHF','GBPCHF','AUDCHF','NZDCHF','CADCHF','CHFJPY']
if L == l7:
    for i in range(len(data_p1)):
        tickvaule_1 = np.array([100000.0 / b1['USDCHF'].values[i], base['EURUSD'].values[i]/b1['EURCHF'].values[i], base['GBPUSD'].values[i]/b1['GBPCHF'].values[i],\
                                base['AUDUSD'].values[i]/b1['AUDCHF'].values[i], base['NZDUSD'].values[i]/b1['NZDCHF'].values[i], \
                                100000.0*100000 / base['USDCAD'].values[i]/b1['CADCHF'].values[i],100000.0*100000 / base['USDCHF'].values[i]/b1['CHFJPY'].values[i],1])
        data_p.append(data_p1[i]*np.abs(tickvaule_1))
# l8 = ['USDJPY', 'EURJPY', 'GBPJPY', 'AUDJPY', 'NZDJPY', 'CADJPY', 'CHFJPY']
if L == l8:
    for i in range(len(data_p1)):
        data_p.append(data_p1[i])

data_p2=np.array(data_p[960:],dtype=np.float32)
data_p=np.array(data_p[0:960],dtype=np.float32)
weight=tf.Variable(tf.random_normal([len(L),1]),dtype=tf.float32,)

weight_abs=tf.square(weight)
sum=tf.reduce_sum(weight_abs,reduction_indices=[0])
w=weight_abs/sum
v=tf.matmul(data_p,w)
loss_mean=tf.reduce_mean(v)
loss=tf.reduce_sum(tf.square(tf.matmul(data_p,w)-loss_mean),reduction_indices=[0])
loss_1=(tf.reduce_max(v)-v[-1])/(v[-1]-tf.reduce_min(v))
# v=tf.matmul(data_p,w)
# print(data_p)
# exit()
# loss_1=tf.reduce_max(v)-tf.reduce_min(v)
# loss=(v[-1]-tf.reduce_min(v))/(tf.reduce_max(v)-v[-1])
# loss_mean=tf.reduce_mean(tf.matmul(data_p,w))
# loss_2=1/tf.reduce_sum(tf.square(tf.matmul(data_p,w)-loss_mean),reduction_indices=[0])

train_step=tf.train.AdadeltaOptimizer(0.1).minimize(loss)
train_step_1=tf.train.AdadeltaOptimizer(0.1).minimize(loss_1)
# train_step_2=tf.train.AdadeltaOptimizer(0.3).minimize(loss_2)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # print sess.run([weight,weight_abs])
    # print sess.run([sign, tf.sign(sign)])
    for i in range(400000):
        sess.run(train_step)

        if i%30000==0:
            print(sess.run(loss))
            print(sess.run(w))

    weight_result=sess.run(w)
    weight_result=np.round(weight_result,2)
    print(weight_result)
    a=np.matmul(data_p,weight_result)
    b=np.matmul(data_p2,weight_result)
    print(a.max(),a.min(),a.mean(),a[-1])

    print('input double mean_value=%.2f;' %a.mean())
    print('input double %s_LOT=%.2f;' %(L[0], weight_result[0][0]))
    print('input double %s_LOT=%.2f;' %(L[1], weight_result[1][0]))
    print('input double %s_LOT=%.2f;' %(L[2], weight_result[2][0]))
    print('input double %s_LOT=%.2f;' %(L[3], weight_result[3][0]))
    print('input double %s_LOT=%.2f;' %(L[4], weight_result[4][0]))
    print('input double %s_LOT=%.2f;' %(L[5], weight_result[5][0]))
    print('input double %s_LOT=%.2f;' %(L[6], weight_result[6][0]))
    print('input double %s_LOT=%.2f;' %(L[7], weight_result[7][0]))
    print('string sym[N]={"%s","%s","%s","%s","%s","%s","%s","%s"};' %(L[0],L[1],L[2],L[3],L[4],L[5],L[6],L[7]))
    print('double base[N]={%d,%d,%d,%d,%d,%d,%d,%d};' %(first_value[0],first_value[1],first_value[2],first_value[3],first_value[4],first_value[5],first_value[6],first_value[7]))
    print(time.time()-t1)

    plt.plot(range(len(a)),a,range(len(a),len(a)+144),b[:144],range(len(a)+144,len(a)+len(b)),b[144:])

    plt.grid()
    plt.show()

