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

data=pd.read_csv('../data/fx_data_15.csv',dtype=np.float32)
data_v=pd.read_csv('../data/fx_60.csv',dtype=np.float32)
swap=pd.read_csv('../data/tickmill_swap.csv',dtype=np.float32)
All_Fx=["EURUSD","GBPUSD","AUDUSD","NZDUSD","USDJPY","USDCAD","USDCHF",\
        "EURGBP","EURAUD","EURNZD","EURCAD","EURCHF","EURJPY",\
        "GBPAUD","GBPNZD","GBPCAD","GBPCHF","GBPJPY", \
        "AUDNZD", "AUDCAD","AUDCHF", "AUDJPY",\
        "NZDCAD","NZDCHF","NZDJPY",\
        "CADJPY","CADCHF",\
        "CHFJPY",]
base=pd.DataFrame(data,columns=All_Fx,dtype=np.float32)
base_v=pd.DataFrame(data_v,columns=All_Fx,dtype=np.float32)

type=[]
for each in All_Fx:
    if swap[each].values[0]>swap[each].values[1]:
        type.append(1)
    else :
        type.append(-1)
type=np.array(type)

first_value=data[All_Fx].values[0]
data_p1=data[All_Fx].values-data[All_Fx].values[0]
first_value_v=data_v[All_Fx].values[0]
data_p1_v=data_v[All_Fx].values-data_v[All_Fx].values[0]

data_p=[]
data_p_v=[]
for i in range(len(base['EURUSD'])):
    tickvaule= np.array([1, 1, 1, 1,100000.0 / base['USDJPY'].values[i], 100000.0 / base['USDCAD'].values[i], 100000.0 / base['USDCHF'].values[i],
     base['EURUSD'].values[i] / base['EURGBP'].values[i], base['EURUSD'].values[i] / base['EURAUD'].values[i], \
     base['EURUSD'].values[i] / base['EURNZD'].values[i], base['EURUSD'].values[i] / base['EURCAD'].values[i], \
     base['EURUSD'].values[i] / base['EURCHF'].values[i], base['EURUSD'].values[i] / base['EURJPY'].values[i], base['GBPUSD'].values[i] / base['GBPAUD'].values[i], \
     base['GBPUSD'].values[i] / base['GBPNZD'].values[i], base['GBPUSD'].values[i] / base['GBPCAD'].values[i], \
     base['GBPUSD'].values[i] / base['GBPCHF'].values[i], base['GBPUSD'].values[i] / base['GBPJPY'].values[i], \
     base['AUDUSD'].values[i] / base['AUDNZD'].values[i], base['AUDUSD'].values[i] / base['AUDCAD'].values[i], \
     base['AUDUSD'].values[i] / base['AUDCHF'].values[i], base['AUDUSD'].values[i] / base['AUDJPY'].values[i], \
     base['NZDUSD'].values[i] / base['NZDCAD'].values[i], base['NZDUSD'].values[i] / base['NZDCHF'].values[i], \
     base['NZDUSD'].values[i] / base['NZDJPY'].values[i], 100000.0 * 100000 / base['USDCAD'].values[i] / base['CADCHF'].values[i],\
     100000.0 * 100000 / base['USDCAD'].values[i] / base['CADJPY'].values[i],100000.0 * 100000 / base['USDCHF'].values[i]/base['CHFJPY'].values[i]])
    data_p.append(data_p1[i]*tickvaule*type)
for i in range(len(base_v['EURUSD'])):
    tickvaule = np.array([1, 1, 1, 1, 100000.0 / base_v['USDJPY'].values[i], 100000.0 / base_v['USDCAD'].values[i],
                          100000.0 / base_v['USDCHF'].values[i],
                          base_v['EURUSD'].values[i] / base_v['EURGBP'].values[i],
                          base_v['EURUSD'].values[i] / base_v['EURAUD'].values[i], \
                          base_v['EURUSD'].values[i] / base_v['EURNZD'].values[i],
                          base_v['EURUSD'].values[i] / base_v['EURCAD'].values[i], \
                          base_v['EURUSD'].values[i] / base_v['EURCHF'].values[i],
                          base_v['EURUSD'].values[i] / base_v['EURJPY'].values[i],
                          base_v['GBPUSD'].values[i] / base_v['GBPAUD'].values[i], \
                          base_v['GBPUSD'].values[i] / base_v['GBPNZD'].values[i],
                          base_v['GBPUSD'].values[i] / base_v['GBPCAD'].values[i], \
                          base_v['GBPUSD'].values[i] / base_v['GBPCHF'].values[i],
                          base_v['GBPUSD'].values[i] / base_v['GBPJPY'].values[i], \
                          base_v['AUDUSD'].values[i] / base_v['AUDNZD'].values[i],
                          base_v['AUDUSD'].values[i] / base_v['AUDCAD'].values[i], \
                          base_v['AUDUSD'].values[i] / base_v['AUDCHF'].values[i],
                          base_v['AUDUSD'].values[i] / base_v['AUDJPY'].values[i], \
                          base_v['NZDUSD'].values[i] / base_v['NZDCAD'].values[i],
                          base_v['NZDUSD'].values[i] / base_v['NZDCHF'].values[i], \
                          base_v['NZDUSD'].values[i] / base_v['NZDJPY'].values[i],
                          100000.0 * 100000 / base_v['USDCAD'].values[i] / base_v['CADCHF'].values[i], \
                          100000.0 * 100000 / base_v['USDCAD'].values[i] / base_v['CADJPY'].values[i],
                          100000.0 * 100000 / base_v['USDCHF'].values[i] / base_v['CHFJPY'].values[i]])
    data_p_v.append(data_p1_v[i]*tickvaule*type)

data_p=np.array(data_p,dtype=np.float32)
start=960
data_p2=np.array(data_p[960+start:],dtype=np.float32)
data_p=np.array(data_p[0:960+start],dtype=np.float32)
weight=tf.Variable(tf.random_normal([len(All_Fx),1]),dtype=tf.float32,)

weight_abs=tf.square(weight)
sum=tf.reduce_sum(weight_abs,reduction_indices=[0])
w=weight_abs/sum
v=tf.matmul(data_p,w)

loss_mean=tf.reduce_mean(v)

loss=tf.reduce_sum(tf.square(v+200),axis=0)
train_step=tf.train.AdadeltaOptimizer(0.5).minimize(loss)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(10000):
        sess.run(train_step)
        if i%2000==0:
            print(sess.run(loss))

    weight_result=sess.run(w)
    weight_result=np.round(weight_result,2)
    print(weight_result)
    a=np.matmul(data_p,weight_result)
    b=np.matmul(data_p_v,weight_result)
    print(a.max(),a.min(),a.mean(),a[-1])

    print('input double mean_value=%.2f;' %a.mean())
    for i in range(len(All_Fx)):
        if weight_result[i][0]!=0:
            print('input double %s_LOT=%.2f;' %(All_Fx[i], weight_result[i][0]*type[i]))

    # print('input double %s_LOT=%.2f;' %(All_Fx[0], weight_result[0][0]))
    # print('input double %s_LOT=%.2f;' %(All_Fx[1], weight_result[1][0]))
    # print('input double %s_LOT=%.2f;' %(All_Fx[2], weight_result[2][0]))
    # print('input double %s_LOT=%.2f;' %(All_Fx[3], weight_result[3][0]))
    # print('input double %s_LOT=%.2f;' %(All_Fx[4], weight_result[4][0]))
    # print('input double %s_LOT=%.2f;' %(All_Fx[5], weight_result[5][0]))
    # print('input double %s_LOT=%.2f;' %(All_Fx[6], weight_result[6][0]))
    # print('input double %s_LOT=%.2f;' %(All_Fx[7], weight_result[7][0]))
    # print('input double %s_LOT=%.2f;' %(All_Fx[8], weight_result[8][0]))
    # print('input double %s_LOT=%.2f;' %(All_Fx[9], weight_result[9][0]))
    # print('input double %s_LOT=%.2f;' %(All_Fx[10], weight_result[10][0]))
    # print('input double %s_LOT=%.2f;' %(All_Fx[11], weight_result[11][0]))
    # print('input double %s_LOT=%.2f;' %(All_Fx[12], weight_result[12][0]))
    # print('input double %s_LOT=%.2f;' %(All_Fx[13], weight_result[13][0]))
    # print('input double %s_LOT=%.2f;' %(All_Fx[14], weight_result[14][0]))
    # print('input double %s_LOT=%.2f;' %(All_Fx[15], weight_result[15][0]))
    # print('input double %s_LOT=%.2f;' %(All_Fx[16], weight_result[16][0]))
    # print('input double %s_LOT=%.2f;' %(All_Fx[17], weight_result[17][0]))
    # print('input double %s_LOT=%.2f;' %(All_Fx[18], weight_result[18][0]))
    # print('input double %s_LOT=%.2f;' %(All_Fx[19], weight_result[19][0]))
    # print('input double %s_LOT=%.2f;' %(All_Fx[20], weight_result[20][0]))
    # print('input double %s_LOT=%.2f;' %(All_Fx[21], weight_result[21][0]))
    # print('input double %s_LOT=%.2f;' %(All_Fx[22], weight_result[22][0]))
    # print('input double %s_LOT=%.2f;' %(All_Fx[23], weight_result[23][0]))
    # print('input double %s_LOT=%.2f;' %(All_Fx[24], weight_result[24][0]))
    # print('input double %s_LOT=%.2f;' %(All_Fx[25], weight_result[25][0]))
    # print('input double %s_LOT=%.2f;' %(All_Fx[26], weight_result[26][0]))
    # print('input double %s_LOT=%.2f;' %(All_Fx[27], weight_result[27][0]))
    # print('input double %s_LOT=%.2f;' %(All_Fx[28], weight_result[28][0]))
    lot_string=""
    sym_string=""
    type_string=""
    sym_num=0
    for i in range(len(All_Fx)):
        if weight_result[i][0]!=0 and i!=len(All_Fx)-1:
            lot_string=lot_string+str(weight_result[i][0])+","
            sym_string=sym_string+"\""+All_Fx[i]+"\""+","
            type_string=type_string+str(type[i])+","
            sym_num=sym_num+1

        if weight_result[i][0]!=0 and i==len(All_Fx)-1:
            lot_string=lot_string+str(weight_result[i][0])
            sym_string = sym_string +"\""+ All_Fx[i]+"\""
            type_string = type_string + str(type[i])
            sym_num = sym_num + 1



    print('#define N %d' %(sym_num))
    print('string fx[%d]={%s};' %(sym_num,sym_string))
    print('double lot[%d]={%s};' %(sym_num,lot_string))
    print('int type[%d]={%s};' %(sym_num,type_string))


    print(time.time()-t1)

    plt.plot(range(len(a)),a,range(len(a),len(a)+len(b)),b)
    # plt.plot(range(len(a)),a,range(len(a),len(a)+len(b)),b)
    plt.grid()
    plt.show()

