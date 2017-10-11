import tensorflow as tf
import sklearn.preprocessing as prep
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import csv
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
t1=time.time()
a=pd.read_csv('single_fx_data.csv',dtype=np.float32)
data=pd.DataFrame(a,columns=['EURUSD'])

eur_data_1=np.array(data.values)
eur_data=[]
for each in eur_data_1:
    eur_data.append(each[0])

n=120
long_profit=500
short_profit=500
input_data_length=300
hidden_dot=500

data_train_1=[]
eur_target=[]
for i in range(30000-n-input_data_length):
    data_train_1.append(eur_data[i:i+input_data_length])
    eur_target.append([])
    for j in range(i+input_data_length,i+input_data_length+n):
        if eur_data[j]-eur_data[i+input_data_length]>long_profit:
            eur_target[-1]=[1,0,0] #buy
            break
        if eur_data[j]-eur_data[i+input_data_length]<-short_profit:
            eur_target[-1]=[0, 0,1]  # sell
            break
        eur_target[-1]=[0, 1, 0]  # nothing
data_train_1=np.array(data_train_1)
# print(len(eur_target))
# print(np.sum(eur_target,axis=0))
# exit()
mm=prep.MinMaxScaler()
data_train=np.array(mm.fit_transform(data_train_1.transpose()))
data_train=data_train.transpose()
train_data, test_data, train_target, test_target=train_test_split(data_train[:-50],eur_target[:-50],test_size=0.2,random_state=10)


def get_random_block_from_data(train_data, target_data, batch_size):
    start_index = np.random.randint(0, len(train_data) - batch_size)
    return train_data[start_index:start_index + batch_size],target_data[start_index:start_index + batch_size]

with tf.Graph().as_default():
    with tf.name_scope('Input'):
        xs=tf.placeholder(tf.float32,[None,input_data_length],name='x_in')
        ys=tf.placeholder(tf.float32,[None,3],name='y_in')
        keep_prob=tf.placeholder(tf.float32)
    with tf.name_scope('Layer1'):
        Weights_1 = tf.Variable(tf.truncated_normal([input_data_length,hidden_dot],stddev=0.1))
        baise_1 = tf.Variable(tf.zeros([hidden_dot])+0.1)
        l1 = tf.nn.relu(tf.matmul(xs, Weights_1) + baise_1)
        l1 = tf.nn.dropout(l1, keep_prob=keep_prob)
    with tf.name_scope('Layer2'):
        Weights_2 = tf.Variable(tf.truncated_normal([hidden_dot, 3],stddev=0.1), name='W')
        baise_2 = tf.Variable(tf.zeros([3]) + 0.1, name='B')
        prediction = tf.nn.softmax(tf.matmul(l1, Weights_2) + baise_2)
    with tf.name_scope('Loss'):
        loss=-tf.reduce_mean(tf.reduce_sum(ys*tf.log(prediction),axis=1))
    with tf.name_scope('Train'):
        train_step=tf.train.GradientDescentOptimizer(0.001).minimize(loss)
    with tf.name_scope('Acc'):
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(prediction,1), tf.argmax(ys,1)), dtype=tf.float32))

    summary_merge=tf.summary.merge_all()
    summay=tf.summary.FileWriter(logdir='fx_test',graph=tf.get_default_graph())
    summay.flush()

    init=tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        # print sess.run(baise_1)
        for i in range(200000):
            x_data,y_data=get_random_block_from_data(train_data,train_target,100)
            sess.run(train_step,feed_dict={xs:x_data,ys:y_data,keep_prob:0.6})

            if i%5000==0:
                print(sess.run(accuracy,feed_dict={xs:test_data,ys:test_target,keep_prob:1}))
                print('b2222:',sess.run(baise_2))

        # p_v = np.argmax(sess.run(prediction, feed_dict={xs:data_train[-50:], keep_prob: 1}), axis=1)
        # a_v = np.argmax(eur_target[-50:], axis=1)
        # for e, f in zip(p_v, a_v):
        #     print(e, f)

        print(np.argmax(sess.run(prediction,feed_dict={xs:data_train[-50:],keep_prob:1}),axis=1))
        print(np.argmax(eur_target[-50:],axis=1))

        data_test=[]
        for i in range(28880,29000):
            data_test.append(eur_data[i:i + input_data_length])
        data_test= np.array(data_test)
        data_test = mm.fit_transform(data_test.transpose())
        data_test=data_test.transpose()
        print(np.argmax(sess.run(prediction,feed_dict={xs:data_test,keep_prob:1}),axis=1))
    print(time.time()-t1)