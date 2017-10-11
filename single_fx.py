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
short_profit=330
input_data_length=1000
hidden_dot=[250,100]

data_train_1=[]
eur_target=[]
for i in range(30000-n-input_data_length):
    data_train_1.append(eur_data[i:i+input_data_length])
    eur_target.append([])
    for j in range(i+input_data_length,i+input_data_length+n):
        if eur_data[j]-eur_data[i]>long_profit:
            eur_target[-1]=[1,0,0] #buy
            break
        if eur_data[j]-eur_data[i]<-short_profit:
            eur_target[-1]=[0, 0,1]  # sell
            break
        eur_target[-1]=[0, 1, 0]  # nothing
data_train_1=np.array(data_train_1)
# print(len(eur_target))
# print(np.sum(eur_target,axis=0))
# exit()
mm=prep.MinMaxScaler()
data_train=mm.fit_transform(data_train_1)
train_data, test_data, train_target, test_target=train_test_split(data_train[:-50],eur_target[:-50],test_size=0.25,random_state=10)

Weights_1 = tf.Variable(tf.random_normal([input_data_length,hidden_dot[0]]))
baise_1   = tf.Variable(tf.zeros([1,hidden_dot[0]])+0.1)
Weights_2 = tf.Variable(tf.random_normal([hidden_dot[0], hidden_dot[1]]))
baise_2   = tf.Variable(tf.zeros([1, hidden_dot[1]]) + 0.1)
Weights_3 = tf.Variable(tf.random_normal([hidden_dot[1], 3]))
baise_3   = tf.Variable(tf.zeros([1, 3]) + 0.1)

xs=tf.placeholder(tf.float32,[None,input_data_length],name='x_in')
ys=tf.placeholder(tf.float32,[None,3],name='y_in')

l1=tf.nn.relu(tf.matmul(xs,Weights_1)+baise_1)
l2=tf.nn.relu(tf.matmul(l1,Weights_2)+baise_2)
l2=tf.nn.dropout(l2,keep_prob=0.5)
prediction=tf.nn.softmax(tf.matmul(l2,Weights_3)+baise_3)


loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))

train_step=tf.train.AdadeltaOptimizer(0.5).minimize(loss)


def get_accuracy(test_data,real_value):
    global prediction
    sess.run(prediction,feed_dict={xs:test_data})
    accuracy=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(prediction,1),tf.argmax(real_value,1)),dtype=tf.float32))
    return sess.run(accuracy,feed_dict={xs:test_data,ys:real_value})

init=tf.global_variables_initializer()
batch=0
with tf.Session() as sess:
    sess.run(init)
    # print sess.run(baise_1)
    for i in range(200000):
        x_data=train_data[batch:batch+100]
        y_data=train_target[batch:batch+100]
        batch=(batch+100)%len(train_data)
        sess.run(train_step,feed_dict={xs:x_data,ys:y_data})

        if i%40000==0:
            print(get_accuracy(test_data,test_target))
            # print sess.run(prediction,feed_dict={xs:test_data})

    print(np.argmax(sess.run(prediction,feed_dict={xs:data_train[-50:]}),axis=1))
    print(np.argmax(eur_target[-50:],axis=1))

    data_test=[]
    for i in range(28880,29000):
        data_test.append(eur_data[i:i + input_data_length])
    data_test= mm.fit_transform(data_test)
    print(np.argmax(sess.run(prediction,feed_dict={xs:data_test}),axis=1))
print(time.time()-t1)