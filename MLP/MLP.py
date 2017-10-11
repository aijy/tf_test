# /usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
t1=time.time()

input_dot=784
hidden_dot=500
# hidden_dot1=500
output_dot=10
epoch=10000
batch_size=100
learning_rate=0.1
show_example_num=10
keep_dot_value=0.5

g1=tf.Graph()
g2=tf.Graph()

def variable_summary(var,name_str='summaries'):
    with tf.name_scope(name_str):
        mean=tf.reduce_mean(var)
        tf.summary.scalar('mean',mean)
        # with tf.name_scope('stddev'):
        #     stddev=tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        #     tf.summary.scalar('stddev',stddev)
        #     tf.summary.scalar('max',tf.reduce_max(var))
        #     tf.summary.scalar('min',tf.reduce_min(var))
        #     tf.summary.histogram('histogram', var)

with g1.as_default():
    with tf.name_scope('Input_Layer'):
        xs=tf.placeholder(tf.float32,[None,input_dot],name='in_x')
        ys=tf.placeholder(tf.float32,[None,output_dot],name='in_y')
        keep_dot=tf.placeholder(tf.float32)
    with tf.name_scope('Hidden_Layer1'):
        weight = tf.Variable(tf.truncated_normal([input_dot, hidden_dot], stddev=0.1, dtype=tf.float32), name='w')
        baise = tf.Variable(tf.fill([hidden_dot], 0.1), name='b')
        h1_b=tf.nn.relu(tf.matmul(xs,weight)+baise)
        h1  =tf.nn.dropout(h1_b,keep_dot)
    # with tf.name_scope('Hidden_Layer2'):
    #     weight = tf.Variable(tf.truncated_normal([hidden_dot, hidden_dot1], stddev=0.1, dtype=tf.float32), name='w')
    #     baise = tf.Variable(tf.fill([hidden_dot1], 0.1), name='b')
    #     h2_b=tf.nn.relu(tf.matmul(h1,weight)+baise)
    #     h2  =tf.nn.dropout(h2_b,keep_dot)
    with tf.name_scope('Output_Layer'):
        weight_1 = tf.Variable(tf.truncated_normal([hidden_dot, output_dot], stddev=0.1, dtype=tf.float32), name='w1')
        baise_1 = tf.Variable(tf.fill([output_dot], 0.1), name='b1')
        yp=tf.nn.softmax(tf.matmul(h1,weight_1)+baise_1)
    with tf.name_scope('Loss'):
        loss=-tf.reduce_mean(tf.reduce_sum(ys*tf.log(yp),axis=1))
        variable_summary(loss,'softmax_loss')
    with tf.name_scope('Train'):
        train_step =tf.train.AdagradOptimizer(learning_rate).minimize(loss)
        tf.summary.scalar('learning_rate',learning_rate)
    with tf.name_scope('Acc'):
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(ys,1),tf.argmax(yp,1)),dtype=tf.float32))
        variable_summary(accuracy,'softmax_acc')
    with tf.name_scope('Image_summary'):
        image_origin=tf.reshape(xs,[-1,28,28,1])
        tf.summary.image('Image_origin',image_origin,10)

    summary_merged=tf.summary.merge_all()
    summary=tf.summary.FileWriter(logdir='MLP',graph=g1)
    summary.flush()

mnist = input_data.read_data_sets('../MNIST_data/', one_hot=True)

with tf.Session(graph=g1) as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(epoch):
        batch_xs,batch_ys=mnist.train.next_batch(batch_size)
        _,sof_loss=sess.run([train_step,loss],feed_dict={xs:batch_xs,ys:batch_ys,keep_dot:keep_dot_value})
        if i%100==0:
            # print('sigmoid_loss:%.4f,softmax_loss:%.4f' %(sig_loss,sof_loss))
            acc_1=sess.run(accuracy, feed_dict={xs: mnist.test.images, ys: mnist.test.labels,keep_dot:1})
            print('softmax_scc:%.4f' %(acc_1,))
        if i%1000==0:
            summary_str = sess.run(summary_merged, feed_dict={xs:batch_xs,ys:batch_ys,keep_dot:1})
            summary.add_summary(summary_str, i)
            summary.flush()


