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
output_dot=10
epoch=10000
batch_size=100
learning_rate=0.1
show_example_num=10

g1=tf.Graph()
g2=tf.Graph()

def variable_summary(var,name_str='summaries'):
    with tf.name_scope(name_str):
        mean=tf.reduce_mean(var)
        tf.summary.scalar('mean',mean)
        with tf.name_scope('stddev'):
            stddev=tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
            tf.summary.scalar('stddev',stddev)
            # tf.summary.scalar('max',tf.reduce_max(var))
            # tf.summary.scalar('min',tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

with g1.as_default():
    with tf.name_scope('Input'):
        xs=tf.placeholder(tf.float32,[None,input_dot],name='in_x')
        ys=tf.placeholder(tf.float32,[None,output_dot],name='in_y')
    with tf.name_scope('Layer'):
        weight = tf.Variable(tf.truncated_normal([input_dot, output_dot], stddev=0.1, dtype=tf.float32), name='w')
        baise = tf.Variable(tf.fill([output_dot], 0.1), name='b')
        weight_1 = tf.Variable(tf.truncated_normal([input_dot, output_dot], stddev=0.1, dtype=tf.float32), name='w1')
        baise_1 = tf.Variable(tf.fill([output_dot], 0.1), name='b1')
        yp_1=tf.nn.softmax(tf.matmul(xs,weight_1)+baise_1)
        yp=tf.nn.sigmoid(tf.matmul(xs,weight)+baise)
    with tf.name_scope('Loss'):
        loss_1=-tf.reduce_mean(tf.reduce_sum(ys*tf.log(yp_1),axis=[1]))
        loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-yp),axis=1))
        variable_summary(loss,'sigmoid_loss')
        variable_summary(loss_1,'softmax_loss')
    with tf.name_scope('Train'):
        train_step =tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
        train_step_1 = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_1)
        tf.summary.scalar('learning_rate',learning_rate)
    with tf.name_scope('Acc'):
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(ys,1),tf.argmax(yp,1)),dtype=tf.float32))
        accuracy_1 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(ys, 1), tf.argmax(yp_1, 1)), dtype=tf.float32))
        variable_summary(accuracy,'sigmoid_acc')
        variable_summary(accuracy_1,'softmax_acc')
    with tf.name_scope('Image_summary'):
        image_origin=tf.reshape(xs,[-1,28,28,1])
        tf.summary.image('Image_origin',image_origin,10)

    summary_merged=tf.summary.merge_all()
    summary=tf.summary.FileWriter(logdir='LP',graph=g1)
    summary.flush()

mnist = input_data.read_data_sets('../MNIST_data/', one_hot=True)

with tf.Session(graph=g1) as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(epoch):
        batch_xs,batch_ys=mnist.train.next_batch(batch_size)
        a,b,sig_loss,sof_loss=sess.run([train_step,train_step_1,loss,loss_1],feed_dict={xs:batch_xs,ys:batch_ys})
        if i%100==0:
            # print('sigmoid_loss:%.4f,softmax_loss:%.4f' %(sig_loss,sof_loss))
            acc_1,acc_2=sess.run([accuracy,accuracy_1], feed_dict={xs: mnist.test.images, ys: mnist.test.labels})
            print('sigmoid_acc:%.4f,softmax_scc:%.4f' %(acc_1, acc_2))
        # if i%10==0:
        #     summary_str = sess.run(summary_merged, feed_dict={xs:batch_xs,ys:batch_ys})
        #     summary.add_summary(summary_str, i)
        #     summary.flush()


