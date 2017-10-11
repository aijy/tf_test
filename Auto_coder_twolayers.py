import numpy as np
import tensorflow as tf
import sklearn.preprocessing as prep
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

learning_rate=0.01
train_epochs=20
batch_size=256
dispaly_step=1
examples_to_show=10

n_hidden_units_1=256
n_hidden_units_2=128
n_hidden_out_units_1=n_hidden_units_2
n_hidden_out_units_2=n_hidden_units_1
n_input_units=784
n_output_units=n_input_units


def WeightsVarible(n_in,n_out,name_str):
    return tf.Variable(tf.random_normal([n_in,n_out]),dtype=tf.float32,name=name_str)


def BaiseVariable(n_out,name_str):
    return tf.Variable(tf.random_normal([n_out]),dtype=tf.float32,name=name_str)


def Encoder(x_origin,active_func=tf.nn.sigmoid):
    with tf.name_scope('Layer1'):
        weights_1=WeightsVarible(n_input_units,n_hidden_units_1,'weights')
        baise_1  =BaiseVariable(n_hidden_units_1,'in_baises')
        x_code_1=active_func(tf.add(tf.matmul(x_origin,weights_1),baise_1))
    with tf.name_scope('Layer2'):
        weights_2=WeightsVarible(n_hidden_units_1,n_hidden_units_2,'weights')
        baise_2  =BaiseVariable(n_hidden_units_2,'in_baises')
        x_code=active_func(tf.add(tf.matmul(x_code_1,weights_2),baise_2))
    return x_code


def Decoder(x_code,active_func=tf.nn.sigmoid):
    with tf.name_scope('Layer1'):
        weights_1=WeightsVarible(n_hidden_out_units_1,n_hidden_out_units_2,name_str='weights')
        baise_1  =BaiseVariable(n_hidden_out_units_2,name_str='baises')
        x_decode_1=active_func(tf.add(tf.matmul(x_code,weights_1),baise_1))
    with tf.name_scope('Layer2'):
        weights_2=WeightsVarible(n_hidden_out_units_2,n_output_units,name_str='weights')
        baise_2  =BaiseVariable(n_output_units,name_str='baises')
        x_decode=active_func(tf.add(tf.matmul(x_decode_1,weights_2),baise_2))
    return x_decode

with tf.Graph().as_default():
    with tf.name_scope('X_origin'):
        X_origin=tf.placeholder(tf.float32,[None,n_input_units])
    with tf.name_scope('Encoder'):
        x_code=Encoder(X_origin,active_func=tf.nn.sigmoid)
    with tf.name_scope('Decoder'):
        x_decode=Decoder(x_code,active_func=tf.nn.sigmoid)
    with tf.name_scope('Loss'):
        Loss=tf.reduce_mean(tf.pow(X_origin-x_decode,2))
    with tf.name_scope('Train'):
        train_step=tf.train.RMSPropOptimizer(learning_rate).minimize(Loss)
    # with tf.name_scope()


    print('write summary to file ....')

    summary=tf.summary.FileWriter(logdir='logs1',graph=tf.get_default_graph())
    summary.flush()

    mnist=input_data.read_data_sets('../MNIST_data/',one_hot=True)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        total_batch=int(mnist.train.num_examples/batch_size)
        for epoch in range(train_epochs):
            for i in range(total_batch):
                batch_xs,batch_ys=mnist.train.next_batch(batch_size)
                _,loss=sess.run([train_step,Loss],feed_dict={X_origin:batch_xs})

            if epoch%dispaly_step==0:
                print("Epoch=",epoch+1,"  Loss=",loss)
        reconstructions=sess.run(x_decode,feed_dict={X_origin:mnist.test.images[:examples_to_show]})

        f,a=plt.subplots(2,10,figsize=(10,2))
        for i in range(examples_to_show):
            a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
            a[1][i].imshow(np.reshape(reconstructions[i],(28,28)))
        f.show()
        plt.draw()
        plt.waitforbuttonpress()

        summary.close()
        print('this is the end.....')

