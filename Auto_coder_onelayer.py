import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

learning_rate=0.01
train_epochs=10
batch_size=128
dispaly_step=1
examples_to_show=10

n_hidden_units=256
n_input_units=784
n_output_units=n_input_units

def variable_summary(var,name_str='summaries'):
    with tf.name_scope(name_str):
        mean=tf.reduce_mean(var)
        tf.summary.scalar('mean',mean)
        with tf.name_scope('stddev'):
            stddev=tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
            tf.summary.scalar('stddev',stddev)
            tf.summary.scalar('max',tf.reduce_max(var))
            tf.summary.scalar('min',tf.reduce_min(var))
            tf.summary.histogram('histogram', var)


def WeightsVarible(n_in,n_out,name_str):
    return tf.Variable(tf.random_normal([n_in,n_out]),dtype=tf.float32,name=name_str)


def BaiseVariable(n_out,name_str):
    return tf.Variable(tf.random_normal([n_out]),dtype=tf.float32,name=name_str)


def Encoder(x_origin,active_func=tf.nn.sigmoid):
    with tf.name_scope('Layer'):
        weights=WeightsVarible(n_input_units,n_hidden_units,'in_weights')
        baise  =BaiseVariable(n_hidden_units,'in_baises')
        x_code=active_func(tf.add(tf.matmul(x_origin,weights),baise))
        variable_summary(weights,'Encoder_Layer')
        variable_summary(baise,'Encoder_Layer')
    return x_code


def Decoder(x_code,active_func=tf.nn.sigmoid):
    with tf.name_scope('Layer'):
        weights=WeightsVarible(n_hidden_units,n_output_units,name_str='weights')
        baise  =BaiseVariable(n_output_units,name_str='baises')
        x_decode=active_func(tf.add(tf.matmul(x_code,weights),baise))
        variable_summary(weights,'Decoder_Layer')
        variable_summary(baise,'Decoder_Layer')
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
    with tf.name_scope('LossSummary'):
        tf.summary.scalar('loss',Loss)
        tf.summary.scalar('learning_rate',learning_rate)
    with tf.name_scope('Image_summary'):
        image_origin=tf.reshape(X_origin,[-1,28,28,1])
        image_reconstrution=tf.reshape(x_decode,[-1,28,28,1])
        tf.summary.image('Image_origin',image_origin,10)
        tf.summary.image('Imge_reconstruction',image_reconstrution,10)


    merged_summary=tf.summary.merge_all()

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
                summary_str=sess.run(merged_summary,feed_dict={X_origin:batch_xs})
                summary.add_summary(summary_str,epoch)
                summary.flush()

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

