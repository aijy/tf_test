import numpy as np
import tensorflow as tf
import sklearn.preprocessing as prep
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

learning_rate_init=0.01
training_epochs=10
batch_size=100
display_step=2

n_input=784
n_output=10

def WeightsVarible(shape,name_str,stddev=0.1):
    initial=tf.random_normal(shape=shape,stddev=stddev,dtype=tf.float32)
    return tf.Variable(initial,dtype=tf.float32,name=name_str)

def BaiseVariable(shape,name_str,stddev=0.0001):
    initial=tf.random_normal(shape=shape,stddev=stddev,dtype=tf.float32)
    return tf.Variable(initial,dtype=tf.float32,name=name_str)

def Conv2d(x,W,b,stride,padding='SAME'):
    with tf.name_scope('wx_b'):
        y=tf.nn.conv2d(x,W,strides=[1,stride,stride,1],padding=padding)
        y=tf.nn.bias_add(y,b)
    return y

def Activation(x,activation=tf.nn.relu,name='relu'):
    with tf.name_scope(name):
        y=activation(x)
    return y
def Pool2d(x,pool=tf.nn.max_pool,k=2,stride=2):
    return pool(x,ksize=[1,k,k,1],strides=[1,stride,stride,1],padding='VALID')


def FullyConnected(x,W,b,activation=tf.nn.relu,act_name='relu'):
    with tf.name_scope('wx_b'):
        y=tf.add(tf.matmul(x,W),b)
    with tf.name_scope(act_name):
        y=activation(y)
    return y

with tf.Graph().as_default():
    with tf.name_scope('Inputs'):
        X_origin=tf.placeholder(tf.float32,[None,n_input],name='X_origin')
        y_true=tf.placeholder(tf.float32,[None,n_output],name="Y_true")
        x_image=tf.reshape(X_origin,[-1,28,28,1])

    with tf.name_scope('Inference'):
        with tf.name_scope('Conv2d'):
            weights=WeightsVarible(shape=[5,5,1,16],name_str='weights')
            bias   =BaiseVariable(shape=[16],name_str='biase')
            conv_out=Conv2d(x_image,weights,bias,stride=1,padding='VALID')

        with tf.name_scope('Activate'):
            activate_out=Activation(conv_out,activation=tf.nn.relu,name='relu')

        with tf.name_scope('Pool2d'):
            pool_out=Pool2d(activate_out,pool=tf.nn.max_pool,k=2,stride=2)

        with tf.name_scope('FeatsReshape'):
            features=tf.reshape(pool_out,[-1,12*12*16])

        with tf.name_scope('FC_layer'):
            weights=WeightsVarible(shape=[12*12*16,n_output],name_str='weight')
            bias=BaiseVariable(shape=[n_output],name_str='bias')
            Ypred_logits=FullyConnected(features,weights,bias,activation=tf.identity,act_name='identy')


    with tf.name_scope('Loss'):
        cross_entropy_loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=Ypred_logits))
        tf.summary.scalar('Loss',cross_entropy_loss)
    with tf.name_scope('Train'):
        learning_rate=tf.placeholder(tf.float32)
        optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate)
        trainer=optimizer.minimize(cross_entropy_loss)

    with tf.name_scope('Evaluate'):
        correc_pred=tf.equal(tf.argmax(Ypred_logits,1),tf.argmax(y_true,1))
        accuracy=tf.reduce_mean(tf.cast(correc_pred,tf.float32))

    init=tf.global_variables_initializer()
    merged_summary=tf.summary.merge_all()
    print 'write graph ......'

    summary_writer=tf.summary.FileWriter(logdir='logs2',graph=tf.get_default_graph())
    summary_writer.close()

    print 'write graph ......done'

    mnist=input_data.read_data_sets('../MNIST_data/',one_hot=True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        total_batch=int(mnist.train.num_examples/batch_size)
        for epoch in range(training_epochs):
            for i in range(total_batch):
                batch_xs,batch_ys=mnist.train.next_batch(batch_size)
                _,loss=sess.run([trainer,cross_entropy_loss],feed_dict={X_origin:batch_xs,y_true:batch_ys,learning_rate:learning_rate_init})
                if loss<0.005:
                    break;

            if epoch%display_step==0:
                print "Epoch=",epoch+1,"  Loss=",loss
                # summary_str=sess.run(merged_summary,feed_dict={X_origin:batch_xs})
                # summary_writer.add_summary(summary_str,epoch)
                # summary_writer.flush()
        print sess.run(accuracy,feed_dict={X_origin:mnist.test.images,y_true:mnist.test.labels,learning_rate:learning_rate_init})

