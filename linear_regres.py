import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

with tf.Graph().as_default():
    x=tf.placeholder(tf.float32)
    y=tf.placeholder(tf.float32)

    w = tf.Variable(tf.zeros([1]))
    b = tf.Variable(tf.zeros([1]))




    writer=tf.summary.FileWriter(logdir='logs',graph=tf.get_default_graph())
    writer.close()