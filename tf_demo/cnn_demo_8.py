import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
mnist = input_data.read_data_sets('../MNIST_data/', one_hot=True)
batch_size=50

def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_poo_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

xs=tf.placeholder(tf.float32,[None,784])
ys=tf.placeholder(tf.float32,[None,10])
keep_prob=tf.placeholder(tf.float32)

x_image=tf.reshape(xs,[-1,28,28,1])

W_cov1=weight_variable([5,5,1,32])
b_cov1=bias_variable([32])
h_cov1=tf.nn.relu(conv2d(x_image,W_cov1)+b_cov1)
h_pool1=max_poo_2x2(h_cov1)

W_cov2=weight_variable([5,5,32,64])
b_cov2=bias_variable([64])
h_cov2=tf.nn.relu(conv2d(h_pool1,W_cov2)+b_cov2)
h_pool2=max_poo_2x2(h_cov2)

h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])

W_fc1=weight_variable([7*7*64,1024])
b_fc1=bias_variable([1024])
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)

W_fc2=weight_variable([1024,10])
b_fc2=bias_variable([10])
prediction=tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)

cross_entropy=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),axis=1))
train_step=tf.train.AdadeltaOptimizer(0.1).minimize(cross_entropy)
accuracy=tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(prediction,1),tf.argmax(ys,1)),tf.float32))


with tf.Session() as sess:
    # with tf.device('/cpu:0'):
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys,keep_prob:0.5})
        if i%100==0:
            # print(sess.run(cross_entropy,feed_dict={xs:batch_xs,ys:batch_ys,keep_prob:1}))
            print(sess.run(accuracy,feed_dict={xs:mnist.test.images,ys:mnist.test.labels,keep_prob:1}))






