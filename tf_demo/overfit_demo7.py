import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.preprocessing import LabelBinarizer
import numpy as np

keep_prob=tf.placeholder(tf.float32)
def add_layer(inputs,insize,outsize,n_layer,activation_function=None):
    layer_name='layer%s'%n_layer
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            Weights=tf.Variable(tf.random_normal([insize,outsize]),name='W')
            tf.summary.histogram(layer_name+'/weights',Weights)
        with tf.name_scope('baise'):
            baise=tf.Variable(tf.zeros([1,outsize])+0.1,name='B')
            tf.summary.histogram(layer_name + '/baise', Weights)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b=tf.add(tf.matmul(inputs,Weights),baise)
            Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
        if activation_function is None:
            outputs=Wx_plus_b
        else:
            outputs=activation_function(Wx_plus_b)
        tf.summary.histogram(layer_name+'/outputs',outputs)
        return outputs

with tf.name_scope('inputs'):
    xs=tf.placeholder(tf.float32,[None,64],name='x_in')
    ys=tf.placeholder(tf.float32,[None,10],name='y_in')

digits=load_digits()
X=digits.data
y=digits.target
y=LabelBinarizer().fit_transform(y)

x_train=X[0:1000]
y_train=y[0:1000]
x_test=X[1000:]
y_test=y[1000:]
l1=add_layer(xs,64,50,'l1',activation_function=tf.nn.tanh)
prediction=add_layer(l1,50,10,'l2',activation_function=tf.nn.softmax)
def accuracy(x_input,y_input):
    global prediction
    pre_prediction=sess.run(prediction,feed_dict={xs:x_input,keep_prob:1})
    correct=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pre_prediction,1),tf.argmax(y_input,1)),tf.float32))
    return 'right ratio:',sess.run(correct,feed_dict={xs:x_input,keep_prob:1})


with tf.name_scope('loss'):
    cross_entropy=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))
    tf.summary.scalar('loss',cross_entropy)


with tf.name_scope('train'):
    train_step=tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


with tf.Session() as sess:
    merged=tf.summary.merge_all()
    writer=tf.summary.FileWriter('logs/',sess.graph)
    sess.run(tf.global_variables_initializer())
    for i in range(5000):
        sess.run(train_step,feed_dict={xs:x_train,ys:y_train,keep_prob:0.5})
        if i%500==0:
            print sess.run(cross_entropy,feed_dict={xs:x_train,ys:y_train,keep_prob:1})
    print accuracy(x_test,y_test)






