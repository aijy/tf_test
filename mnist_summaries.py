import tensorflow as tf
import mnist_loader
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

training_data,validation_data,test_data=mnist_loader.load_data_wrapper()
x_train=[]
y_train=[]
x_test=[]
y_test=[]
for each in training_data:
    a=each[0].transpose()
    x_train.append(a[0])
    b=each[1].transpose()
    y_train.append(b[0])
for each in test_data:
    a=each[0].transpose()
    x_test.append(a[0])
    y_test.append(each[1])
training_data=zip(x_train,y_train)

sess=tf.InteractiveSession()
x=tf.placeholder(tf.float32,[None,784],name='x-input')
w=tf.Variable(tf.zeros([784,10]),name='weight')
b=tf.Variable(tf.zeros([10]),name='bias')

with tf.name_scope('wx_b'):
    y=tf.nn.softmax(tf.matmul(x,w)+b)
tf.summary.histogram('weight',w)
tf.summary.histogram('bias',b)
tf.summary.histogram('y',y)

y_=tf.placeholder(tf.float32,[None,10],name='y-input')

with tf.name_scope('loss'):
    cross_entropy=-tf.reduce_mean(tf.reduce_sum(y_*tf.log(y)))
    tf.summary.scalar('loss',cross_entropy)
with tf.name_scope('train'):
    train_step=tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

with tf.name_scope('test'):
    correct_prediction=tf.equal(tf.argmax(y_,1),tf.argmax(y,1))
    acc=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    tf.summary.scalar('accuracy',acc)
merge=tf.summary.merge_all()
writer=tf.summary.FileWriter('logs',sess.graph)
sess.run(tf.global_variables_initializer())

np.random.shuffle(training_data)
mini_batches = [
    training_data[k:k + 100]
    for k in xrange(0, len(training_data), 100)]
for i in range(10):
    for mini_batch in mini_batches:
        for batch_xs,batch_ys in mini_batch:
            sess.run([train_step,merge],feed_dict={x:[batch_xs],y_:[batch_ys]})

print 'accuracy:', sess.run(acc, feed_dict={x: x_test, y_: y_test})





