import numpy as np
import tensorflow as tf
import sklearn.preprocessing as prep
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def xavier_init(fan_in,fan_out,constant=1):
    low=-constant*np.sqrt(6.0/(fan_in+fan_out))
    high=constant*np.sqrt(6.0/(fan_in+fan_out))
    return tf.random_uniform((fan_in,fan_out),minval=low,maxval=high,dtype=tf.float32)

class AdditiveGaussianNoiseAutoencoder(object):
    def __init__(self,n_input,n_hidden,transfer_function=tf.nn.sigmoid,optimizer=tf.train.AdadeltaOptimizer(),scale=0.1):
        self.n_input=n_input
        self.n_hidden=n_hidden
        self.transfer=transfer_function
        self.scale=tf.placeholder(tf.float32)
        self.training_scale=scale
        self.weights=dict()
        self.sess=tf.Session()

        with tf.name_scope('input_layer'):
            self.x=tf.placeholder(tf.float32,[None,self.n_input])
        with tf.name_scope('NoiseAdder'):
            self.scale=tf.placeholder(tf.float32)
            self.noise_x=self.x+self.scale*tf.random_normal([self.n_input,])
        with tf.name_scope('Encoder'):
            self.weights['w1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden), name='weight1')
            self.weights['b1'] = tf.Variable(tf.zeros([self.n_hidden]), dtype=tf.float32, name='bias1')
            self.hidden=self.transfer(tf.add(tf.matmul(self.noise_x,self.weights['w1']),self.weights['b1']))
        with tf.name_scope('Reconstruction'):
            self.weights['w2'] = tf.Variable(tf.zeros([self.n_hidden,self.n_input]),dtype=tf.float32,name='weight2')
            self.weights['b2'] = tf.Variable(tf.zeros([self.n_input]),dtype=tf.float32,name='bias2')
            self.reconstruction=self.transfer(tf.add(tf.matmul(self.hidden,self.weights['w2']),self.weights['b2']))
        with tf.name_scope('Loss'):
            self.cost=tf.reduce_mean(tf.pow(tf.subtract(self.reconstruction,self.x),2))
        with tf.name_scope('Train'):
            self.optimizer=optimizer.minimize(self.cost)
        self.sess.run(tf.global_variables_initializer())
        print('begin to run session...')

    def partial_fit(self,X):
        cost,opt=self.sess.run([self.cost,self.optimizer],feed_dict={self.x:X,self.scale:self.training_scale})
        return cost

    def calc_total_cost(self,X):
        return self.sess.run(self.cost,feed_dict={self.x:X,self.scale:self.training_scale})

    def generate(self,hidden=None):
        if hidden==None:
            hidden=np.random.normal(size=self.weights['b1'])
        return self.sess.run(self.reconstruction,feed_dict={self.hidden:hidden})

    def reconstruction_1(self,X):
        return self.sess.run(self.reconstruction,feed_dict={self.x:X,self.scale:self.training_scale})

    def getWeights(self):
        return self.sess.run(self.weights['w1'])

    def getBiases(self):
        return self.sess.run(self.weights['b1'])

AGN_AC=AdditiveGaussianNoiseAutoencoder(n_input=784,n_hidden=256,transfer_function=tf.nn.sigmoid,optimizer=tf.train.RMSPropOptimizer(learning_rate=0.005),scale=0.1)

writer=tf.summary.FileWriter(logdir='logs1',graph=AGN_AC.sess.graph)
writer.close()

print('tensorboard is ready....')

mnist=input_data.read_data_sets('../MNIST_data/',one_hot=True)

def standard_scale(X_train,X_test):
    preprocessor=prep.StandardScaler().fit(X_train)
    X_train=preprocessor.transform(X_train)
    X_test=preprocessor.transform(X_test)
    return X_train,X_test

def get_random_block_from_data(data,batch_size):
    start_index=np.random.randint(0,len(data)-batch_size)
    return data[start_index:start_index+batch_size]

X_train,X_test=standard_scale(mnist.train.images,mnist.test.images)


n_sample=int(mnist.train.num_examples)
training_epochs=20
batch_size=128
display_step=1

for epoch in range(training_epochs):
    avg_cost=0
    total_batch=int(n_sample/batch_size)
    for i in range(total_batch):
        batch_xs=get_random_block_from_data(X_train,batch_size)
        cost=AGN_AC.partial_fit(batch_xs)
    if cost<0.05:
        break;
    if epoch%display_step==0:
        print('epoch:%04d,cost:%.9f' %(epoch+1,cost))


f, a = plt.subplots(2, 10, figsize=(10, 2))
for i in range(10):
    a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
    a[1][i].imshow(np.reshape(AGN_AC.reconstruction_1([X_test[i]]), (28, 28)))
f.show()
plt.draw()
plt.waitforbuttonpress()

