import numpy as np
import tensorflow as tf
import Tkinter as tk
import matplotlib.pyplot as plt
import pickle as pl
import shutil
import csv

x_data=np.random.rand(100,2).astype(np.float32)
y_data=x_data*[0.1,0.3]+[0.1,2]

Weight=tf.Variable(tf.random_uniform([2],-1,1))
print Weight
biase=tf.Variable(tf.zeros([2]))
print biase

y=Weight*x_data+biase

loss=tf.reduce_mean(tf.square(y-y_data))
optimizer=tf.train.GradientDescentOptimizer(0.5)
train=optimizer.minimize(loss)

init=tf.global_variables_initializer()

sess=tf.Session()
sess.run(init)

for step in range(201):
    sess.run(train)
    if step%20==0:
        print step,sess.run(Weight),sess.run(biase)
