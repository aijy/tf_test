import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(inputs,insize,outsize,activation_function=None):
    Weights=tf.Variable(tf.random_normal([insize,outsize]))
    baise=tf.Variable(tf.zeros([1,outsize])+0.1)
    Wx_plus_b=tf.matmul(inputs,Weights)+baise
    if activation_function is None:
        outputs=Wx_plus_b
    else:
        outputs=activation_function(Wx_plus_b)
    return outputs

x_data=np.linspace(-1,1,300,dtype=np.float32)[:,np.newaxis]
noise=np.random.normal(0,0.01,x_data.shape).astype(np.float32)
y_data=np.sin(x_data*5)-0.5+noise

xs=tf.placeholder(tf.float32,[None,1])
ys=tf.placeholder(tf.float32,[None,1])

l1=add_layer(xs,1,50,activation_function=tf.nn.tanh)
prediction=add_layer(l1,50,1,activation_function=tf.nn.tanh)
loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))

train_step=tf.train.GradientDescentOptimizer(0.02).minimize(loss)

init=tf.global_variables_initializer()

fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
plt.ion()
plt.show()

with tf.Session() as sess:
    sess.run(init)
    for i in range(20000):
        sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
        if i%50==0:
            print i
            try:
                ax.lines.remove(lines[0])
            except:
                pass
            prediction_value=sess.run(prediction,feed_dict={xs:x_data,ys:y_data})
            lines=ax.plot(x_data,prediction_value,'r-',lw=6)
            plt.pause(0.5)
            plt.show()




