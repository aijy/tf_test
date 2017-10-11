import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

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
        if activation_function is None:
            outputs=Wx_plus_b
        else:
            outputs=activation_function(Wx_plus_b)
        tf.summary.histogram(layer_name+'/outputs',outputs)
        return outputs

x_data=np.linspace(-1,1,300,dtype=np.float32)[:,np.newaxis]
noise=np.random.normal(0,0.01,x_data.shape).astype(np.float32)
y_data=np.square(x_data)-0.5+noise

with tf.name_scope('inputs'):
    xs=tf.placeholder(tf.float32,[None,1],name='x_in')
    ys=tf.placeholder(tf.float32,[None,1],name='y_in')

l1=add_layer(xs,1,10,n_layer=1,activation_function=tf.nn.relu)
prediction=add_layer(l1,10,1,n_layer=2,activation_function=None)

with tf.name_scope('loss'):
    loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))
    tf.summary.scalar('loss',loss)
with tf.name_scope('train'):
    train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init=tf.global_variables_initializer()

# fig=plt.figure()
# ax=fig.add_subplot(1,1,1)
# ax.scatter(x_data,y_data)
# plt.ion()
# plt.show()

with tf.Session() as sess:
    merged=tf.summary.merge_all()
    writer=tf.summary.FileWriter('logs/',sess.graph)
    sess.run(init)
    for i in range(2000):
        sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
        if i%50==0:
            rs=sess.run(merged,feed_dict={xs:x_data,ys:y_data})
            writer.add_summary(rs,i)
            # prediction_value=sess.run(prediction,feed_dict={xs:x_data,ys:y_data})
            # lines=ax.plot(x_data,prediction_value,'r-',lw=5)
            # plt.pause(0.5)
            # plt.show()




