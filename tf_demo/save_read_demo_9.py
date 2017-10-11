import tensorflow as tf
import mnist_loader
import numpy as np


training_data,validation_data,test_data=mnist_loader.load_data_wrapper()
x_train=[]
y_train=[]
x_test=[]
y_test=[]
for each in training_data[0:1000]:
    a=each[0].transpose()
    x_train.append(a[0])
    b=each[1].transpose()
    y_train.append(b[0])
for each in test_data[0:1000]:
    a=each[0].transpose()
    x_test.append(a[0])
    y_test.append(each[1])

training_data=zip(x_train,y_train)
test_data=zip(x_test,y_test)

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

cross_entropy=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))

train_step=tf.train.AdadeltaOptimizer(0.3).minimize(cross_entropy)

def compute_accuracy(v_xs,v_ys):
    global prediction
    v_xs=np.array(v_xs)
    y_pre=sess.run(prediction,feed_dict={xs:v_xs,keep_prob:1})
    correct_pridiction=tf.equal(tf.arg_max(y_pre,1),v_ys)
    accuracy=tf.reduce_mean(tf.cast(correct_pridiction,tf.float32))
    result=sess.run(accuracy,feed_dict={xs:v_xs,keep_prob:1})
    return result


saver=tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10):
        sess.run(train_step,feed_dict={xs:x_train,ys:y_train,keep_prob:0.5})
        if i%5==0:
            print sess.run(cross_entropy,feed_dict={xs:x_train,ys:y_train,keep_prob:1})
    print compute_accuracy(x_test,y_test)
    save_path=saver.save(sess,"my_net/save_net.ckpt")
    print 'save to path: ',save_path
#
# with tf.Session() as sess:
#     saver.restore(sess,"my_net/save_net.ckpt")
#     print "weight1:",sess.run(W_cov1)
#     print 'biase1:',sess.run(b_cov1)
#     print "weight2:",sess.run(W_cov2)
#     print 'biase2:',sess.run(b_cov2)
