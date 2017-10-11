import tensorflow as tf
import mnist_loader
import numpy as np

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
print training_data

xs = tf.placeholder(tf.float32,[None,784])
ys = tf.placeholder(tf.float32,[None,10])


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

prediction=add_layer(xs,784,10,1,activation_function=tf.nn.softmax)


def compute_accuracy(v_xs,v_ys):
    global prediction
    v_xs=np.array(v_xs)
    y_pre=sess.run(prediction,feed_dict={xs:v_xs})
    correct_pridiction=tf.equal(tf.arg_max(y_pre,1),v_ys)
    accuracy=tf.reduce_mean(tf.cast(correct_pridiction,tf.float32))
    result=sess.run(accuracy,feed_dict={xs:v_xs})
    return result

cross_entropy=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))
train_step=tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    np.random.shuffle(training_data)
    mini_batches = [
        training_data[k:k + 100]
        for k in xrange(0, len(training_data), 100)]
    for i in range(100):
        for mini_batch in mini_batches:
            for batch_xs,batch_ys in mini_batch:
                sess.run(train_step,feed_dict={xs:[batch_xs],ys:[batch_ys]})
        if i % 20 == 0:
            print 'loss:',sess.run(cross_entropy,feed_dict={xs:x_train,ys:y_train})
    print compute_accuracy(x_test, y_test)




