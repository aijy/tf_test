import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

input_1=tf.placeholder(tf.float32)
input_2=tf.placeholder(tf.float32)

output_add=tf.add(input_1,input_2)
output_minus=tf.subtract(input_1,input_2)
output_multiply=tf.multiply(input_1,input_2)
output_divede=tf.div(input_1,input_2)


with tf.Session() as sess:
    print sess.run(output_add,feed_dict={input_1:[7,7,7],input_2:[3,7,9]})
    print sess.run(output_minus,feed_dict={input_1:[7,7,7],input_2:[3,7,9]})
    print sess.run(output_multiply, feed_dict={input_1: [7, 7, 7], input_2: [3, 7, 9]})
    print sess.run(output_divede, feed_dict={input_1: [7, 7, 7], input_2: [3, 7, 9]})














