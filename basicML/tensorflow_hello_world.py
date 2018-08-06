import tensorflow as tf


c = tf.Constant('hello world!')
with tf.Session as sess:
    print(sess.run(c))