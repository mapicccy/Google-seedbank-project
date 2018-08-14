import tensorflow as tf

c = tf.constant('hello world!')
with tf.Session() as sess:
    print(sess.run(c))
