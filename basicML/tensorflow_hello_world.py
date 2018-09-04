from __future__ import print_function
import tensorflow as tf

try:
    tf.contrib.eager.enable_eager_execution()
except ValueError:
    pass


c = tf.constant('hello world!')
with tf.Session(graph=tf.get_default_graph) as sess:
    print(sess.run(c))
