from __future__ import print_function
import tensorflow as tf

try:
    tf.contrib.eager.enable_eager_execution()
except ValueError:
    print('value error!')

graph = tf.Graph()
with graph.as_default():
    c = tf.constant('hello world!')
    with tf.Session(graph=graph) as sess:
        print(sess.run([c]))
