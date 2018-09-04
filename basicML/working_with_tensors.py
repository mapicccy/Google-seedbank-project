from __future__ import print_function
import tensorflow as tf

try:
    tf.contrib.eager.enable_eager_execution()
except ValueError:
    print('value error!')
else:
    print('executing_eagerly:', tf.executing_eagerly())

primes = tf.constant([2, 3, 5, 7, 11, 13], dtype=tf.int32)
print('primes', primes)

ones = tf.ones([6], dtype=tf.int32)
print('ones:', ones)

just_beyond_primes = tf.add(primes, ones)
print('just_beyond_primes:', just_beyond_primes)

twos = tf.constant([2, 2, 2, 2, 2, 2], dtype=tf.int32)
primes_doubled = primes * twos
print('primes_doubled:', primes_doubled)

some_matrix = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.int32)
print('some_matrix:', some_matrix)
print('value of some_matrix is:', some_matrix.numpy())

scalar = tf.zeros([])
vector = tf.zeros([3])
matrix = tf.zeros([2, 3])
print('scalar has shape:', scalar.get_shape(), 'and value:', scalar.numpy())
print('vector has shape:', vector.get_shape(), 'and value:', vector.numpy())
print('matrix has shape:', matrix.get_shape(), 'and value:', matrix.numpy())

primes = tf.constant([2, 3, 5, 7, 11, 13], dtype=tf.int32)
print('primes:', primes)
one = tf.constant(1, dtype=tf.int32)
print('one:', one)
just_beyond_primes = tf.add(primes, one)
print('just_beyond_primes:', just_beyond_primes)
two = tf.constant(2, dtype=tf.int32)
primes_doubled = primes * two
print(('primes_doubled:', primes_doubled))

# Tensor Reshaping
matrix = tf.constant([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]])
reshaped_2x8_matrix = tf.reshape(matrix, [2, 8])
reshaped_4x4_matrix = tf.reshape(matrix, [4, 4])
print('origin matrix:', matrix)
print('reshape matrix(2x8):', reshaped_2x8_matrix)
print('reshape matrix(4x4):', reshaped_4x4_matrix)

v = tf.contrib.eager.Variable([3])
w = tf.contrib.eager.Variable(tf.random_normal([1, 4], mean=1., stddev=.35))
print('v:', v)
print('w:', w)

print('before v:', v.numpy())
tf.assign(v, [7])
print('after v:', v.numpy())
v.assign([4])
print('v.assign([4]):', v.numpy())

v = tf.contrib.eager.Variable([[1, 2, 3], [4, 5, 6]])
print('v:', v.numpy())
try:
    v.assign([7, 8, 9])
except ValueError as e:
    print('Exception e:', e)
