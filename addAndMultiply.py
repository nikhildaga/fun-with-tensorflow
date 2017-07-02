import tensorflow as tf

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
c = tf.placeholder(tf.float32)

add_and_multiply = (a + b) * c

sess = tf.Session()

print(add_and_multiply)
print(sess.run(add_and_multiply, {a: 1, b: 2, c: 3}))
print(sess.run(add_and_multiply, {a: [1,2], b: [2,3], c: 3}))
print(sess.run(add_and_multiply, {a: [1,2], b: [2,3], c: [4,5]}))
