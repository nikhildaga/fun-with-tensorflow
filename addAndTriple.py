import tensorflow as tf

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b  # + provides a shortcut for tf.add(a, b)

add_and_triple = adder_node * 3

sess = tf.Session()

print(add_and_triple)
print(sess.run(add_and_triple, {a: 1, b: 2}))
print(sess.run(add_and_triple, {a: [1,2], b: [2,3]}))
