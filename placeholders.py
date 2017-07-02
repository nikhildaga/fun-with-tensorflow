import tensorflow as tf

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b  # + provides a shortcut for tf.add(a, b)

sess = tf.Session()

print(adder_node)
print(sess.run(adder_node, {a: 1, b: 2}))
print(sess.run(adder_node, {a: [1,2], b: [2,3]}))
