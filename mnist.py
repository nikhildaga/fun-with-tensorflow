from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf

# Here we assign x a shape of [None, 784],
# where 784 is the dimensionality of a single flattened 28 by 28 pixel MNIST image,
# and None indicates that the first dimension, corresponding to the batch size, can be of any size.
x = tf.placeholder(tf.float32, [None, 784])

# W has a shape of [784, 10] because we want to multiply the 784-dimensional image vectors by it
# to produce 10-dimensional vectors of evidence for the difference classes.
# b has a shape of [10] so we can add it to the output.

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

session = tf.InteractiveSession()

tf.global_variables_initializer().run()

for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    session.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(session.run(
    accuracy, feed_dict={x: mnist.test.images,
                         y_: mnist.test.labels}))
