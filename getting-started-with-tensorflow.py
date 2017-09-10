#import
import tensorflow as tf

#constants
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0)

print("node1, node2:", node1, node2)

#session
session = tf.Session()

print("evaluate node1, node2:", session.run([node1, node2]))

node3 = tf.add(node1, node2)

print("add node1, node2:", node3)

print(session.run(node3))

#placeholders
a = tf.placeholder(tf.float32)

b = tf.placeholder(tf.float32)

adder_node = a + b

print(adder_node, {a: 1, b: 2})

print(session.run(adder_node, {a: [1, 9], b: 2}))

add_and_triple = adder_node * 3

print(session.run(add_and_triple, {a: [1, 9], b: 2}))

#Variables
W = tf.Variable(0.5, dtype=tf.float32)
b = tf.Variable(-0.5, dtype=tf.float32)
x = tf.placeholder(tf.float32)

#linear_model
linear_model = W * x + b

#init variables
init = tf.global_variables_initializer()
session.run(init)

print(linear_model, {x: 1})

print(session.run(linear_model, {x: [1, 2, 3, 4]}))

#loss function
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)

print(session.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

#manual fix
fixW = tf.assign(W, -1)
fixb = tf.assign(b, 1)

session.run([fixW, fixb])
print(session.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

#fix using GradientDescentOptimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

#reset w , b to initial values
session.run(init)

#taining data
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

#training loop
for i in range(1000):
    session.run(train, {x: x_train, y: y_train})

# evaluate training accuracy
curr_W, curr_b, curr_loss = session.run([W, b, loss], {x: x_train, y: y_train})
print("W: %s b: %s loss: %s" % (curr_W, curr_b, curr_loss))

#
#
#
#
#
### so far we have trained model using tf's low level api
### now we will see how to train using tf's high level api
import numpy as np

feature_columns = [tf.feature_column.numeric_column("x", shape=[1])]

estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns)

#data
x_train = np.array([1, 2, 3, 4.])
y_train = np.array([0, -1, -2, -3])
x_eval = np.array([2, 5, 8, 1])
y_eval = np.array([-1.01, -4.1, -7, 0])

#train
input_fn = tf.estimator.inputs.numpy_input_fn(
    {
        "x": x_train
    }, y_train, batch_size=4, num_epochs=None, shuffle=True)

estimator.train(input_fn=input_fn, steps=1000)

#evaluate
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    {
        "x": x_train
    }, y_train, batch_size=4, num_epochs=1000, shuffle=False)

eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    {
        "x": x_eval
    }, y_eval, batch_size=4, num_epochs=1000, shuffle=False)

train_metrics = estimator.evaluate(input_fn=train_input_fn)
eval_metrics = estimator.evaluate(input_fn=eval_input_fn)

print("train metrics: %r" % train_metrics)
print("eval metrics: %r" % eval_metrics)


#
#
#custom function in estimator
# Declare list of features, we only have one real-valued feature
def model_fn(features, labels, mode):
    # Build a linear model and predict values
    W = tf.get_variable("W", [1], dtype=tf.float64)
    b = tf.get_variable("b", [1], dtype=tf.float64)
    y = W * features['x'] + b
    # Loss sub-graph
    loss = tf.reduce_sum(tf.square(y - labels))
    # Training sub-graph
    global_step = tf.train.get_global_step()
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = tf.group(optimizer.minimize(loss), tf.assign_add(global_step, 1))
    # EstimatorSpec connects subgraphs we built to the
    # appropriate functionality.
    return tf.estimator.EstimatorSpec(
        mode=mode, predictions=y, loss=loss, train_op=train)


estimator = tf.estimator.Estimator(model_fn=model_fn)
# define our data sets
x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])
x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7, 0.])
input_fn = tf.estimator.inputs.numpy_input_fn(
    {
        "x": x_train
    }, y_train, batch_size=4, num_epochs=None, shuffle=True)
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    {
        "x": x_train
    }, y_train, batch_size=4, num_epochs=1000, shuffle=False)
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    {
        "x": x_eval
    }, y_eval, batch_size=4, num_epochs=1000, shuffle=False)

# train
estimator.train(input_fn=input_fn, steps=1000)
# Here we evaluate how well our model did.
train_metrics = estimator.evaluate(input_fn=train_input_fn)
eval_metrics = estimator.evaluate(input_fn=eval_input_fn)
print("train metrics: %r" % train_metrics)
print("eval metrics: %r" % eval_metrics)

#
#
#
#
#tensorboard
fileWriter = tf.summary.FileWriter('./tensorboard', session.graph)

tf.global_variables_initializer().run(session=session)
