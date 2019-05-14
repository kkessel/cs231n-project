#!/usr/bin/env python
# coding: utf-8
import sys
import os
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import math
import numpy as np
import skimage.measure

mnist = input_data.read_data_sets('./mnist_data')
file_name = "mnist4.pb"

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def reshape_batch(x):
    modified_x = []
    for image in x:
        i = np.reshape(image, (28, 28))
        modified_x.append(np.reshape(skimage.measure.block_reduce(i, (2, 2), np.max), (196)))
    modified_x = np.array(modified_x)
    return modified_x

F = 5
C = 5
INPUT = 784
OUTPUT = 10
HFC = 50

x_in = tf.placeholder(tf.float32, [None, INPUT], name='input_op')

# W_conv = weight_variable([F, F, 1, C])
# b_conv = bias_variable([C])

#W_fc = weight_variable([INPUT, HFC])
#b_fc = bias_variable([HFC])

#W_o = weight_variable([HFC, OUTPUT])
#b_o = bias_variable([OUTPUT])

W_fc = weight_variable([INPUT, HFC])
b_fc = bias_variable([HFC])

W_fc1 = weight_variable([HFC, HFC])
b_fc1 = bias_variable([HFC])

W_fc2 = weight_variable([HFC, HFC])
b_fc2 = bias_variable([HFC])

W_fc3 = weight_variable([HFC, HFC])
b_fc3 = bias_variable([HFC])

W_fc4 = weight_variable([HFC, HFC])
b_fc4 = bias_variable([HFC])

W_fc5 = weight_variable([HFC, HFC])
b_fc5 = bias_variable([HFC])

W_o = weight_variable([HFC, OUTPUT])
b_o = bias_variable([OUTPUT])





x_image = tf.reshape(x_in, [-1, int(math.sqrt(INPUT)), int(math.sqrt(INPUT)), 1])

# h_conv = tf.reshape(tf.nn.relu(conv2d(x_image, W_conv) + b_conv), [-1, INPUT*C])


h_fc = tf.nn.relu(tf.add(tf.matmul(x_in, W_fc), b_fc))
h_fc1 = tf.nn.relu(tf.add(tf.matmul(h_fc, W_fc1), b_fc1))
h_fc2 = tf.nn.relu(tf.add(tf.matmul(h_fc1, W_fc2), b_fc2))
h_fc3 = tf.nn.relu(tf.add(tf.matmul(h_fc2, W_fc3), b_fc3))
h_fc4 = tf.nn.relu(tf.add(tf.matmul(h_fc3, W_fc4), b_fc4))
h_fc5 = tf.nn.relu(tf.add(tf.matmul(h_fc4, W_fc5), b_fc5))


y = tf.nn.relu(tf.add(tf.matmul(h_fc5, W_o), b_o, name="output_op"))

y_ = tf.placeholder(tf.int64, [None])
cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y)
train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(70000):
        if i%1000 == 0:
            print(i)
        batch_xs, batch_ys = mnist.train.next_batch(50)
        # modified_batch_xs = reshape_batch(batch_xs)
        sess.run(train_step, feed_dict={x_in: batch_xs, y_: batch_ys})
    correct_prediction = tf.equal(tf.argmax(y, 1), y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # modified_test_x = reshape_batch(test_x)
    print(sess.run(accuracy, feed_dict={x_in: mnist.test.images, y_: mnist.test.labels}))
    output_graph_def = tf.graph_util.convert_variables_to_constants(sess, tf.get_default_graph().as_graph_def(), ['output_op'])
    with tf.gfile.GFile('models/' + file_name, "wb") as f:
        f.write(output_graph_def.SerializeToString())

# from maraboupy import Marabou
# net = Marabou.read_tf('conv_model.pb', ['input_op'], 'output_op')
#
# mnist.test.images.shape
# mnist.test.labels.shape
# os.system('rm -rf ./numpy_format')
# for i in range(10):
#     os.system('mkdir -p ./numpy_format/correct/{}/'.format(i))
#     os.system('mkdir -p ./numpy_format/incorrect/{}/'.format(i))
#
# import numpy as np
# for i in range(10000):
#     label = mnist.test.labels[i]
#     image = mnist.test.images[i]
#     outs = net.evaluateWithoutMarabou([image])
#     ans = np.argmax(outs)
#     if ans==label:
#         correct = 'correct'
#     else:
#         correct = 'incorrect'
#     f = './numpy_format/{}/{}/{}.npy'.format(correct, label, i)
#     np.save(f, image)
#
