import tensorflow as tf

# Helper functions to define the model
def weight_variable(shape, name='weight'):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name='bias'):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

# The actual model function.
# TODO: parametrize this atrocity
def model_fn_with_scopes():
  F = 5
  C = 5
  INPUT = 28*28*3
  OUTPUT = 10
  HFC = 50


  with tf.variable_scope('in', reuse=True):
    W_fc = weight_variable([INPUT, HFC])
    b_fc = bias_variable([HFC])
    x_in = tf.placeholder(tf.float32, [None, 28,28,3], name='input_op')
    layer_in = tf.layers.Flatten(data_format='channels_last')(x_in)
    h_fc = tf.nn.relu(tf.add(tf.matmul(layer_in, W_fc), b_fc))

  with tf.variable_scope('fc1', reuse=True):
    W_fc1 = weight_variable([HFC, HFC])
    b_fc1 = bias_variable([HFC])
    h_fc1 = tf.nn.relu(tf.add(tf.matmul(h_fc, W_fc1), b_fc1))
  with tf.variable_scope('fc2', reuse=True):
    W_fc2 = weight_variable([HFC, HFC])
    b_fc2 = bias_variable([HFC])
    h_fc2 = tf.nn.relu(tf.add(tf.matmul(h_fc1, W_fc2), b_fc2))
  with tf.variable_scope('fc3', reuse=True):
    W_fc3 = weight_variable([HFC, HFC])
    b_fc3 = bias_variable([HFC])
    h_fc3 = tf.nn.relu(tf.add(tf.matmul(h_fc2, W_fc3), b_fc3))
  with tf.variable_scope('fc4', reuse=True):
    W_fc4 = weight_variable([HFC, HFC])
    b_fc4 = bias_variable([HFC])
    h_fc4 = tf.nn.relu(tf.add(tf.matmul(h_fc3, W_fc4), b_fc4))
  with tf.variable_scope('fc5', reuse=True):
    W_fc5 = weight_variable([HFC, HFC])
    b_fc5 = bias_variable([HFC])
    h_fc5 = tf.nn.relu(tf.add(tf.matmul(h_fc4, W_fc5), b_fc5))

  with tf.variable_scope('out', reuse=True):
      W_o = weight_variable([HFC, OUTPUT])
      b_o = bias_variable([OUTPUT])
      y = tf.nn.relu(tf.add(tf.matmul(h_fc5, W_o), b_o, name="output_op"))

  y_ = tf.placeholder(tf.int64, [None], name='labels')
  cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y)

  return x_in, y, cross_entropy, y_, b_fc

def model_fn(mode='tcav'):
    F = 5
    C = 5
    INPUT = 28*28*3
    OUTPUT = 10
    HFC = 50


    W_fc = weight_variable([INPUT, HFC])
    b_fc = bias_variable([HFC])
    if mode == 'tcav':
      x_in = tf.placeholder(tf.float32, [None, 28,28,3], name="input_op")
      layer_in = tf.layers.Flatten(data_format='channels_last')(x_in)
      h_fc = tf.nn.relu(tf.add(tf.matmul(layer_in, W_fc), b_fc, name="fc0/add"), name="fc0/relu")
    elif mode == 'proto':
      x_in = tf.placeholder(tf.float32, [None, INPUT], name='input_op')
      h_fc = tf.nn.relu(tf.add(tf.matmul(x_in, W_fc), b_fc, name="fc0/add"), name="fc0/relu")

    W_fc1 = weight_variable([HFC, HFC])
    b_fc1 = bias_variable([HFC])
    h_fc1 = tf.nn.relu(tf.add(tf.matmul(h_fc, W_fc1), b_fc1, name="fc1/add"), name="fc1/relu")

    W_fc2 = weight_variable([HFC, HFC])
    b_fc2 = bias_variable([HFC])
    h_fc2 = tf.nn.relu(tf.add(tf.matmul(h_fc1, W_fc2), b_fc2, name="fc2/add"), name="fc2/relu")

    W_fc3 = weight_variable([HFC, HFC])
    b_fc3 = bias_variable([HFC])
    h_fc3 = tf.nn.relu(tf.add(tf.matmul(h_fc2, W_fc3), b_fc3, name="fc3/add"), name="fc3/relu")

    W_fc4 = weight_variable([HFC, HFC])
    b_fc4 = bias_variable([HFC])
    h_fc4 = tf.nn.relu(tf.add(tf.matmul(h_fc3, W_fc4), b_fc4, name="fc4/add"), name="fc4/relu")

    W_fc5 = weight_variable([HFC, HFC])
    b_fc5 = bias_variable([HFC])
    h_fc5 = tf.nn.relu(tf.add(tf.matmul(h_fc4, W_fc5), b_fc5, name="fc5/add"), name="fc5/relu")

    W_o = weight_variable([HFC, OUTPUT])
    b_o = bias_variable([OUTPUT])
    y = tf.add(tf.matmul(h_fc5, W_o), b_o, name="output_op")

    y_ = tf.placeholder(tf.int64, [None], name='labels')
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y)

    return x_in, y, cross_entropy, y_, b_fc
