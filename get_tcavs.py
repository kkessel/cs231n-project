import sys
sys.path.append('./tcav/')
import tensorflow as tf
import numpy as np
from tcav import model

# Helper functions to define the model
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# The actual model function.
# TODO: parametrize this atrocity
def model_fn():
  F = 5
  C = 5
  INPUT = 784
  OUTPUT = 10
  HFC = 50
  # variables
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

  # Compuational graph definition
  x_in = tf.placeholder(tf.float32, [None, INPUT], name='input_op')

  h_fc = tf.nn.relu(tf.add(tf.matmul(x_in, W_fc), b_fc))
  h_fc1 = tf.nn.relu(tf.add(tf.matmul(h_fc, W_fc1), b_fc1))
  h_fc2 = tf.nn.relu(tf.add(tf.matmul(h_fc1, W_fc2), b_fc2))
  h_fc3 = tf.nn.relu(tf.add(tf.matmul(h_fc2, W_fc3), b_fc3))
  h_fc4 = tf.nn.relu(tf.add(tf.matmul(h_fc3, W_fc4), b_fc4))
  h_fc5 = tf.nn.relu(tf.add(tf.matmul(h_fc4, W_fc5), b_fc5))


  y = tf.nn.relu(tf.add(tf.matmul(h_fc5, W_o), b_o, name="output_op"))

  y_ = tf.placeholder(tf.int64, [None])
  cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y)

  return x_in, y, cross_entropy, y_

class MNISTModelWrapper(ImageModelWrapper):
  def __init__(self,
               sess,
               model_fn,
               image_shape):
    super(MNISTModelWrapper, self).__init__(image_shape)
    self.sess = sess
    x_in, y, cross_entropy, y_ = model_fn()
    # TODO: set all these things in here
    # A dictionary of bottleneck tensors.
    self.bottlenecks_tensors = None
    # A dictionary of input, 'logit' and prediction tensors.
    self.ends = {'input': None,
                 'logit': None,
                 'prediction': None}
    # The model name string.
    self.model_name = None
    # a place holder for index of the neuron/class of interest.
    # usually defined under the graph. For example:
    # with g.as_default():
    #   self.tf.placeholder(tf.int64, shape=[None])
    self.y_input = tf.placeholder(tf.int64, shape=[None])
    # The tensor representing the loss (used to calculate derivative).
    self.loss = cross_entropy

    # Always call this at the end of __init__
    self._make_gradient_tensors()

  @abstractmethod
  def label_to_id(self, label):
    """Convert label (string) to index in the logit layer (id)."""
    pass

  @abstractmethod
  def id_to_label(self, idx):
    """Convert index in the logit layer (id) to label (string)."""
    pass


if __name__ == '__main__':
  model = model.ImageModelWrapper(224)
  print(model)