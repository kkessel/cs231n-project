import sys
sys.path.append('./tcav/')
import tensorflow as tf
import numpy as np
from tcav import model as tcav_models
import models

class MNISTModelWrapper(tcav_models.ImageModelWrapper):
  def __init__(self,
               sess,
               model_fn,
               ckpt_path,
               image_shape,
               bottleneck_scope):
    super(MNISTModelWrapper, self).__init__(image_shape)
    self.sess = sess
    # Create the model and restore the weights
    x_in, y, cross_entropy, y_, b_fc4 = model_fn()
    self.sess.run(tf.global_variables_initializer())
    print(self.sess.run(b_fc4))
    saver = tf.train.Saver()
    saver.restore(self.sess, ckpt_path)
    print(self.sess.run(b_fc4))
    # TODO: set all these things in here
    # A dictionary of bottleneck tensors.
    self.bottlenecks_tensors = MNISTModelWrapper.get_bottleneck_tensors(
        bottleneck_scope)
    print(self.bottlenecks_tensors)
    # A dictionary of input, 'logit' and prediction tensors.
    self.ends = {'input': x_in,
                 'logit': y,
                 'prediction': tf.argmax(y, axis=1)}
    # The model name string.
    self.model_name = 'MNIST Model 1'
    # a place holder for index of the neuron/class of interest.
    # usually defined under the graph. For example:
    # with g.as_default():
    #   self.tf.placeholder(tf.int64, shape=[None])
    self.y_input = tf.placeholder(tf.int64, shape=[None])
    # The tensor representing the loss (used to calculate derivative).
    self.loss = cross_entropy

    # Always call this at the end of __init__
    self._make_gradient_tensors()

  def label_to_id(self, label):
    """Convert label (string) to index in the logit layer (id)."""
    return int(label)

  def id_to_label(self, idx):
    """Convert index in the logit layer (id) to label (string)."""
    return str(idx)

  @staticmethod
  def get_bottleneck_tensors(scope):
    """Add bottlenecks and their pre-Relu versions to bn_endpoints dict."""
    graph = tf.get_default_graph()
    bn_endpoints = {}
    for op in graph.get_operations():
      if op.name.startswith(scope+'/') and \
         ('Add' in op.type or 'Relu' in op.type) :
        name = op.name.split('/')[1]
        bn_endpoints[name] = op.outputs[0]
    return bn_endpoints

if __name__ == '__main__':
  with tf.Session() as sess:
    ckpt_path = 'models/mnist5.ckpt'
    model = MNISTModelWrapper(sess, models.model_fn, ckpt_path, 28, 'fc1')
    print(model)