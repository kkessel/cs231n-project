import sys
sys.path.append('./tcav/')
import tensorflow as tf
import numpy as np
from tcav import model as tcav_models
from tcav import activation_generator as act_gen
from tcav import utils, utils_plot
from tcav import tcav
import models

from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

DEBUG = True

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

    if DEBUG:
      print("Tensor values in checkpoint")
      print_tensors_in_checkpoint_file(ckpt_path, all_tensors=True, tensor_name='')
      tensor_names = [n.name for n in tf.get_default_graph().as_graph_def().node]
      print("="*80)
      print("Tensors in checkpoint")
      for name in tensor_names:
        print(name)
      print("="*80)
      print("Variables in graph:")
      variables_names = [v.name for v in tf.trainable_variables()]
      for v in variables_names:
        print(v)
      print("="*80)
      print("Variables before loading checkpoint")
      dgraph = tf.get_default_graph()
      bias = dgraph.get_tensor_by_name("fc4/bias:0")
      print(self.sess.run(bias))
      saver = tf.train.Saver()
      saver.restore(self.sess, ckpt_path)
      print("="*80)
      print("Variables after loading checkpoint")
      graph = tf.get_default_graph()
      bias = graph.get_tensor_by_name("fc4/bias:0")
      print(self.sess.run(bias))

    saver = tf.train.Saver()
    saver.restore(self.sess, ckpt_path)
    # TODO: set all these things in here
    # A dictionary of bottleneck tensors.
    self.bottlenecks_tensors = MNISTModelWrapper.get_bottleneck_tensors(
        bottleneck_scope)
    if DEBUG:
      print("="*80)
      print("Bottleneck tensors")
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
    self.y_input = y_
    # The tensor representing the loss (used to calculate derivative).
    self.loss = cross_entropy

    # Always call this at the end of __init__
    self._make_gradient_tensors()

  def label_to_id(self, label):
    """Convert label (string) to index in the logit layer (id)."""
    if label == 'zebra':
      label = '2'
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

def main():
  with tf.Session() as sess:
    ckpt_path = 'models/mnist5.ckpt'
    model = MNISTModelWrapper(sess,
                              models.model_fn,
                              ckpt_path,
                              [28, 28, 3],
                              bottleneck_scope='fc1')

    if DEBUG:
      print(model)

    # This is the name of your model wrapper (InceptionV3 and GoogleNet are provided in model.py)
    model_to_run = 'MNIST Model 1'
    user = 'beenkim'
    # the name of the parent directory that results are stored (only if you want to cache)
    project_name = 'tcav_class_test'
    #working_dir = "/tmp/" + user + '/' + project_name
    working_dir = project_name
    # where activations are stored (only if your act_gen_wrapper does so)
    activation_dir =  working_dir+ '/activations/'
    # where CAVs are stored.
    # You can say None if you don't wish to store any.
    cav_dir = working_dir + '/cavs/'
    # where the images live.
    source_dir = "data/"
    bottlenecks = ['Add', "Relu"]  # @param

    utils.make_dir_if_not_exists(activation_dir)
    utils.make_dir_if_not_exists(working_dir)
    utils.make_dir_if_not_exists(cav_dir)

    # this is a regularizer penalty parameter for linear classifier to get CAVs.
    alphas = [0.1]

    target = "zebra"
    concepts = ["blue","green","red","cyan","magenta","yellow"]
    random_concepts = ["not_blue", "not_red", "not_green", "not_cyan", "not_magenta", "not_yellow"]
    act_generator = act_gen.ImageActivationGenerator(model,
                                                     source_dir,
                                                     activation_dir,
                                                     max_examples=100)
    # Run TCAV!
    tf.logging.set_verbosity(0)

    mytcav = tcav.TCAV(sess,
                       target,
                       concepts,
                       bottlenecks,
                       act_generator,
                       alphas,
                       cav_dir=cav_dir,
                       random_concepts=random_concepts)

    results = mytcav.run()
    print(results)

    utils_plot.plot_results(results, random_concepts=random_concepts)

if __name__ == '__main__':
  main()
