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
from tensorflow.python import pywrap_tensorflow

# If you set this to True, we'll get suuuuuuper verbose output to debug stuff
DEBUG = False

'''MNISTModelWrapper Class
TCAV requires us to subclass tcav_models.ImageModelWrapper.

All the class variables set in the constructor need to be set for TCAV to run
'''
class MNISTModelWrapper(tcav_models.ImageModelWrapper):
  def __init__(self,
               sess,
               model_fn,
               ckpt_path,
               image_shape,
               bottleneck_scopes,
               network_arch):
    super(MNISTModelWrapper, self).__init__(image_shape)
    self.sess = sess
    # Create the model and restore the weights
    x_in, y, cross_entropy, y_ = model_fn(network_arch)
    self.sess.run(tf.global_variables_initializer())

    # Debug the weird loading behavior...
    if DEBUG:
      print("Tensor values in checkpoint")
      print_tensors_in_checkpoint_file(ckpt_path, all_tensors=True, tensor_name='')
      print("="*80)
      tensor_model_names = [n.name for n in tf.get_default_graph().as_graph_def().node]
      print("Tensors in model")
      for name in tensor_model_names:
        print(name)
      print("="*80)
      print("Tensors in checkpoint")
      reader = pywrap_tensorflow.NewCheckpointReader(ckpt_path)
      var_to_shape_map = reader.get_variable_to_shape_map()
      tensor_ckpt_names = [n for n in sorted(var_to_shape_map)]
      for name in tensor_ckpt_names:
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

    # Restore the weights of the network
    saver = tf.train.Saver()
    saver.restore(self.sess, ckpt_path)

    # A dictionary of bottleneck tensors, i.e. the tensors that have the
    # activations at the desired depths in the layer
    # a.k.a. features in latent space
    self.bottlenecks_tensors = MNISTModelWrapper.get_bottleneck_tensors(
        bottleneck_scopes)
    if DEBUG:
      print("="*80)
      print("Bottleneck tensors")
      print(self.bottlenecks_tensors)

    # A dictionary of input, 'logit' and prediction tensors.
    self.ends = {'input': x_in,
                 'logit': y,
                 'prediction': tf.argmax(y, axis=1)}
    # The model name string. Not sure why we need this, but TCAV asked for it
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
    if 'all_blue_2s' in label:
      label = '2'
    return int(label)

  def id_to_label(self, idx):
    """Convert index in the logit layer (id) to label (string)."""
    return str(idx)

  @staticmethod
  def get_bottleneck_tensors(scopes):
    """Add bottlenecks and their pre-Relu versions to bn_endpoints dict."""
    graph = tf.get_default_graph()
    bn_endpoints = {}
    for op in graph.get_operations():
      for scope in scopes:
        if op.name.startswith(scope + '/') and \
           ('Add' in op.type or 'Relu' in op.type) :
          #name = op.name.split('/')[1]
          name = "_".join(op.name.split('/'))
          bn_endpoints[name] = op.outputs[0]
    return bn_endpoints

# This runs TCAV :)
def main(model_name='MNIST Model 1',
         cav_dir='tcav_class_test',
         ckpt_path='models/mnist5_blue2.ckpt',
         target="zebra",
         network_arch=5*[50]):
  tf.reset_default_graph()

  with tf.Session() as sess:
    model = MNISTModelWrapper(sess=sess,
                              model_fn=models.model_fn,
                              ckpt_path=ckpt_path,
                              image_shape=[28, 28, 3],
                              bottleneck_scopes=['fc1', 'fc4'],
                              network_arch=network_arch)

    # More debugging beauty
    if DEBUG:
      print(model)

    # This is the name of your model wrapper (InceptionV3 and GoogleNet are provided in model.py)
    model_to_run = model_name
    # the name of the parent directory that results are stored (only if you want to cache)
    project_name = cav_dir
    #working_dir = "/tmp/" + user + '/' + project_name
    working_dir = project_name
    # where activations are stored (only if your act_gen_wrapper does so)
    activation_dir =  working_dir + '/activations/'
    # where CAVs are stored.
    # You can say None if you don't wish to store any.
    cav_dir = working_dir + '/cavs/'
    # where the images live.
    source_dir = "data/"
    bottlenecks = ["fc1_relu", "fc4_relu"]  # @param

    utils.make_dir_if_not_exists(activation_dir)
    utils.make_dir_if_not_exists(working_dir)
    utils.make_dir_if_not_exists(cav_dir)

    # this is a regularizer penalty parameter for linear classifier to get CAVs.
    alphas = [0.1]

    target = target
    # concepts = ["blue", "green", "red", "cyan", "magenta", "yellow"]
    concepts = ["blue_0"]
    # random_concepts = ["not_blue_0", "not_blue_1", "not_blue_2", "not_blue_3", "not_blue_4",
    #                    "not_blue_5", "not_blue_6", "not_blue_7", "not_blue_8", "not_blue_9"]
    random_concepts = ["not_blue_0", "not_blue_1", "not_blue_2", "not_blue_3", "not_blue_4",
                       "not_blue_5", "not_blue_6", "not_blue_7"]
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
