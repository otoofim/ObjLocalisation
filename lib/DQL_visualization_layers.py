import numpy as np
import os
import sys
import psutil
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
plt.switch_backend('agg')
    
from readingFileEfficiently import *
import VOC2012_npz_files_writter
from DNN import *
from Agent import ObjLocaliser




def plotNNFilter(units, model_name, layer_num):
    """
    Helper function to visualize conv layers
    
    Args:
      units: Conv layer filters
      model_name: Model name that is used to visualize its layer
      layer_num: Layer number to be visualized
    """

    filters = units.shape[3]
    if not os.path.exists('../experiments/{}/visu/layer_{}'.format(model_name, layer_num)):
        os.makedirs('../experiments/{}/visu/layer_{}'.format(model_name, layer_num))
    
    for i in range(filters):
	fig = plt.figure(1, figsize=(10,10))
	plt.imshow(units[0,:,:,i], interpolation="nearest", cmap="gray")
        fig.suptitle('layer{} filter{}'.format(layer_num, i+1), fontsize=60)
	fig.savefig('../experiments/{}/visu/layer_{}/layer{}filter{}.png'.format(model_name,layer_num,layer_num, i+1))
	plt.close()
	print "filter {} is plotted.".format(i)
    print "The plots can be found in ../experiments/{}/visu/layer_{}".format(model_name, layer_num)
	   
    




def visualize_layers(model_name, add, layer_num):
        """
        Visualizing sequence of actions 

        Args:
          model_name: The model parameters that will be loaded for visualizing.
          add: Path to an image
          layer_num: Layer number to be visualized
        """

        # Initiates Tensorflow graph
	tf.reset_default_graph()

	# Where we save our checkpoints and graphs
	experiment_dir = os.path.abspath("../experiments/{}".format(model_name))

	# Create a glboal step variable
	global_step = tf.Variable(0, name='global_step', trainable=False)
	    
	# Create estimators
	q_estimator = Estimator(scope="q_estimator", summaries_dir=experiment_dir)

	# State processor
	state_processor = StateProcessor()

	with tf.Session() as sess:
	    

	    # For 'system/' summaries, usefull to check if currrent process looks healthy
	    current_process = psutil.Process()

	    # Create directories for checkpoints and summaries
	    checkpoint_dir = os.path.join(experiment_dir, "bestModel")
	    checkpoint_path = os.path.join(checkpoint_dir, "model")


            # Initiates a saver and loads previous saved model if one was found
	    saver = tf.train.Saver()
	    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
	    if latest_checkpoint:
		print("Loading model checkpoint {}...\n".format(latest_checkpoint))
		saver.restore(sess, latest_checkpoint)

	    # The policy we're following
	    policy = make_epsilon_greedy_policy(
		q_estimator,
		len(VALID_ACTIONS))
	    

            # Creates an object localizer instance
	    im2 = np.array(Image.open(add))
	    env = ObjLocaliser(np.array(im2),{'xmin':[0], 'xmax':[1], 'ymin':[0], 'ymax':[1]})
		

	    # Reset the environment
	    env.Reset(np.array(im2))
	    state = env.wrapping()
	    state = state_processor.process(sess, state)
	    state = np.stack([state] * 4, axis=2)
       
            # Visualizing the network layers
	    layer = q_estimator.visulize_layers(sess, state.reshape((-1, 84, 84, 4)), layer_num)
	    plotNNFilter(layer, model_name, layer_num)




