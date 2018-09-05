
# coding: utf-8

# In[1]:


#get_ipython().magic(u'matplotlib inline')

import itertools
import numpy as np
import os
import random
import sys
import psutil
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
plt.switch_backend('agg')

if "./lib" not in sys.path:
    sys.path.append("./lib")
    
from collections import deque, namedtuple
from readingFileEfficiently import *
import VOC2012_npz_files_writter
from DNN import *
from Agent import ObjLocaliser






# Helper function to visualize conv layers
def plotNNFilter(units, model_name, layer_num):
    filters = units.shape[3]
    if not os.path.exists('../experiments/{}/visu/layer_{}'.format(model_name, layer_num)):
        os.makedirs('../experiments/{}/visu/layer_{}'.format(model_name, layer_num))
    
    for i in range(filters):
	fig = plt.figure(1, figsize=(32,32))
	#fig.imsave('../experiments/{}/visu/layer_{}/layer{}filter{}.png'.format(model_name,layer_num,layer_num, i+1), units[0,:,:,i], cmap="gray", dpi=fig.dpi)

	plt.imshow(units[0,:,:,i], interpolation="nearest", cmap="gray")
	fig.savefig('../experiments/{}/visu/layer_{}/layer{}filter{}.png'.format(model_name,layer_num,layer_num, i+1))
	plt.close()
	print "filter {} is plotted.".format(i)
    print "The plots can be found in ../experiments/{}/visu/layer_{}".format(model_name, layer_num)

    '''filters = units.shape[3]
    plt.figure(1, figsize=(64,64))
    n_columns = 4
    n_rows = math.ceil(filters / n_columns) + 1
    plt.subplots_adjust(wspace=0.2, hspace=0.3)
    for i in range(filters):
        plt.subplot(n_rows, n_columns, i+1)
        plt.title('Filter ' + str(i))
        #plt.imshow(units[0,:,:,i], interpolation="nearest", cmap="gray")
        plt.show(units[0,:,:,i], interpolation="nearest", cmap="gray", block=True)'''
	   
    





# In[2]:

def visualize_layers(model_name, add, layer_num):

	#model_name = "inc/cat_dropout_ep.2_episode5"


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
	    
	    sess.run(tf.global_variables_initializer()) 
	    num_located = 0

	    # For 'system/' summaries, usefull to check if currrent process looks healthy
	    current_process = psutil.Process()

	    # Create directories for checkpoints and summaries
	    checkpoint_dir = os.path.join(experiment_dir, "bestModel")
	    checkpoint_path = os.path.join(checkpoint_dir, "model")


	    saver = tf.train.Saver()
	    # Load a previous checkpoint if we find one
	    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
	    if latest_checkpoint:
		print("Loading model checkpoint {}...\n".format(latest_checkpoint))
		saver.restore(sess, latest_checkpoint)

	    # Get the current time step
	    total_t = sess.run(tf.contrib.framework.get_global_step())


	    # The policy we're following
	    policy = make_epsilon_greedy_policy(
		q_estimator,
		len(VALID_ACTIONS))
	    


	    im2 = np.array(Image.open(add))
	    env = ObjLocaliser(np.array(im2),{'xmin':[0], 'xmax':[1], 'ymin':[0], 'ymax':[1]})
		

	    # Reset the environment
	    env.Reset(np.array(im2))
	    state = env.wrapping()
	    state = state_processor.process(sess, state)
	    state = np.stack([state] * 4, axis=2)
	    layer = q_estimator.visulize_layers(sess, state.reshape((-1, 84, 84, 4)), layer_num)
	    plotNNFilter(layer, model_name, layer_num)




