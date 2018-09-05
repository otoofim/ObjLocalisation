
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
import matplotlib.animation as animation
plt.switch_backend('agg')

if "./lib" not in sys.path:
    sys.path.append("./lib")

from collections import deque, namedtuple
from readingFileEfficiently import *
import VOC2012_npz_files_writter
from DNN import *
from Agent import ObjLocaliser


# In[ ]:




def visualizing_seq_act(model_name, add, ground_truth, output_name):


	tf.reset_default_graph()

	# Where we save our checkpoints and graphs
	experiment_dir = os.path.abspath("../experiments/{}".format(model_name))

	# Create a glboal step variable
	global_step = tf.Variable(0, name='global_step', trainable=False)

	# Create estimators
	q_estimator = Estimator(scope="q_estimator", summaries_dir=experiment_dir)

	# State processor
	state_processor = StateProcessor()



	im2 = np.array(Image.open(add))
	env = ObjLocaliser(np.array(im2),{'xmin':[ground_truth[0]], 'xmax':[ground_truth[2]], 'ymin':[ground_truth[1]], 'ymax':[ground_truth[3]]})

	with tf.Session() as sess:

	    fig = plt.figure()
       	    ims = []

	    sess.run(tf.global_variables_initializer())
	    #num_located = 0

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
	    #total_t = sess.run(tf.contrib.framework.get_global_step())


	    # The policy we're following
	    policy = make_epsilon_greedy_policy(
		      q_estimator,
		      len(VALID_ACTIONS))

	    precisions = []


            final_reward = 0

	    while final_reward != 3:

		plt.close()
		fig = plt.figure()
		ims = []
    		# Reset the environment
    		env.Reset(np.array(im2))
    		state = env.wrapping()
    		state = state_processor.process(sess, state)
    		state = np.stack([state] * 4, axis=2)

    		t=0
    		action = 0


    		# One step in the environment
    		while (action != 10) and (t < 50):

    		    action_probs, qs = policy(sess, state, 0.2)

    		    action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
    		    reward = env.takingActions(VALID_ACTIONS[action])

    		    if reward == 3:
    		        final_reward = 3

    		    next_state = env.wrapping()

                    imgplot = plt.imshow(env.my_draw())
                    ims.append([imgplot])

    		    next_state = state_processor.process(sess, next_state)
    		    next_state = np.append(state[:,:,1:], np.expand_dims(next_state, 2), axis=2)
    		    state = next_state

    		    t += 1
                print "unsuccessfull"
		#final_reward = 3


	    ani = animation.ArtistAnimation(fig, ims, interval=1000, blit=True, repeat_delay=1000)

	    if not os.path.exists('../experiments/{}/anim/'.format(model_name)):
        	os.makedirs('../experiments/{}/anim/'.format(model_name))
        
	    path = '../experiments/{}/anim/'.format(model_name)
	    ani.save('../experiments/{}/anim/{}.mp4'.format(model_name,output_name))






