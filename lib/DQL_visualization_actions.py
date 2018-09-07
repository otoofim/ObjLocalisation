import numpy as np
import os
import sys
import psutil
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from readingFileEfficiently import *
import VOC2012_npz_files_writter
from DNN import *
from Agent import ObjLocaliser





def visualizing_seq_act(model_name, add, ground_truth, output_name):
        """
        Visualizing sequence of actions 

        Args:
          model_name: The model parameters that will be loaded for testing.
          add: Path to an image
          ground_truth: Target coordinates
          output_name: Name of the output file
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


        # Creates an object localizer instance
	im2 = np.array(Image.open(add))
	env = ObjLocaliser(np.array(im2),{'xmin':[ground_truth[0]], 'xmax':[ground_truth[2]], 'ymin':[ground_truth[1]], 'ymax':[ground_truth[3]]})

	with tf.Session() as sess:

	    fig = plt.figure()
       	    ims = []

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

	    precisions = []
            final_reward = 0

            # Keeps going until the agent could successfully localize and object
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


    		# The agent searches in an image until terminatin action is used or the agent reaches threshold 50 actions
    		while (action != 10) and (t < 50):

                    # Choosing action based on epsilon-greedy with probability 0.8
    		    action_probs, qs = policy(sess, state, 0.2)
    		    action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

		    # Takes action and observes new state and reward
    		    reward = env.takingActions(VALID_ACTIONS[action])
    		    next_state = env.wrapping()
    		    if reward == 3:
    		        final_reward = 3

                    imgplot = plt.imshow(env.my_draw())
                    ims.append([imgplot])

		    # Processing the new state
    		    next_state = state_processor.process(sess, next_state)
    		    next_state = np.append(state[:,:,1:], np.expand_dims(next_state, 2), axis=2)
    		    state = next_state

    		    t += 1
                print "Unsuccessfull. Next try!"


	    # Saving animation
	    ani = animation.ArtistAnimation(fig, ims, interval=1000, blit=True, repeat_delay=1000)
	    if not os.path.exists('../experiments/{}/anim/'.format(model_name)):
        	os.makedirs('../experiments/{}/anim/'.format(model_name))
	    path = '../experiments/{}/anim/'.format(model_name)
	    ani.save('../experiments/{}/anim/{}.mp4'.format(model_name,output_name))
            print "The video is stored in ../experiments/{}/anim/{}.mp4".format(model_name,output_name)






