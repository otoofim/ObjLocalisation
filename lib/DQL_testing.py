
# coding: utf-8

# In[1]:


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


# In[2]:


def DQL_testing(num_episodes, category, model_name):

    #This cell reads VOC 2012 dataset and save them in .npz files for future.
    #The process of reading data and put them in prper format is time consuming so they are stored in a file.

    destination = "../data/"

    #It splits dataset to 80% for training and 20% validation.
    if not (os.path.isfile(destination+"test_input.npz") or os.path.isfile(destination+"test_target.npz")):
        print("Files are not ready!!!")
    else:
        print("Records are already prepared!!!")


    # In[ ]:


    #num_episodes = 15
    #category = ['cat']

    #model_name = "catDogCowHorsePerson"


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

        sess.run(tf.initialize_all_variables())
        num_located = 0

        # For 'system/' summaries, usefull to check if currrent process looks healthy
        current_process = psutil.Process()

        # Create directories for checkpoints and summaries
        checkpoint_dir = os.path.join(experiment_dir, "bestModel")
        checkpoint_path = os.path.join(checkpoint_dir, "model")
        #report_path = os.path.join(experiment_dir, "report")


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

        precisions = []

        for indx,tmp in enumerate(extractData(category, "test", 32)):


            #if indx>200:
                #break

            img=tmp[0]
            target=tmp[1]
            succ = 0

            im2 = Image.frombytes("RGB",(img['image_width'],img['image_height']),img['image'])
            env = ObjLocaliser(np.array(im2),target)
            print "Image{} is being loaded: {}".format(indx, img['image_filename'])


            for i_episode in range(num_episodes):

                # Save the current checkpoint
                #saver.save(tf.get_default_session(), checkpoint_path)

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
                    #print action_probs
                    #print qs
                    action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
                    #if t==49: action = 10
                    reward = env.takingActions(VALID_ACTIONS[action])

                    if reward == 3:
                        succ += 1

                    #env.drawActions()
                    next_state = env.wrapping()
                    next_state = state_processor.process(sess, next_state)
                    next_state = np.append(state[:,:,1:], np.expand_dims(next_state, 2), axis=2)
                    state = next_state

                    t += 1

                print "number of actions for step {} is: {}".format(i_episode, t)


            precisions.append(float(succ)/num_episodes)
            print "image {} precision: {}".format(img['image_filename'], precisions[-1])



    print "num of images:{}".format(len(precisions))

    print "mean precision: {}".format(np.mean(precisions))

    return np.mean(precisions)



