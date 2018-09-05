
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
import argparse
import urllib
plt.switch_backend('agg')

if "./lib" not in sys.path:
    sys.path.append("./lib")

#import plotting
from collections import deque, namedtuple
from readingFileEfficiently import *
import VOC2012_npz_files_writter
from DNN import *
from Agent import ObjLocaliser





def preparedataset():

    xml_path = "../VOC2012/Annotations/*.xml"
    destination = "../data/"


    #It splits dataset to 80% for training and 20% validation.
    if not (os.path.isfile(destination+"test_input.npz") or os.path.isfile(destination+"test_target.npz")):

        if not os.path.isfile("../VOCtrainval_11-May-2012.tar"):

        	print "downloading VOC2012 dataset to ../pascal-voc-2012.zip ..."
        	os.system("wget -P ../ http://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar")
        	print "download finished."

        if not os.path.isdir("../VOC2012"):

        	#u = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
        	#u='https://www.kaggle.com/huanghanchina/pascal-voc-2012/downloads/pascal-voc-2012.zip'



        	#urllib.urlretrieve(u, "../pascal-voc-2012.zip")

        	print "Unziping the files ..."
        	os.system("tar xf ../VOCtrainval_11-May-2012.tar -C ../")
        	os.system("cp -r ../VOCdevkit/* ../")
        	os.system("rm -r ../VOCdevkit")
	
	os.system("mkdir ../data")
        VOC2012_npz_files_writter.writting_files(xml_path, destination, percentage=0)
        print("Files are ready!!!")
        
    else:
        print("Records are already prepared!!!")


def evaluate(tmp, num_of_proposal, state_processor, policy, sess):

    img=tmp[0]
    target=tmp[1]
    succ = 0

    im2 = Image.frombytes("RGB",(img['image_width'],img['image_height']),img['image'])
    env = ObjLocaliser(np.array(im2),target)
    #print "Image {} is being loaded: {}".format(indx, img['image_filename'])


    for i_episode in range(num_of_proposal):


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
            reward = env.takingActions(VALID_ACTIONS[action])

            if reward == 3:
                succ += 1

            #env.drawActions()
            next_state = env.wrapping()
            next_state = state_processor.process(sess, next_state)
            #print next_state.shape
            #print np.expand_dims(next_state, 2).shape
            next_state = np.append(state[:,:,1:], np.expand_dims(next_state, 2), axis=2)
            #print next_state.shape
            state = next_state

            t += 1
        #print "number of actions for step {} is: {}".format(i_episode, t)

    return (float(succ)/num_of_proposal)
    #print "image {} precision: {}".format(img['image_filename'], precisions[-1])


# In[4]:

def DQL(num_episodes,
	 replay_memory_size,
	 replay_memory_init_size,
	 update_target_estimator_every,
	 discount_factor,
	 epsilon_start,
	 epsilon_end,
	 epsilon_decay_steps,
	 category,
	 model_name):





    #This cell reads VOC 2012 dataset and save them in .npz files for future.
    #The process of reading data and put them in prper format is time consuming so they are stored in a file.

    preparedataset()




    #num_episodes=1
    #replay_memory_size=500000
    #replay_memory_init_size=500
    #update_target_estimator_every=10000
    #discount_factor=0.99
    #epsilon_start=1.0
    #epsilon_end=0.2
    #epsilon_decay_steps=500
    batch_size=32
    #category = ['person','cat','cow','horse','dog']


    tf.reset_default_graph()

    # Where we save our checkpoints and graphs
    experiment_dir = os.path.abspath("../experiments/{}".format(model_name))

    # Create a glboal step variable
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Create estimators
    q_estimator = Estimator(scope="q_estimator", summaries_dir=experiment_dir)
    target_estimator = Estimator(scope="target_q")

    # State processor
    state_processor = StateProcessor()



    done = False
    elist = []
    rlist = []


    with tf.Session() as sess:

        sess.run(tf.initialize_all_variables())
        Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

        # The replay memory
        replay_memory = []

        num_located = 0

        # Make model copier object
        estimator_copy = ModelParametersCopier(q_estimator, target_estimator)


        # For 'system/' summaries, usefull to check if currrent process looks healthy
        current_process = psutil.Process()

        # Create directories for checkpoints and summaries
        checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
        checkpoint_path = os.path.join(checkpoint_dir, "model")
        report_path = os.path.join(experiment_dir, "report")
        best_model_dir = os.path.join(experiment_dir, "bestModel")
        best_model_path = os.path.join(best_model_dir, "model")


        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        if not os.path.exists(report_path):
            os.makedirs(report_path)
        if not os.path.exists(best_model_dir):
            os.makedirs(best_model_dir)
        f = open(report_path+"/log.txt", 'w')
        saver = tf.train.Saver()
        # Load a previous checkpoint if we find one
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            print("Loading model checkpoint {}...\n".format(latest_checkpoint))
            saver.restore(sess, latest_checkpoint)

        # Get the current time step
        total_t = sess.run(tf.contrib.framework.get_global_step())

        # The epsilon decay schedule
        epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

        # The policy we're following
        policy = make_epsilon_greedy_policy(
            q_estimator,
            len(VALID_ACTIONS))
        episode_counter = 0
        best_pre = 0
        eval_pre = []
        eval_set = []
        #CatCounter = {'cat':0, 'dog':0, 'car':0, 'person':0, 'cow':0, 'horse':0}
        #CatCounter2 = {'cat':0, 'dog':0, 'car':0, 'person':0, 'cow':0, 'horse':0}


	for indx,tmp in enumerate(extractData(category, "train", batch_size)):

            img=tmp[0]
            target=tmp[1]

            #CatCounter[target['objName']] = CatCounter[target['objName']] + 1

            #if CatCounter[target['objName']] > 1100:

               #continue

            if len(eval_set) < 100:
                print "Populating evaluation set..."
                eval_set.append(tmp)
    	    #CatCounter2[tmp[1]['objName']] = CatCounter2[tmp[1]['objName']] + 1

            else:
                #print CatCounter2
    	    #break
                if indx%20 == 0:
                    print "Evaluation started ..."
                    for tmp2 in eval_set:
                        eval_pre.append(evaluate(tmp2, 15, state_processor, policy, sess))
                        if len(eval_pre) > 99:
                            print "Evaluation mean precision: {}".format(np.mean(eval_pre))
                            f.write("Evaluation mean precision: {}\n".format(np.mean(eval_pre)))
                            episode_summary = tf.Summary()
                            episode_summary.value.add(simple_value=np.mean(eval_pre), tag="episode/eval_acc")
                            q_estimator.summary_writer.add_summary(episode_summary, episode_counter)
                            q_estimator.summary_writer.flush()


                            if np.mean(eval_pre) > best_pre:
                                print "Best model changed with mean precision: {}".format(np.mean(eval_pre))
                                f.write("Best model changed with mean precision: {}\n".format(np.mean(eval_pre)))
                                best_pre = np.mean(eval_pre)
                                saver.save(tf.get_default_session(), best_model_path)
                            eval_pre = []

                #img=tmp[0]
                #target=tmp[1]

                #CatCounter[target['objName']] = CatCounter[target['objName']] + 1

                #if CatCounter[target['objName']] > 1100:

                    #continue

                im2 = Image.frombytes("RGB",(img['image_width'],img['image_height']),img['image'])
                env = ObjLocaliser(np.array(im2),target)
                print "Image{} is being loaded: {}".format(indx, img['image_filename'])
                f.write("Image{} is being loaded: {}".format(indx, img['image_filename']))

                if len(replay_memory) < replay_memory_init_size:

                    # Populate the replay memory with initial experience
                    print("Populating replay memory...\n")

                    env.Reset(np.array(im2))
                    state = env.wrapping()

                    state = state_processor.process(sess, state)
    		#cv2.imwrite('test.png',state)
    		#break

                    state = np.stack([state] * 4, axis=2)

                    for i in range(replay_memory_init_size):

                        action_probs, _ = policy(sess, state, epsilons[min(total_t, epsilon_decay_steps-1)])
                        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

                        reward = env.takingActions(VALID_ACTIONS[action])
                        next_state = env.wrapping()

                        if action == 10:
                            done = True
                        else:
                            done = False

                        next_state = state_processor.process(sess, next_state)
                        next_state = np.append(state[:,:,1:], np.expand_dims(next_state, 2), axis=2)
                        replay_memory.append(Transition(state, action, reward, next_state, done))
                        state = next_state


                        if done:
                            env.Reset(np.array(im2))
                            state = env.wrapping()
                            state = state_processor.process(sess, state)
                            state = np.stack([state] * 4, axis=2)
                        else:
                            state = next_state




                for i_episode in range(num_episodes):

                    # Save the current checkpoint
                    saver.save(tf.get_default_session(), checkpoint_path)

                    # Reset the environment
                    env.Reset(np.array(im2))
                    state = env.wrapping()
                    state = state_processor.process(sess, state)
                    state = np.stack([state] * 4, axis=2)
                    loss = None
                    t=0
                    action = 0
                    e = 0
                    r = 0

                    # One step in the environment
                    while (action != 10) and (t < 50):

                        # Epsilon for this time step
                        epsilon = epsilons[min(total_t, epsilon_decay_steps-1)]

                        # Maybe update the target estimator
                        if total_t % update_target_estimator_every == 0:
                            estimator_copy.make(sess)
                            print("\nCopied model parameters to target network.")



                        # Take a step
                        action_probs, qs = policy(sess, state, epsilon)
                        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)


                        #next_state, reward, done, _ = env.step(VALID_ACTIONS[action])
                        reward = env.takingActions(VALID_ACTIONS[action])
                        next_state = env.wrapping()
                        if action == 10:
                            done = True
                        else:
                            done = False


                        next_state = state_processor.process(sess, next_state)
                        next_state = np.append(state[:,:,1:], np.expand_dims(next_state, 2), axis=2)

                        # If our replay memory is full, pop the first element
                        if len(replay_memory) == replay_memory_size:
                            replay_memory.pop(0)

                        # Save transition to replay memory
                        replay_memory.append(Transition(state, action, reward, next_state, done))


                        # Sample a minibatch from the replay memory
                        samples = random.sample(replay_memory, batch_size)
                        states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*samples))

                        # Calculate q values and targets
                        q_values_next = target_estimator.predict(sess, next_states_batch)
                        targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * discount_factor * np.amax(q_values_next, axis=1)

                        # Perform gradient descent update
                        states_batch = np.array(states_batch)
                        loss = q_estimator.update(sess, states_batch, action_batch, targets_batch)
                        print("Step {} ({}) @ Episode {}/{}, action {}, reward {},loss: {}".format(t, total_t, i_episode + 1, num_episodes, action, reward, loss))
                        f.write("Step {} ({}) @ Episode {}/{}, action {}, reward {},loss: {}\n".format(t, total_t, i_episode + 1, num_episodes, action, reward, loss))


                        # Counting number of correct localized objects
                        if reward == 3:
                            num_located += 1

                        state = next_state
                        t += 1
                        total_t += 1
                        e = e + loss
                        r = r + reward

                    episode_counter += 1

                    # Add summaries to tensorboard
                    episode_summary = tf.Summary()
                    episode_summary.value.add(simple_value=epsilon, tag="episode/epsilon")
                    episode_summary.value.add(simple_value=r, tag="episode/reward")
                    episode_summary.value.add(simple_value=t, tag="episode/length")
                    episode_summary.value.add(simple_value=current_process.cpu_percent(), tag="system/cpu_usage_percent")
                    episode_summary.value.add(simple_value=current_process.memory_percent(), tag="system/v_memeory_usage_percent")
                    q_estimator.summary_writer.add_summary(episode_summary, episode_counter)
                    q_estimator.summary_writer.flush()

                    print("Episode Reward: {} Episode Length: {}".format(r, t))
                    f.write("Episode Reward: {} Episode Length: {}".format(r, t))

                    elist.append(float(e)/t)
                    rlist.append(float(r)/t)

		break


    f.close()
    print "number of correct located objects:{}".format(num_located)



'''
    plt.xlabel("episods")
    plt.ylabel("avg reward per epi")
    plt.title("num of correct obj localisation:{0}".format(num_located))
    plt.plot(rlist)
    plt.savefig("reward")
    print experiment_dir+"/graphs/reward"
    plt.close()


    # In[ ]:


    plt.xlabel("episods")
    plt.ylabel("error")
    plt.plot(elist)
    plt.savefig(experiment_dir+"/graphs/error")
'''
