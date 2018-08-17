
# coding: utf-8

# In[1]:


get_ipython().magic(u'matplotlib inline')

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
    
import plotting
from collections import deque, namedtuple
from readingFileEfficiently import *
import VOC2012_npz_files_writter
from DNN import *
from Agent import ObjLocaliser


# In[2]:


#This cell reads VOC 2012 dataset and save them in .npz files for future.
#The process of reading data and put them in prper format is time consuming so they are stored in a file.

xml_path = "../VOC2012/Annotations/*.xml"
destination = "../data/"

#It splits dataset to 80% for training and 20% validation.
if not (os.path.isfile(destination+"test_input.npz") or os.path.isfile(destination+"test_target.npz")):
    VOC2012_npz_files_writter.writting_files(xml_path, destination, percentage=0)
    print("Files are ready!!!")
else:
    print("Records are already prepared!!!")


# In[3]:


num_episodes=5  #200     
replay_memory_size=500000    #2500   
replay_memory_init_size=500 #500  
update_target_estimator_every=10000 #100  
discount_factor=0.99
epsilon_start=1.0
epsilon_end=0.1
epsilon_decay_steps=500000 #10000  
batch_size=32
category = "cat"

#model_name = "defaul_DQL_architecture_epis{}_memorySize{}_UTE{}_EDS{}".format(num_episodes, replay_memory_size, update_target_estimator_every, epsilon_decay_steps)
model_name = "test"


tf.reset_default_graph()

# Where we save our checkpoints and graphs
experiment_dir = os.path.abspath("./experiments/{}".format(model_name))

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
    sess.run(tf.global_variables_initializer()) 
    # Old API: sess.run(tf.initialize_all_variables())  
    Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

    # The replay memory
    replay_memory = []
    
    num_located = 0

    # Make model copier object
    estimator_copy = ModelParametersCopier(q_estimator, target_estimator)

    # Keeps track of useful statistics
    #stats = plotting.EpisodeStats(
        #episode_lengths=np.zeros(num_episodes),
        #episode_rewards=np.zeros(num_episodes))

    # For 'system/' summaries, usefull to check if currrent process looks healthy
    current_process = psutil.Process()

    # Create directories for checkpoints and summaries
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "model")
    report_path = os.path.join(experiment_dir, "report")

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(report_path):
        os.makedirs(report_path)
    f = open(report_path+"/log.txt", 'w')
    saver = tf.train.Saver()
    # Load a previous checkpoint if we find one
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        print("Loading model checkpoint {}...\n".format(latest_checkpoint))
        saver.restore(sess, latest_checkpoint)

    # Get the current time step
    total_t = sess.run(tf.contrib.framework.get_global_step())
    #print "init:{}".format(total_t)

    # The epsilon decay schedule
    epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

    # The policy we're following
    policy = make_epsilon_greedy_policy(
        q_estimator,
        len(VALID_ACTIONS))
    episode_counter = 0
    #mybreak = 0
    
    for indx,tmp in enumerate(extractData(category, "train", batch_size)):

        #if mybreak > 5:
            #break
        
        img=tmp[0]
        target=tmp[1]

        im2 = Image.frombytes("RGB",(img['image_width'],img['image_height']),img['image'])
        env = ObjLocaliser(np.array(im2),target)
        print "New image is being loaded: {}".format(img['image_filename'])
        
        if len(replay_memory) < replay_memory_init_size:
            
            # Populate the replay memory with initial experience
            print("Populating replay memory...\n")

            env.Reset(np.array(im2))
            state = env.wrapping()


            state = state_processor.process(sess, state)
            state = np.stack([state] * 4, axis=2)

            for i in range(replay_memory_init_size):

                #env.Reset(np.array(im2))
                #state = env.wrapping()
                #state = state_processor.process(sess, state)
                #state = np.stack([state] * 4, axis=2)
                #action = 0
                #counter = 0
                #done = False

                #while (action != 10) or (counter < 50):

                action_probs, _ = policy(sess, state, epsilons[min(total_t, epsilon_decay_steps-1)])
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
                replay_memory.append(Transition(state, action, reward, next_state, done))
                state = next_state

                #counter += 1

                if done:
                    #state = env.reset()
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
            #state = env.reset()
            env.Reset(np.array(im2))
            state = env.wrapping()
            state = state_processor.process(sess, state)
            state = np.stack([state] * 4, axis=2)
            loss = None
            t=0
            action = 0
            e = 0
            r = 0
            #done = False
            # One step in the environment
            while (action != 10) and (t < 50):
                #print "hello22:{}".format(loss)
                #env.drawActions()
                # Epsilon for this time step
                epsilon = epsilons[min(total_t, epsilon_decay_steps-1)]

                # Maybe update the target estimator
                if total_t % update_target_estimator_every == 0:
                    estimator_copy.make(sess)
                    print("\nCopied model parameters to target network.")

                # Print out which step we're on, useful for debugging.
                #sys.stdout.flush()

                # Take a step
                #print epsilon
                action_probs, qs = policy(sess, state, epsilon)
                print qs
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)


                #next_state, reward, done, _ = env.step(VALID_ACTIONS[action])
                reward = env.takingActions(VALID_ACTIONS[action])
                env.drawActions()
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
                # Update statistics
                #stats.episode_rewards[i_episode] += reward
                #stats.episode_lengths[i_episode] = t

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


                # Counting number of correct localsied objects
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

            #prin_stats = plotting.EpisodeStats(
                #episode_lengths=stats.episode_lengths[:i_episode+1],
                #episode_rewards=stats.episode_rewards[:i_episode+1])

            #print("Episode Reward: {} Episode Length: {}".format(prin_stats.episode_rewards[-1], prin_stats.episode_lengths[-1]))
            #f.write("Episode Reward: {} Episode Length: {}".format(prin_stats.episode_rewards[-1], prin_stats.episode_lengths[-1]))
            print("Episode Reward: {} Episode Length: {}".format(r, t))
            f.write("Episode Reward: {} Episode Length: {}".format(r, t))

            elist.append(float(e)/t)
            rlist.append(float(r)/t)
        break
        #mybreak += 1

        
f.close()
print "number of correct located objects:{}".format(num_located)


# In[4]:


plt.xlabel("episods")
plt.ylabel("avg reward per epi")
plt.title("num of correct obj localisation:{0}".format(num_located))
plt.plot(rlist)
plt.savefig("./graphs/reward")
plt.close()


# In[5]:


plt.xlabel("episods")
plt.ylabel("error")
plt.plot(elist)
plt.savefig("./graphs/error")


# link to code: https://github.com/dennybritz/reinforcement-learning/blob/master/DQN/Deep%20Q%20Learning%20Solution.ipynb
# 
# https://www.oreilly.com/ideas/reinforcement-learning-with-tensorflow
