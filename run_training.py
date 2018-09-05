#import itertools
#import numpy as np
import os
#import random
import sys
#import psutil
#import tensorflow as tf
#from PIL import Image
#import matplotlib.pyplot as plt
import argparse
#plt.switch_backend('agg')

if "./lib" not in sys.path:
    sys.path.append("./lib")

#import plotting
#from collections import deque, namedtuple
#from readingFileEfficiently import *
#import VOC2012_npz_files_writter
#from DNN import *
#from Agent import ObjLocaliser
from DQL import *





if __name__== "__main__":


    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser = argparse.ArgumentParser(description='Train an object localizer')

    parser.add_argument('-n','--num_episodes', type=int, default=5, help = "Number of episodes that the agect can interact with an image. Default: 5")
    parser.add_argument('-rms','--replay_memory_size', type=int, default=500000, help = "Number of the most recent experiences that would be stored. Default: 500000")
    parser.add_argument('-rmis','--replay_memory_init_size', type=int, default=500, help = "Number of experiences to initialize replay memory. Default: 500")
    parser.add_argument('-u','--update_target_estimator_every', type=int, default=10000, help = "Number of steps after which estimator parameters are copied to target network. Default: 10000")
    parser.add_argument('-d','--discount_factor', type=int, default=0.99, help = "Discount factor. Default: 0.99")
    parser.add_argument('-es','--epsilon_start', type=int, default=1.0, help = "Epsilon decay schedule start point. Default: 1.0")
    parser.add_argument('-ee','--epsilon_end', type=int, default=0.2, help="Epsilon decay schedule end point. Default: 0.2")
    parser.add_argument('-ed','--epsilon_decay_steps', type=int, default=500, help="Epsilon decay step rate. This number indicates epsilon would be decearsed from start to end point after how many steps. Default: 500")
    #parser.add_argument('-b','--batch_size', type=int, default=32, help="Epsilon decay schedule end point. Default: 0.2")
    parser.add_argument('-c','--category', type=str, nargs='+', default=['cat'], help='Indicating the categories are going to be used for training. You can list name of the classes you want to use in training. If you wish to use all classes then you can use *. For instnce <-c cat dog>. Default: cat')
    parser.add_argument('-m','--model_name', type=str, default='default_model', help='The trained model would be saved with this name under the path ../experiments/model_name. Default: default_model')
    #parser.add_argument('-co','--coco', type=bool, default=False)

    args = parser.parse_args()


    DQL(args.num_episodes,
        args.replay_memory_size,
        args.replay_memory_init_size,
        args.update_target_estimator_every,
        args.discount_factor,
        args.epsilon_start,
        args.epsilon_end,
        args.epsilon_decay_steps,
        args.category,
        args.model_name)



