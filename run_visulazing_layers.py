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
from DQL_visualization_layers import *





if __name__== "__main__":


    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser = argparse.ArgumentParser(description='Train an object localizer')

    parser.add_argument('-m','--model_name', type=str, default='default_model', help='The model parameters that will be loaded for testing. The model should be placed in ../experiments/model_name. Default: default_model')
    parser.add_argument('-i','--image_path', type=str, default=None, help='Path to an image.')
    parser.add_argument('-ln','--layer_num', type=str, default="1", help='Layer number you wish to visualize.')

    args = parser.parse_args()


    visualize_layers(args.model_name,
	args.image_path,
	args.layer_num)



