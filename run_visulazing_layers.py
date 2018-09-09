import os
import sys
import argparse

if "./lib" not in sys.path:
    sys.path.append("./lib")

from DQL_visualization_layers import *





if __name__== "__main__":


    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser = argparse.ArgumentParser(description='Visualizing CNN layers')

    parser.add_argument('-m','--model_name', type=str, default='default_model', help='The model parameters that will be loaded for testing. Do not forget to put the model under the path ../experiments/model_name. Default: default_model')
    parser.add_argument('-i','--image_path', type=str, default=None, help='Path to an image.')
    parser.add_argument('-ln','--layer_num', type=str, default="1", help='Layer number you wish to visualize.')

    args = parser.parse_args()


    visualize_layers(args.model_name,
	args.image_path,
	args.layer_num)



