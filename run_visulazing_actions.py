import os
import sys
import argparse


if "./lib" not in sys.path:
    sys.path.append("./lib")

from DQL_visualization_actions import *





if __name__== "__main__":


    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser = argparse.ArgumentParser(description='Train an object localizer')

    parser.add_argument('-m','--model_name', type=str, default='default_model', help='The model parameters that will be loaded for testing. The model should be placed in ../experiments/model_name. Default: default_model')
    parser.add_argument('-i','--image_path', type=str, default=None, help='Path to an image.')
    parser.add_argument('-g','--ground_truth', type=int, nargs='+', default=[0,0,1,1], help='Target coordinates. The order of coordinates should be like: xmin ymin xmax ymax. Default: 0 0 1 1')
    parser.add_argument('-n','--name', type=str, default="anim", help='Name of the output file. It will be stord in ../experiments/model_name/anim/')

    args = parser.parse_args()


    print args.ground_truth 
    visualizing_seq_act(args.model_name,
	args.image_path,
	args.ground_truth,
	args.name)



