import os
import sys
import argparse

if "./lib" not in sys.path:
    sys.path.append("./lib")

from DQL_testing import *





if __name__== "__main__":


    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser = argparse.ArgumentParser(description='Train an object localizer')

    parser.add_argument('-n','--num_episodes', type=int, default=15, help = "Number of episodes that the agent can interact with an image. Default: 15")
    parser.add_argument('-c','--category', type=str, nargs='+', default=['cat'], help='Indicating the categories are going to be used for training. You can list name of the classes you want to use in testing. If you wish to use all classes then you can use *. For instnce <-c cat dog>. Default: cat')
    parser.add_argument('-m','--model_name', type=str, default='default_model', help='The trained model would be saved with this name under the path ../experiments/model_name. Default: default_model')

    args = parser.parse_args()
    map = []
    for category in args.category:
	print "{} images are being evaluated... \n\n\n\n".format(category)
        map.append(DQL_testing(args.num_episodes,
            category,
            args.model_name))
    print "MAP over the given category(s): {}".format(np.mean(map))
