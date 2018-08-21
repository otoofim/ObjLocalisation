# ObjLocalisation

The aim of this project is to extend the algorithm ["active object localization"](https://arxiv.org/abs/1511.06015). In this project, I am going to use reinforcemnt learning for object localisation task. Currently, all the methods use a sliding window to process an entire image in differnt scale to localise an object however, the method proposed by "active objet localisation" is to find an object efficiently by looking for visual clue in the image and using those clues the agent tries to localise an object. Pascal VOC dataset is used for training. The final goal is to learn a policy to localise an object in a scene with a sequence of actions including zooming in, zooming out, left, right, up, and down. Deep reinforcement learning is used to learn the policy.

# Overview

This implementation is divided five parts:

  -  Input pipeline: this pipeline reads Pascal VOC 2012 images and their annotations. Having read the data, the pipline writes data into .npz files which are later used to train the models. This pipeline includes VOC2012DataProvider.py, VOC2012_npz_files_writter.py, and readingFileEfficiently.py files.
  -  Agent: implementation of the agent which holds image playground and the current window. The defination of the agent class is in the file Agent.py. Each instace of Agent class gets as input an image and its ground truth (bounding boxes). This implementation primerly adopted methods from the [orginal implementation of active object localization](https://github.com/jccaicedo/localization-agent).
  - Neural network implementation: The file DNN.py consist of Tensorflow implementation of DQL. To implement DQL with Tensorflow, [this tutorial](https://github.com/dennybritz/reinforcement-learning) is mainly used.
  - Setting up environment and traning models: the files DQL.py, DQL_testing.py, DQL_visualization_actions.py, and DQL_visualization-layers.py are using above stages to provide an easy to use interface for traning, evaluating, and visualizing DQL.
  
You can use the files above by installing requierments using `pip install -r requirements.txt`.

# Files description:

#### DQL.py: 
  Around line 50 hyper-parameters can be set. This file includes the main for loop for training.
#### DQL_visualization.py:
Using this file convolutional layers can be visualized.
#### DQL_testing.py:
This evaluate a trained model on testdata.



## TO do list:

- [ ] Imlement active object localisation
  - [x] writting data provider for Pascal VOC2012
  - [x] Designing the neural network architecture as described in the paper
  - [ ] Implementing the scene processing method to conduct the agent actions and return the result after each action taken.
    - [x] initialising agent window
    - [x] completing "wrapping" function. This function is responsible to prepare the image after each action is taken. 
       suggestion: Do not try to wrap the image instead you can resize it although its resolution would decrease. 
    - [x] Completing skip region and put mark actions
    - [x] adding error function
  - [x] Trining the network to localise one object in the scene
  - [x] Model is unstable. Error and mean reward per episode diverge
    - [x] Add experience replay
    - [x] Add separate network for generating the targets 
  - [ ] Training the network to localise multiple object
  - [ ] Load a pretrained CNN for the first part of the network
- [ ] Combine retina work to "active object localisation" and report how much it improves efficiency  
- [ ] Would be grate if I can also use ["Weakly-supervised learning"](http://leon.bottou.org/publications/pdf/cvpr-2015.pdf) to make the whole process independent from needing to have annotated data.  


## Possible extensions:

- [ ] Train agent to learn how to change alpha
- [ ] Adding a new action for agent to be able to finish the seach in an image 
- [ ] Training the network to detect objects as well in an end-to-end manner
