# ObjLocalisation

The aim of this project is to learn a policy to localize objects in images by turning visual attention to the salient parts of images. In order to achieve this goal, the popular RL algorithm, Q-learning, is adopted by  incorporating the approximation method, CNNs. [DQL](https://www.nature.com/articles/nature14236) is the method resulting from cooperating Q-learning and CNNs. While using this method for object localization is not new and was tried before in [Active Object Localization with Deep Reinforcement Learning](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Caicedo_Active_Object_Localization_ICCV_2015_paper.pdf), in this project despite implementing the algorithm with the novel deep learning framework, Tensorflow, a new set of experiments were conducted by using a new neural network architecture to show that representation learning can happen by Q-learning. More specifically, the original paper uses a pre-trained CNN as a feature extractor. However, in this project,  the model was trained without using a pre-trained network for feature extraction. This MSc project was conducted in [Computer Vision & Autonomous Systems Group](https://www.gla.ac.uk/schools/computing/research/researchoverview/computervisionandautonomoussystems/) at the university of Glasgow under supervision of [Dr Jan Paul Siebert](http://www.dcs.gla.ac.uk/~psiebert/). Below is the examples of a trained model on VOC 2012 dataset. The following sections describe user manual for researhers who intend to use this implementation. In addition, in order to make modifying the implementation modest all the files are commented.


<div class="row" align="center">
  <div class="column" align="center">
    <img src="https://drive.google.com/uc?export=view&id=1QsOi-zVPicMfMej0OfFBV52cDQKGJbEh" width="900px" />
  </div>
</div>

# Getting started

You can clone this project using this [link](https://github.com/otoofim/ObjLocalisation.git) and install requierments by `pip install -r requirements.txt`. Despite `requirements.txt`, it is required  to installe Tensorflow. The code works fine for the latest version of Tensorflow. However, in order to run the code on cluster it requires some changes. In the file `DNN.py` the parts of the code that needs to be modified in order to run on GPU cluster is marked as Old API. In this way the code can run with the older Tensorflow APIs. It is recommended to follow [this tutorial](https://www.tensorflow.org/install/install_linux)  to create a virtual environment and then install Tensorflow and all requirements within that. This code was developed and tested on Ubuntu 16.04 using Python 2.7.12 and Tensorflow 1.8.




## Inputs

In this project [Pascal VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) dataset was used to train the model. It is organized to download and prepare the dataset for training in the first run. However, if you need to use another dataset then the input pipeline needs to be modified. To change the default dataset you need to make some changes in the files `VOC2012DataProvider.py`, `readingFileEfficiently.py`, and `VOC2012_npz_files_writter.py`. In Pacal VOC dataset the gorund truth is provided seperately in `xml` files. For this resean, it is needed to write images and their corrensponding ground truth to a single file, .npz, in order to create image batches for efficient learning. That is done by `VOC2012_npz_files_writter.py`. Later .npz files are used by `DQL.py` for training. Since Pascal VOC 2012 consists of 19386 images loading all images into memory makes trouble. For this, `readingFileEfficiently.py` loads input images into memory in an efficient way. Further, `VOC2012DataProvider.py` reads .npz files and provides datapoints to `readingFileEfficiently.py`.


## Command Line Options and Configuration

Having set up the environment, training can begin using `run_training.py`. Its command line options is as follow:

          usage: run_training.py [-h] [-n NUM_EPISODES] [-rms REPLAY_MEMORY_SIZE]
                       [-rmis REPLAY_MEMORY_INIT_SIZE]
                       [-u UPDATE_TARGET_ESTIMATOR_EVERY] [-d DISCOUNT_FACTOR]
                       [-es EPSILON_START] [-ee EPSILON_END]
                       [-ed EPSILON_DECAY_STEPS] [-c CATEGORY [CATEGORY ...]]
                       [-m MODEL_NAME]

          Train an object localizer

          optional arguments:
          -h, --help            show this help message and exit
          -n NUM_EPISODES, --num_episodes NUM_EPISODES
                        Number of episodes that the agect can interact with an
                        image. Default: 5
          -rms REPLAY_MEMORY_SIZE, --replay_memory_size REPLAY_MEMORY_SIZE
                        Number of the most recent experiences that would be
                        stored. Default: 500000
          -rmis REPLAY_MEMORY_INIT_SIZE, --replay_memory_init_size REPLAY_MEMORY_INIT_SIZE
                        Number of experiences to initialize replay memory.
                        Default: 500
          -u UPDATE_TARGET_ESTIMATOR_EVERY, --update_target_estimator_every UPDATE_TARGET_ESTIMATOR_EVERY
                        Number of steps after which estimator parameters are
                        copied to target network. Default: 10000
          -d DISCOUNT_FACTOR, --discount_factor DISCOUNT_FACTOR
                        Discount factor. Default: 0.99
          -es EPSILON_START, --epsilon_start EPSILON_START
                        Epsilon decay schedule start point. Default: 1.0
          -ee EPSILON_END, --epsilon_end EPSILON_END
                        Epsilon decay schedule end point. Default: 0.2
          -ed EPSILON_DECAY_STEPS, --epsilon_decay_steps EPSILON_DECAY_STEPS
                        Epsilon decay step rate. This number indicates epsilon
                        would be decearsed from start to end point after how
                        many steps. Default: 500
          -c CATEGORY [CATEGORY ...], --category CATEGORY [CATEGORY ...]
                        Indicating the categories are going to be used for
                        training. You can list name of the classes you want to
                        use in training. If you wish to use all classes then
                        you can use *. For instnce <-c cat dog>. Default: cat
          -m MODEL_NAME, --model_name MODEL_NAME
                        The trained model would be saved with this name under
                        the path ../experiments/model_name. Default:
                        default_model
                        
Note: If you need to train a model on multiple categories the command would be `python run_training.py -c cat dog`. In addition, if you want to trian a new mdoel on top of a previously trained model then you need to copy the content of the bestModel folder of the previously trained model to its checkpoints folder. In this way, the best model will be loaded for training.                   

To evaluate a trained model on the test set `run_testing.py` is used. Testing conditions can be set as below:

          usage: run_testing.py [-h] [-n NUM_EPISODES] [-c CATEGORY [CATEGORY ...]]
                      [-m MODEL_NAME]

          Evaluate a model on test set

          optional arguments:
          -h, --help            show this help message and exit
          -n NUM_EPISODES, --num_episodes NUM_EPISODES
                        Number of episodes that the agent can interact with an
                        image. Default: 15
          -c CATEGORY [CATEGORY ...], --category CATEGORY [CATEGORY ...]
                        Indicating the categories are going to be used for
                        training. You can list name of the classes you want to
                        use in testing. If you wish to use all classes then
                        you can use *. For instnce <-c cat dog>. Default: cat
          -m MODEL_NAME, --model_name MODEL_NAME
                        The trained model would be saved with this name under
                        the path ../experiments/model_name. Default:
                        default_model

There are two other python files that are useful for visualization purposes. `run_visulazing_actions.py` can be used to visualize a sequence of actions:

          usage: run_visulazing_actions.py [-h] [-m MODEL_NAME] [-i IMAGE_PATH]
                                 [-g GROUND_TRUTH [GROUND_TRUTH ...]]
                                 [-n NAME]

          Visualizing sequence of actions

          optional arguments:
            -h, --help            show this help message and exit
            -m MODEL_NAME, --model_name MODEL_NAME
                        The model parameters that will be loaded for testing.
                        The model should be placed in
                        ../experiments/model_name. Default: default_model
            -i IMAGE_PATH, --image_path IMAGE_PATH
                        Path to an image.
            -g GROUND_TRUTH [GROUND_TRUTH ...], --ground_truth GROUND_TRUTH [GROUND_TRUTH ...]
                        Target coordinates. The order of coordinates should be
                        like: xmin ymin xmax ymax. Default: 0 0 1 1
            -n NAME, --name NAME  Name of the output file. It will be stored in
                        ../experiments/model_name/anim/
 
In addition, the neural network layers can be visualized using `run_visulazing_layers.py`:
 
 
            usage: run_visulazing_layers.py [-h] [-m MODEL_NAME] [-i IMAGE_PATH]
                                [-ln LAYER_NUM]

            Visualizing CNN layers

            optional arguments:
              -h, --help            show this help message and exit
              -m MODEL_NAME, --model_name MODEL_NAME
                        The model parameters that will be loaded for testing.
                        The model should be placed in
                        ../experiments/model_name. Default: default_model
              -i IMAGE_PATH, --image_path IMAGE_PATH
                        Path to an image.
              -ln LAYER_NUM, --layer_num LAYER_NUM
                        Layer number you wish to visualize.
 
**Note:** In all visualization and evaluation files the best model saved in the directory of the given model is used. 

## Outputs

**run_training.py:**

  The output of trainng process is stored in `../experiments/ModelName`. The result will be saved in four folders. The first one is `summaries_q_estimator` consists of an TF event record. Using tensorboard, graphs related to the training  can be visualised. To run tensorboard, it is needed to call tensorboard in this way `tensorboard --logdir=../experiments/Modelname/summaries_q_estimator`. The second folder is report. This folder includes `log.txt` file which is the log showed in terminal during training. The third one is 'checkpoints' folder contains three files, which corresponds to the final model saved at the end of training process. The final folder is 'bestModel' that includes the best model based on validation accuracy.
  
**run_testing.py:**

The output of evaluation process is stored in `../experiments/ModelName/report/evaluate_[categories].txt`. That file consists of the results evaluated separately on each category and mean average precision (MAP) over all categories.

**run_visulazing_actions.py:**

The output of this file is a short video shows the agent interactions with the given image. The result is stored in `../experiments/ModelName/anim`. 

**run_visulazing_layers.py:**

The output of this script is a set of images each of which corresponds to a filter in a given layer. The result is stored in `../experiments/ModelName/visu`. 

# Acknowledgments
This code is implemented by getting help from the following sources:
- [Original implementation of active object localization algorithm](https://github.com/jccaicedo/localization-agent)
- [Tutorial for deep reinforcement learning](https://github.com/dennybritz/reinforcement-learning)
- [Tutorial for deep learning from the university of Edinburgh](https://github.com/otoofim/mlpractical)
