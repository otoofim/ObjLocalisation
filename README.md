# ObjLocalisation

The aim of this project is to extend the algorithm ["active object localization"](https://arxiv.org/abs/1511.06015). In this project, I am going to use reinforcemnt learning for object localisation task. Currently, all the methods use a sliding window to process whole an image in differnt scale to localise an image however, the method proposed by "active objet localisation" is to find an object efficiently by looking for visual clue in the image and using those clues the agent tries to localise an object. Pascal VOC dataset is used for training. The final goal is to learn a policy to localise an object in a scene with a sequence of actions including zooming in, zooming out, left, right, up, and down. Deep reinforcement learning is used to learn the policy.

## TO do list:

- [ ] Imlement active object localisation
  - [x] writting data provider for Pascal VOC2012
  - [x] Designing the neural network architecture as described in the paper
  - [ ] Implementing the scene processing method to conduct the agent actions and return the result after each action taken.
    - [ ] initialising agent window
    - [ ] completing "wrapping" function. This function is responsible to prepare the image after each action is taken. 
       suggestion: Do not try to wrap the image instead you can resize it although its resolution would decrease. 
  - [ ] Trining the network to localise one object in the scene
  - [ ] Training the network to localise multiple object
  - [ ] Load a pretrained CNN for the first part of the network
- [ ] Combine retina work to "active object localisation" and report how much it improves efficiency  
- [ ] Would be grate if I can also use ["Weakly-supervised learning"](http://leon.bottou.org/publications/pdf/cvpr-2015.pdf) to make the whole process independent from needing to have annotated data.  
