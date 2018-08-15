# coding: utf-8


import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import cv2 as cv
import random




# ACTIONS
MOVE_RIGHT         = 0
MOVE_DOWN          = 1
SCALE_UP           = 2
ASPECT_RATIO_UP    = 3
MOVE_LEFT          = 4
MOVE_UP            = 5
SCALE_DOWN         = 6
ASPECT_RATIO_DOWN  = 7
SPLIT_HORIZONTAL   = 8
SPLIT_VERTICAL     = 9
PLACE_LANDMARK     = 10
SKIP_REGION        = 11



DIMENSION = 84
STEP_FACTOR = 0.2
MAX_ASPECT_RATIO = 6.00
MIN_ASPECT_RATIO = 0.15
MIN_BOX_SIDE     = 10




class ObjLocaliser(object):

    def __init__(self, image, boundingBoxes):

        #Loading image using PIL librarry to resize it to standard dimention which is acceptable by the network
        #PILimg = Image.fromarray(image)
        PILimg = image
        #Resizing the image to be compatible to the network
        resized_img = cv.resize(PILimg,(DIMENSION,DIMENSION))
        #Image is stored as np array
        self.image_playground = np.array(resized_img)
        self.yscale = float(DIMENSION)/image.shape[0]
        self.xscale = float(DIMENSION)/image.shape[1]
        self.targets = self.gettingTargerReady(boundingBoxes)

        #Initializing sliding window from top left corner of the image
        self.agent_window = np.array([0,0,DIMENSION,DIMENSION])
        self.iou = 0
        #self.memory_replay = []
        #self.memorySize = memorySize
        #self.batch_size = batch_size



        self.actoins = {
            0: 'MOVE_RIGHT',
            1: 'MOVE_DOWN',
            2: 'SCALE_UP',
            3: 'ASPECT_RATIO_UP',
            4: 'MOVE_LEFT',
            5: 'MOVE_UP',
            6: 'SCALE_DOWN',
            7: 'ASPECT_RATIO_DOWN',
            8: 'SPLIT_HORIZONTAL',
            9: 'SPLIT_VERTICAL',
            10: 'PLACE_LANDMARK',
            11: 'SKIP_REGION'
        }



    def Reset(self,image):
        self.agent_window = np.array([0,0,DIMENSION,DIMENSION])
        PILimg = image
        resized_img = cv.resize(PILimg,(DIMENSION,DIMENSION))
        self.image_playground = np.array(resized_img)


    def gettingTargerReady(self, boundingBoxes):
        numOfObj = len(boundingBoxes['xmax'])
        objs = []
        for i in range(numOfObj):
            temp = [boundingBoxes['xmin'][i]*self.xscale, boundingBoxes['ymin'][i]*self.yscale, boundingBoxes['xmax'][i]*self.xscale, boundingBoxes['ymax'][i]*self.yscale]
            objs.append(temp)
        return objs

    def wrapping(self):

        #Pick selected window from image
        im2 = self.image_playground[self.agent_window[1]:self.agent_window[3],self.agent_window[0]:self.agent_window[2]]
        #Resizing the agent window to be compatible for network input
        #resized = Image.fromarray(im2).resize((DIMENSION,DIMENSION))
        resized = cv.resize(im2,(DIMENSION,DIMENSION))
        #resized = np.stack((resized[:,:,0],resized[:,:,1],resized[:,:,2],self.image_playground[:,:,0],self.image_playground[:,:,1],self.image_playground[:,:,2]), axis=2)

        return resized


    """def pushingToMemory(self, s1, s2, rew, act):

        if len(self.memory_replay) < self.memorySize:
            self.memory_replay.append([s1, s2, rew, act])
        else:
            del self.memory_replay[0]
            self.memory_replay.append([s1, s2, rew, act])

    def sampleBatch(self):

        s1 = []
        s2 = []
        rew = []
        act = []

        for i in sorted(random.sample(xrange(len(self.memory_replay)), self.batch_size)):
            s1.append(self.memory_replay[i][0])
            s2.append(self.memory_replay[i][1])
            rew.append(self.memory_replay[i][2])
            act.append(self.memory_replay[i][3])

        return s1, s2, rew, act"""



    def takingActions(self,action):

        newbox = np.array([0,0,0,0])
        termination = False
        if action == MOVE_RIGHT:
            newbox = self.MoveRight()
        elif action == MOVE_DOWN:
            newbox = self.MoveDown()
        elif action == SCALE_UP:
            newbox = self.scaleUp()
        elif action == ASPECT_RATIO_UP:
            newbox = self.aspectRatioUp()
        elif action == MOVE_LEFT:
            newbox = self.MoveLeft()
        elif action == MOVE_UP:
            newbox = self.MoveUp()
        elif action == SCALE_DOWN:
            newbox = self.scaleDown()
        elif action == ASPECT_RATIO_DOWN:
            newbox = self.aspectRatioDown()
        elif action == SPLIT_HORIZONTAL:
            newbox = self.splitHorizontal()
        elif action == SPLIT_VERTICAL:
            newbox = self.splitVertical()
        elif action == PLACE_LANDMARK:
            newbox = self.placeLandmark()
            termination = True
        #elif action == SKIP_REGION:
            #self.skipRegion()

        self.agent_window = newbox
        self.adjustAndClip()
        r, new_iou = self.ComputingReward(self.agent_window, termination)
        self.iou = new_iou

        return r




    def MoveRight(self):

        newbox = np.copy(self.agent_window)
        boxW = newbox[2] - newbox[0]
        step = STEP_FACTOR * boxW
        # This action preserves box width and height
        if newbox[2] + step < self.image_playground.shape[0]:
            newbox[0] += step
            newbox[2] += step
        else:
            newbox[0] = self.image_playground.shape[0] - boxW - 1
            newbox[2] = self.image_playground.shape[0] - 1

        return newbox



    def MoveDown(self):

        newbox = np.copy(self.agent_window)
        boxH = newbox[3] - newbox[1]
        step = STEP_FACTOR * boxH
        # This action preserves box width and height
        if newbox[3] + step < self.image_playground.shape[1]:
            newbox[1] += step
            newbox[3] += step
        else:
            newbox[1] = self.image_playground.shape[1] - boxH - 1
            newbox[3] = self.image_playground.shape[1] - 1

        return newbox





    def scaleUp(self):

        newbox = np.copy(self.agent_window)
        boxW = newbox[2] - newbox[0]
        boxH = newbox[3] - newbox[1]

        # This action preserves aspect ratio
        widthChange = STEP_FACTOR * boxW
        heightChange = STEP_FACTOR * boxH

        if boxW + widthChange < self.image_playground.shape[0]:
            if boxH + heightChange < self.image_playground.shape[1]:
                newDelta = STEP_FACTOR
            else:
                newDelta = self.image_playground.shape[1] / boxH - 1
        else:
            newDelta = self.image_playground.shape[0] / boxW - 1
            if boxH + (newDelta * boxH) >= self.image_playground.shape[1]:
                newDelta = self.image_playground.shape[1] / boxH - 1

        widthChange = newDelta * boxW / 2.0
        heightChange = newDelta * boxH / 2.0
        newbox[0] -= widthChange
        newbox[1] -= heightChange
        newbox[2] += widthChange
        newbox[3] += heightChange

        return newbox




    def aspectRatioUp(self):

        newbox = np.copy(self.agent_window)
        boxH = newbox[3] - newbox[1]
        boxW = newbox[2] - newbox[0]

        # This action preserves width
        heightChange = STEP_FACTOR * boxH

        if boxH + heightChange < self.image_playground.shape[1]:
            ar = (boxH + heightChange) / boxW
            if ar < MAX_ASPECT_RATIO:
                newDelta = STEP_FACTOR
            else:
                newDelta = 0.0
        else:
            newDelta = self.image_playground.shape[1] / boxH - 1
            ar = (boxH + newDelta * boxH) / boxW
            if ar > MAX_ASPECT_RATIO:
                newDelta =  0.0

        heightChange = newDelta * boxH / 2.0
        newbox[1] -= heightChange
        newbox[3] += heightChange


        return newbox



    def MoveLeft(self):

        newbox = np.copy(self.agent_window)
        boxW = newbox[2] - newbox[0]
        step = STEP_FACTOR * boxW

        # This action preserves box width and height
        if newbox[0] - step >= 0:
            newbox[0] -= step
            newbox[2] -= step
        else:
            newbox[0] = 0
            newbox[2] = boxW

        return newbox


    def MoveUp(self):

        newbox = np.copy(self.agent_window)

        boxH = newbox[3] - newbox[1]

        step = STEP_FACTOR * boxH
        # This action preserves box width and height
        if newbox[1] - step >= 0:
            newbox[1] -= step
            newbox[3] -= step
        else:
            newbox[1] = 0
            newbox[3] = boxH

        return newbox


    def scaleDown(self):

        newbox = np.copy(self.agent_window)

        boxH = newbox[3] - newbox[1]
        boxW = newbox[2] - newbox[0]

        # This action preserves aspect ratio
        widthChange = STEP_FACTOR * boxW
        heightChange = STEP_FACTOR * boxH

        if boxW - widthChange >= MIN_BOX_SIDE:
            if boxH - heightChange >= MIN_BOX_SIDE:
                newDelta = STEP_FACTOR
            else:
                newDelta = MIN_BOX_SIDE / boxH - 1
        else:
            newDelta = MIN_BOX_SIDE / boxW - 1
            if  boxH - newDelta *  boxH < MIN_BOX_SIDE:
                newDelta = MIN_BOX_SIDE /  boxH - 1
        widthChange = newDelta * boxW / 2.0
        heightChange = newDelta * boxH / 2.0
        newbox[0] += widthChange
        newbox[1] += heightChange
        newbox[2] -= widthChange
        newbox[3] -= heightChange

        return newbox



    def splitHorizontal(self):

        newbox = np.copy(self.agent_window)
        boxW = newbox[2] - newbox[0]
        if boxW >  MIN_BOX_SIDE:
            half = boxW / 2.0
            newbox[2] -= half
        return newbox


    def splitVertical(self):

        newbox = np.copy(self.agent_window)
        boxH = newbox[3] - newbox[1]
        if boxH > MIN_BOX_SIDE:
            half = boxH/2.0
            newbox[3] -= half
        return newbox


    def aspectRatioDown(self):

        newbox = np.copy(self.agent_window)

        boxW = newbox[2] - newbox[0]
        boxH = newbox[3] - newbox[1]

        # This action preserves height
        widthChange = STEP_FACTOR * boxW
        if boxW + widthChange < self.image_playground.shape[0]:
            ar = boxH / (boxW + widthChange)
            if ar >= MIN_ASPECT_RATIO:
                newDelta = STEP_FACTOR
            else:
                newDelta = 0.0
        else:
            newDelta = self.image_playground.shape[0] / boxW - 1
            ar = boxH / (boxW + newDelta * boxW)
            if ar < MIN_ASPECT_RATIO:
                newDelta =  0.0
        widthChange = newDelta * boxW / 2.0
        newbox[0] -= widthChange
        newbox[2] += widthChange

        return newbox

    def placeLandmark(self):

        newbox = np.copy(self.agent_window)

        h = (newbox[3] - newbox[1])/2
        h_l = h/5
        w = (newbox[2] - newbox[0])/2
        w_l = w/5

        self.image_playground[newbox[1]+h-h_l:newbox[1]+h+h_l,newbox[0]:newbox[2]] = 0
        self.image_playground[newbox[1]:newbox[3],newbox[0]+w-w_l:newbox[0]+w+w_l] = 0

        return newbox





    def adjustAndClip(self):

        #Cheching if x coordinate of the top left corner is out of bound
        if self.agent_window[0] < 0:
            step = -self.agent_window[0]
            if self.agent_window[2] + step < self.image_playground.shape[0]:
                self.agent_window[0] += step
                self.agent_window[2] += step
            else:
                self.agent_window[0] = 0
                self.agent_window[2] = self.image_playground.shape[0] - 1

        #Cheching if y coordinate of the top left corner is out of bound
        if self.agent_window[1] < 0:
            step = -self.agent_window[1]
            if self.agent_window[3] + step < self.image_playground.shape[1]:
                self.agent_window[1] += step
                self.agent_window[3] += step
            else:
                self.agent_window[1] = 0
                self.agent_window[3] = self.image_playground.shape[1] - 1

        #Cheching if x coordinate of the bottom right corner is out of bound
        if self.agent_window[2] >= self.image_playground.shape[0]:
            step = self.agent_window[2] - self.image_playground.shape[0]
            if self.agent_window[0] - step >= 0:
                self.agent_window[0] -= step
                self.agent_window[2] -= step
            else:
                self.agent_window[0] = 0
                self.agent_window[2] = self.image_playground.shape[0] - 1

        #Cheching if y coordinate of the bottom right corner is out of bound
        if self.agent_window[3] >= self.image_playground.shape[1]:
            step = self.agent_window[3] - self.image_playground.shape[1]
            if self.agent_window[1] - step >= 0:
                self.agent_window[1] -= step
                self.agent_window[3] -= step
            else:
                self.agent_window[1] = 0
                self.agent_window[3] = self.image_playground.shape[1] - 1

        if self.agent_window[0] == self.agent_window[2]:
            if self.agent_window[2] + MIN_BOX_SIDE < self.image_playground.shape[0]:
                self.agent_window[2] = self.agent_window[2] + MIN_BOX_SIDE
            else:
                self.agent_window[0] = self.agent_window[0] - MIN_BOX_SIDE

        if self.agent_window[1] == self.agent_window[3]:
            if self.agent_window[3] + MIN_BOX_SIDE < self.image_playground.shape[1]:
                self.agent_window[3] = self.agent_window[3] + MIN_BOX_SIDE
            else:
                self.agent_window[1] = self.agent_window[1] - MIN_BOX_SIDE


    def intersectionOverUnion(self, boxA, boxB):

        # determine the (x, y)-coordinates of the intersection rectangle
        #boxA = self.agent_window
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground_truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        return iou


    def ComputingReward(self, agent_window, termination = False):
	max_iou = -2
	reward = 0
        for target in self.targets: #for checking all objs change [self.targets[0]] ---> self.targets
            new_iou = self.intersectionOverUnion(agent_window, np.array(target))
	    if new_iou > max_iou:
		max_iou = new_iou
            	reward = self.ReturnReward(new_iou, termination)
            #if reward > 0:
                #break
        if termination: max_iou = 0
        return reward, max_iou

    """def ComputingReward(self, agent_window, termination = False):
	new_iou = 0
	reward = 0
        for target in self.targets: #for checking all objs change [self.targets[0]] ---> self.targets
            new_iou = self.intersectionOverUnion(agent_window, np.array(target))
            reward = self.ReturnReward(new_iou, termination)
            if reward > 0:
                break
        if termination: new_iou = 0
        return reward, new_iou"""


    def ReturnReward(self,new_iou, termination):
        reward = 0

        if new_iou - self.iou > 0:
            reward = 1
        else:
            reward = -1

        if termination:

            if (new_iou > 0.5):
                reward = 3
            else:
                reward = -3
            self.iou = 0
        return reward


    def drawActions(self):

        # Create figure and axes
        fig,ax = plt.subplots(1)

        # Display the image
        ax.imshow(self.image_playground)
        
        # Drawing agent window
        rect = patches.Rectangle((self.agent_window[0],self.agent_window[1]),self.agent_window[2]-self.agent_window[0],self.agent_window[3]-self.agent_window[1],linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)

        # Drawing target objects bouning boxes
        for target in self.targets:
            rect2 = patches.Rectangle((target[0],target[1]),target[2]-target[0],target[3]-target[1],linewidth=1,edgecolor='b',facecolor='none')
            ax.add_patch(rect2)



        # Add the patch to the Axes
        plt.draw()
        plt.show()
        
        
    def my_draw(self):
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        from matplotlib.figure import Figure
        
        fig = Figure()
        canvas = FigureCanvas(fig)
        ax = fig.gca()
       
        ax.imshow(self.image_playground)
        
        # Drawing agent window
        rect = patches.Rectangle((self.agent_window[0],self.agent_window[1]),self.agent_window[2]-self.agent_window[0],self.agent_window[3]-self.agent_window[1],linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)

        # Drawing target objects bouning boxes
        for target in [self.targets[0]]:
            rect2 = patches.Rectangle((target[0],target[1]),target[2]-target[0],target[3]-target[1],linewidth=1,edgecolor='b',facecolor='none')
            ax.add_patch(rect2)
            
        canvas.draw()       # draw the canvas, cache the renderer
        
        width, height = fig.get_size_inches() * fig.get_dpi()

        return np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
        

    def smartExploration(self):

        action_set = []
        newbox = self.MoveRight()
        r, _ = self.ComputingReward(newbox)
        if r>0:
            action_set.append(0)
        newbox = self.MoveDown()
        r, _ = self.ComputingReward(newbox)
        if r>0:
            action_set.append(1)
        newbox = self.scaleUp()
        r, _ = self.ComputingReward(newbox)
        if r>0:
            action_set.append(2)
        newbox = self.aspectRatioUp()
        r, _ = self.ComputingReward(newbox)
        if r>0:
            action_set.append(3)
        newbox = self.MoveLeft()
        r, _ = self.ComputingReward(newbox)
        if r>0:
            action_set.append(4)
        newbox = self.MoveUp()
        r, _ = self.ComputingReward(newbox)
        if r>0:
            action_set.append(5)
        newbox = self.scaleDown()
        r, _ = self.ComputingReward(newbox)
        if r>0:
            action_set.append(6)
        newbox = self.aspectRatioDown()
        r, _ = self.ComputingReward(newbox)
        if r>0:
            action_set.append(7)
        newbox = self.splitHorizontal()
        r, _ = self.ComputingReward(newbox)
        if r>0:
            action_set.append(8)
        newbox = self.splitVertical()
        r, _ = self.ComputingReward(newbox)
        if r>0:
            action_set.append(9)
        newbox = self.agent_window
        r, _ = self.ComputingReward(newbox)
        if r>0:
            action_set.append(10)
        act = 0

        if len(action_set) < 1:
            act = np.random.randint(11, size=1)[0]
        else:
            ind = (random.sample(xrange(len(action_set)), 1))[0]
            act = action_set[ind]

        return act

