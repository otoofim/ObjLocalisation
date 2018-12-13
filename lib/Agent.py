import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import cv2 as cv
import random
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure




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



# Transformation coefficients
DIMENSION = 84
STEP_FACTOR = 0.2
MAX_ASPECT_RATIO = 6.00
MIN_ASPECT_RATIO = 0.15
MIN_BOX_SIDE     = 10




class ObjLocaliser(object):
    """
    Object localizer agent
    """

    def __init__(self, image, boundingBoxes):

        PILimg = image
        # Resizing the image to be compatible with the network
        resized_img = cv.resize(PILimg,(DIMENSION,DIMENSION))
        # Image is stored as np array
        self.image_playground = np.array(resized_img)
	# Computing the scale for transforming bounding boxes
        self.yscale = float(DIMENSION)/image.shape[0]
        self.xscale = float(DIMENSION)/image.shape[1]
	# Loading all ground truth to be used for computing IoU
        self.targets = self.gettingTargerReady(boundingBoxes)

        # Initializing sliding window from top left corner to the bottom right corner
        self.agent_window = np.array([0,0,DIMENSION,DIMENSION])
        self.iou = 0

 
    def Reset(self,image):
	"""
	Reset the agent window to the initial situation to prepare it for a new episode
	Args: 
	image: The image that the ageny is going to interact with
	"""

        self.agent_window = np.array([0,0,DIMENSION,DIMENSION])
        PILimg = image
        resized_img = cv.resize(PILimg,(DIMENSION,DIMENSION))
        self.image_playground = np.array(resized_img)


    def gettingTargerReady(self, boundingBoxes):
	"""
	Loading bounding boxes for an image. They are organized with this format [xmin, ymin, xmax, ymax]
	Args:
	boundingBoxes: A dictionary of boudong boxes
	returns:
	A list of bounding boxes
	"""

        numOfObj = len(boundingBoxes['xmax'])
        objs = []
        for i in range(numOfObj):
            temp = [boundingBoxes['xmin'][i]*self.xscale, boundingBoxes['ymin'][i]*self.yscale, boundingBoxes['xmax'][i]*self.xscale, boundingBoxes['ymax'][i]*self.yscale]
            objs.append(temp)
        return objs
 
    def wrapping(self):
	"""
	Resizing the agent current window to be compatible for the network. The window is resized to 84X84. It can be altered by DIMENSION variable
	Returns:
	The resized current window
	"""
        im2 = self.image_playground[self.agent_window[1]:self.agent_window[3],self.agent_window[0]:self.agent_window[2]]
        resized = cv.resize(im2,(DIMENSION,DIMENSION))
        return resized



    def takingActions(self,action):
	"""
	This function performs actions and computes the new window and its reward
	Args:
	action: Action that is going to be taken
	Returns: 
	Reward corrsponding to the taken action
	"""

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

	# Storing new window
        self.agent_window = newbox
	# Cheching whether the new window is out of boundaries
        self.adjustAndClip()
        # computing the reward and IoU corresponding to the taken action
        r, new_iou = self.ComputingReward(self.agent_window, termination)
        self.iou = new_iou

        return r


    def MoveRight(self):
	"""
	Action moving right
	Returns:
	New window
	"""

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
	"""
	Action moving down
	Returns:
	New window
	"""
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
	"""
	Action scaling up
	Returns:
	New window
	"""

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
	"""
	Action increasing aspect ratio
	Returns:
	New window
	"""

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
	"""
	Action moving left
	Returns:
	New window
	"""

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
	"""
	Action moving up
	Returns:
	New window
	"""

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
	"""
	Action moving down
	Returns:
	New window
	"""

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
	"""
	Action horizontal splitting
	Returns:
	New window
	"""

        newbox = np.copy(self.agent_window)
        boxW = newbox[2] - newbox[0]
        if boxW >  MIN_BOX_SIDE:
            half = boxW / 2.0
            newbox[2] -= half
        return newbox

   
    def splitVertical(self):
	"""
	Action vertical splitting
	Returns:
	New window
	"""

        newbox = np.copy(self.agent_window)
        boxH = newbox[3] - newbox[1]
        if boxH > MIN_BOX_SIDE:
            half = boxH/2.0
            newbox[3] -= half
        return newbox

 
    def aspectRatioDown(self):
	"""
	Action decreasing aspect ratio
	Returns:
	New window
	"""

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
	"""
	Termination action. This action returns the last window without any changes however for visualization purposes a black cross sign is put on the image to detemine search termination
	Returns:
	New window
	"""

        newbox = np.copy(self.agent_window)

        h = int((newbox[3] - newbox[1])/2)
        h_l = int(h/5)
        w = int((newbox[2] - newbox[0])/2)
        w_l = int(w/5)

        self.image_playground[newbox[1]+h-h_l:newbox[1]+h+h_l,newbox[0]:newbox[2]] = 0
        self.image_playground[newbox[1]:newbox[3],newbox[0]+w-w_l:newbox[0]+w+w_l] = 0

        return newbox




    def adjustAndClip(self):
	"""
	Cheching whether the new window is out of boundaries
	"""

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
	"""
	Computing IoU
	Args:
	boxA: First box
	boxB: Second box
	Returns:
	IoU for the given boxes
	"""

        # determine the (x, y)-coordinates of the intersection rectangle
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
	"""
	Going over the all bounding boxes to compute the reward for a given action.
	Args: 
	agent_window: Current agent window
	termination: Is this termination action?
	Returns:
	Reward and IoU coresponding to the agent window
	"""

	max_iou = -2
	reward = 0
        # Going over all the bounding boxes and return the reward corresponding to the closest object
        for target in self.targets:
            # Computing IoU between agent window and ground truth
            new_iou = self.intersectionOverUnion(agent_window, np.array(target))
	    if new_iou > max_iou:
		max_iou = new_iou
            	reward = self.ReturnReward(new_iou, termination)
        if termination: max_iou = 0
        return reward, max_iou

 
    def ReturnReward(self,new_iou, termination):
	"""
	Computing reward regarding new window
	Args:
	new_iou: new IoU corresponding to the recent action
	termination: Is this termination action?
	Returns: 
	Reward corresponding to the action
	"""
        reward = 0
	# If new IoU is bigger than the last IoU the agent will be recieved positive reward
        if new_iou - self.iou > 0:
            reward = 1
        else:
            reward = -1

        # If the action is the trigger then the new IoU will compare to the threshold 0.5 
        if termination:

            if (new_iou > 0.5):
                reward = 3
            else:
                reward = -3
            self.iou = 0
        return reward


    def drawActions(self):
	"""
	This function is to show the image with bounding boxes and the agent current window 
	"""

        fig,ax = plt.subplots(1)
        ax.imshow(self.image_playground)
        
        rect = patches.Rectangle((self.agent_window[0],self.agent_window[1]),self.agent_window[2]-self.agent_window[0],self.agent_window[3]-self.agent_window[1],linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)

        for target in self.targets:
            rect2 = patches.Rectangle((target[0],target[1]),target[2]-target[0],target[3]-target[1],linewidth=1,edgecolor='b',facecolor='none')
            ax.add_patch(rect2)

        plt.draw()
        plt.show()
        

   
    def my_draw(self):
        """
	This function is used by DQL_visualization_actions.py to make video from sequence of actions        
        """
        
        fig = Figure()
        canvas = FigureCanvas(fig)
        ax = fig.gca()
       
        ax.imshow(self.image_playground)
        
        rect = patches.Rectangle((self.agent_window[0],self.agent_window[1]),self.agent_window[2]-self.agent_window[0],self.agent_window[3]-self.agent_window[1],linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)

        for target in [self.targets[0]]:
            rect2 = patches.Rectangle((target[0],target[1]),target[2]-target[0],target[3]-target[1],linewidth=1,edgecolor='b',facecolor='none')
            ax.add_patch(rect2)
            
        canvas.draw()   
        
        width, height = fig.get_size_inches() * fig.get_dpi()

        return np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
        



