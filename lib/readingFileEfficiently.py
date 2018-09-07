import gc
import VOC2012DataProvider


def giveData(which_set, batch_size):
    """
    Reads .npz files
    Args:
       which_set: This indicates whether training or testing set needs to be loaded
       batch_size: Batch size
    Returns:
       A batch of images
    """
    if which_set == 'train':
	for i in range(1,4):
		print "input file {} is loading...".format(i)
		yield VOC2012DataProvider.PascalDataProvider(i, which_set = which_set, batch_size = batch_size)
    elif which_set == 'test':
		print "input file is loading..."
		yield VOC2012DataProvider.PascalDataProvider("", which_set = which_set, batch_size = batch_size)
	


def extractData(objClassName, which_set, batch_size):
    """
    Reads dataset 
    Args:
      objClassName: Object category that is needed
      which_set: This indicates whether training or testing set needs to be loaded
      batch_size: Batch size
    Returns:
      Image and its ground truth from the given category
    """
 
    # Loading .npz files
    for fileInp in giveData(which_set, batch_size):

        # Getting images batch in the current file
        for img_batch, targ_batch in fileInp:

            # Iterating over the current batch
            for batch_index, _ in enumerate(img_batch):
                xmin = []
                xmax = []
                ymin = []
                ymax = []
		objectName = ''
                found = False

                # Iterates over objects in the current image
                for objInd, objName in enumerate(targ_batch[batch_index]['objName']):

		    # Checks whether the desired object exist in the loaded image
                    if (objName in objClassName) or ('*' in objClassName):
                        xmin.append(targ_batch[batch_index]['xmin'][objInd])
                        ymin.append(targ_batch[batch_index]['ymin'][objInd]) 
                        xmax.append(targ_batch[batch_index]['xmax'][objInd])
                        ymax.append(targ_batch[batch_index]['ymax'][objInd])
			objectName = objName
                        found = True

                        
                    del objInd
                    del objName
                
                groundtruth = {'xmin':xmin, 'ymin':ymin, 'xmax':xmax, 'ymax':ymax, 'objName':objectName}
                if found:
                    yield img_batch[batch_index], groundtruth
                else:
                    pass
            
            del img_batch
            del targ_batch

        del fileInp
        gc.collect()
