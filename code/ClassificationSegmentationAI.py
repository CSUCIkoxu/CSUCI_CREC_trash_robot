# -*- coding: utf-8 -*-
"""
CSUCI Conservation Robotics and Engineering Club
Trash Robot Trash Classification Segmentation AI

@author: Christopher Chang
"""

#Standard Libraries
import os
import cv2 as opencv
#import json
import fiftyone as fo #Library read json files in the COCO format

#AI Libraries
import numpy as np
import pandas as pd
#import xarray as xr #3d dataframes
import sklearn
import tensorflow as tf
from tensorflow import keras #Must use this to import these libraries or you get import errors
from tensorflow.keras import layers

DATASET_DIR = "..\\data"

#General Utility Functions ######################################################
def combineChannels(imgData):
    '''
    Combines an image that has been separated into its color channels back to the 
    original image format:
        [[width x height], [width x height], [width x height]] -> [width x height x channels]

    Parameters
    ----------
    imgData : [pandas.DataFrame[[float or int]]]
        The separated image that you want to reconstruct [3 x width x height]

    Returns
    -------
    reconstructedImg : numpy.array[[[float or int]]]
        The reconstructed image [width x height x 3]

    '''
    reconstructedImg = []
    
    tempImg = [i.values.tolist() for i in imgData]
    
    tempNp = np.array(tempImg)
    
    reconstructedImg = tempNp.transpose(1,2,0)
    
    return reconstructedImg

def separateChannels(imgData):
    '''
    Takes a 3D numpy array or list that contains an images data and separates
    the image into its color channels. That is,
        [width x height x channels] -> [[width x height], [width x height], [width x height]]
    In other words, separates the color channels of a 2D image as their own image

    Parameters
    ----------
    imgData : [[[float or int]]] or numpy.array[[[float or int]]]
        The image data to separate image data from [width x height x 3]

    Returns
    -------
    separatedImg : [pandas.DataFrame[[float or int]]]
         The separated color channels of the image [3 x width x height]

    '''
    separatedImg = []
    
    tempImg = imgData
    
    if not isinstance(imgData, np.ndarray):
        tempImg = np.array(imgData)
    
    transposed = tempImg.transpose(2,0,1)
    
    separatedImg = [pd.DataFrame(i) for i in transposed]
    
    return separatedImg

def combineChannelsInArr(imgs):
    '''
    Calls combineChannels() on all images in an array

    Parameters
    ----------
    imgs : [[pandas.DataFrame[[float or int]]], ...]
        A list with all the image data with separated channels (as specified in 
        separateChannels)

    Returns
    -------
    combinedImgs : [numpy.array[[[float or int]]]]
        The list of all images with combined channels (as specified in 
        combinedChannels)

    '''
    combinedImgs = []
    
    for i in imgs:
        combinedImgs.append(combineChannels(i))
        
    return combinedImgs

def separateChannelsInArr(imgs):
    '''
    Calls separateChannels() on all images in an array

    Parameters
    ----------
    imgs : [numpy.array[[[float or int]]]]
        The list of all images with combined channels (as specified in 
        combinedChannels)

    Returns
    -------
    separatedImgs : [[pandas.DataFrame[[float or int]]], ...]
        A list with all the image data with separated channels (as specified in 
        separateChannels)

    '''
    separatedImgs = []
    
    for i in imgs:
        separatedImgs.append(separateChannels(i))
        
    return separatedImgs

def bool2int(mask):
    '''
    Converts numpy arrays of booleans to numpy arrays of ints

    Parameters
    ----------
    mask : numpy.array[bool]
        Any dimension numpy array of booleans

    Returns
    -------
    numpy.array[int]
        The mask as a numpy array of ints where 0 is false and 1 is true

    '''
    return mask.astype(int)

def expandMask(imgDim, bounding_box, mask):
    '''
    Expands the masks stored in fiftyone to a full size mask with the dimensions
    of the original image.
    
    COCO JSON only stores the parts of the mask that is in the bounding box,
    in addition, Fiftyone stores the bounding box coordinates in relative 
    coordinates [<top-left-x>, <top-left-y>, <width>, <height>] between 0 and 1.
    This is fine until we need to resize images and their masks, thus, this
    function is made to expand the mask so it may be resized with ease or use
    in training.
    
    If there is a need to speed up processing time, this is an area for optimization
    as this is a "path of least resistance" solution.

    Parameters
    ----------
    imgDim : [int]
        An array that holds the original image's shape [Width, Height]
        (note: np.shape gives [Height, Width])
    bounding_box : [float]
        An array containing the relative bounding box lengths of structure:
            [<top-left-x>, <top-left-y>, <width>, <height>]
        Which is the standard format of the 'bounding_box' field in Fiftyone
    mask : np.Array[[bool]]
        The mask array that stores the segmentation data of the mask within
        the bounding box

    Returns
    -------
    eMask : np.Array[bool]
        The expanded, full-size mask of the target image instead of a small
        box within the image.
        

    '''
    #Create numpy array of size imgDim filled with False
    eMask = np.full((imgDim[0], imgDim[1]), False)
    
    #Calculate the position of the bounding box since Fiftyone stores the bounding box (and thus the mask position) as a float between 0 and 1 relative to the image size
    bboxCoor = (int((imgDim[0] * bounding_box[0]) + 0.5), int((imgDim[1] * bounding_box[1]) + 0.5))
    bboxDim = (int((imgDim[0] * bounding_box[2]) + 0.5), int((imgDim[1] * bounding_box[3]) + 0.5)) #I could just use the mask size
    
    #Iterate through each classification in the mask and sets it in the appropriate space in the expandded mask
    for i in range(bboxDim[0]):
        for j in range(bboxDim[1]):
            eMask[bboxCoor[0] + i][bboxCoor[1] + j] = mask[i][j]
    
    return eMask

#Data Fetching ##################################################################

def loadImgInfoFromFolder(path):
    '''
    Only loads the image file path into an array so that it may be retrieved
    When needed.
    Intended to be used inconjunction with the fiftyone dataset for train-test
    splitting

    Parameters
    ----------
    path : str
        The file path to the folder

    Returns
    -------
    images : [str]
        A list with all of the images in the folder

    '''
    images = []
    
    for filename in os.listdir(path):
        
        if filename.lower().endswith(('.jpg','.png','.jpeg')):  #Checks if the file is an image
            images.append(os.path.join(path,filename).replace('/', '\\'))
    
    return images

def loadImg(filePath):
    '''
    Loads the image data from the file path

    Parameters
    ----------
    filePath : str
        Path to the file

    Returns
    -------
    imageData : [pd.DataFrame[[float]]]
        The pixel data of the image, spearated into 3 dataframes, each for the RGB
        color values [r,g,b]

    '''
    imageData = None
    
    imageData = separateChannels(opencv.imread(filePath))
    
    return imageData

def loadAllImgs(filepaths):
    '''
    Loads all of the images in the given list

    Parameters
    ----------
    filepaths : [str]
        The list of filepaths to the images you want to load

    Returns
    -------
    data : pandas.DataFrame[[pd.DataFrame[[float]]]]
        A dataframe containing the data of all images of the specified paths.
        The dataframe is 1d, each element is a list of size 3 for rgb, [r,g,b], and each element
        of the list is a dataframe of the image values

    '''
    data = []
    
    for f in filepaths:
        data.append(loadImg(f))
    
    return data

def getYLabels(filePaths, foDataset):
    labels = []
    
    for f in filePaths:
        labels.append(foDataset[os.path.abspath(f)])
        
    return labels

def cleanUpDatasets():
    '''
    Call this function before rerunning code to remove any reference to any lingering
    datasets still in memory without restarting the kernel/console

    '''
    datasetList = fo.list_datasets()
    for ds in datasetList:
        fo.load_dataset(ds).delete()

def getData():
    dataX = []                          #Raw image Data stored in array along with file name
    dataY = fo.Dataset("trash-dataset") #Ground Truth labels stored in Fiftyone data set
    
    for filename in os.listdir(DATASET_DIR):
        pathName = os.path.join(DATASET_DIR,filename)
        
        if (os.path.isdir(pathName)):
            dataX += loadImgInfoFromFolder(pathName)
        elif (os.path.isfile(pathName)):
            if (filename == "annotations.json"): #or filename == "annotations_unofficial.json"):    #We did not download the unofficial data, if we do, uncomment this              
                #Use fiftyone API to decode json data
                dataset = fo.Dataset.from_dir(
                    data_path=DATASET_DIR,
                    labels_path=pathName,
                    dataset_type=fo.types.COCODetectionDataset,
                    name=filename)
                dataY.merge_samples(dataset)
        else:
            pass    #Skip anything that does not have important data
    
    
    return dataX, dataY

#Data Preprocessing #############################################################

def resizeImg(img, masks, targetH, targetW):
    '''
    Resizes a given image and associated mask to the specified Height and Width

    Parameters
    ----------
    img : [pandas.Dataframe[[float]]] or [[pd.Dataframe[[float]], ...], ...]
        The image data to resize
    masks : pandas.Dataframe[[[float]]]
        An array of images that contain the mask labels
    targetH : int
        The height you want to resize to
    targetW : int
        The width you want to resize to

    Returns
    -------
    reimg : pandas.Dataframe[[[float]]]
        The resized image of size [targetH,targetW]
    remasks : [pandas.Dataframe[[float]]]
        An array of masks resized to the size [targetH,targetW]

    '''
    reimg = None
    remasks = []
    
    combinedImgs = combineChannelsInArr(img)
    # print(combinedImgs)
    
    tempImgs = []
    for i in combinedImgs:
        newI = tf.image.resize(i, (targetH, targetW), method='nearest')
        tempImgs.append(newI.numpy())
    
    reimg = separateChannelsInArr(tempImgs)
    
    cntr = 0
    
    for maskArr in masks:
        print("Mask " + str(cntr))
        subRemasks = []
        for mask in maskArr:
            intMask = bool2int(mask.mask)
            newMaskDim = (int((targetW * mask.bounding_box[2]) + 0.5), int((targetH * mask.bounding_box[3]) + 0.5))
            subRemasks.append(opencv.resize(intMask, (newMaskDim[0], newMaskDim[1]), interpolation=opencv.INTER_NEAREST))
        remasks.append(subRemasks)
        cntr += 1
    
    return reimg, remasks

def normalize(img):
    '''
    Normalize the provided image so its values are between 0 and 1

    Parameters
    ----------
    img : pandas.Dataframe[[[float]]]
        The image to normalize

    Returns
    -------
    imgNorm : pandas.Dataframe[[[float]]]
        The normalized image with values between 0-1

    '''
    imgNorm = [i.copy() / 255.0 for i in img]
    
    #imgNorm = imgNorm / 255.0
    
    return imgNorm

def normalize_SimpleMask(img, mask):
    '''
    Normalizes the data, sets the mask data to either 0 for non-trash and 1 for
    trash instead of values 1 or greater for different categories of trash

    Parameters
    ----------
    img : pandas.Dataframe[[[float]]]
        The image data to normalize
    mask : pandas.DataFrame[[int]]
        The label mask for the image.  Expects only one since it is assumed that
        categorical classifications are all stored in one mask as values 1 or greater

    Returns
    -------
    imgNorm : pandas.Dataframe[[[float]]]
        The image data as a value between 0 and 1
    maskNorm : pandas.DataFrame[[int]]
        The simplified mask data, removing category labels and replacing it 
        with a trash boolean (1 = trash, 0 = no trash)

    '''
    imgNorm = img.copy()
    maskNorm = mask.copy()
    
    imgNorm = imgNorm / 255.0   #Divide color values to get value between 0-1
    maskNorm.clip(upper=1, inplace=True)    #Sets all trash labels to 1 instead of categorical types of trash
                
    return imgNorm, maskNorm

def normalize_SingleMask(img, masks):
    imgNorm = img.copy()
    maskNorm = pd.DataFrame()
    
    imgNorm = imgNorm / 255.0   #Divide color values to get value between 0-1
    
    for i in range(masks[0].shape[0]):
        for j in range(masks[0].shape[1]):
            maskNorm[i][j] = 0 
            
            for mask in masks:
                if (mask[i][j] > 0):
                    maskNorm[i][j] = 1
                    break
                
    return imgNorm, maskNorm

def preprocessData(imgs, masks, resizeDim=(3264,2448)): #The dimensions (3264,2448) is the most common image size of the dataset
    '''
    Applies Preprocessing to the images and associated masks
    Preprocessing includes:
        Resizing
        Normalizing

    Parameters
    ----------
    imgs : [pandas.DataFrame[[[float]]]]
        A list of pandas.DataFrame, each holding the image data
    masks : [pandas.DataFrame[[int]]
        A list of pandas.DataFrames that hold the mask data to the associated
        image
    resizeDim : (int, int), optional
        The size of the images you want to resize to. 
        The default is (3264,2448). (Height, Width)

    Returns
    -------
    imgNew : [pandas.DataFrame[[[float]]]]
        The set of preprocessed images for the AI
    masksNew : [pandas.DataFrame[[int]]
        The resized masks of the associated images for the AI

    '''
    imgNew = []
    masksNew = []
    
    img_R, masks_R = resizeImg(imgs, masks, resizeDim[0], resizeDim[1])
    
    for i in range(len(imgs)):
        # img_R, masks_R = resizeImg(imgs[i], masks[i], resizeDim[0], resizeDim[1])
        img_N = normalize(img_R[i])
        
        imgNew.append(img_N)        #Append the resized and normalized image
        
    masksNew = masks_R
        
    
    return imgNew, masksNew

#Machine Learning Utility Functions #############################################

def trainTestSplit(imgs, seed=None):
    '''
    Splits the given data into training and testing sets (70-30)

    Parameters
    ----------
    imgs : [str]
        A list of filepaths to an associated image you want to split
    seed : int, optional
        The seed you want to use for the random splitting. The default is None.

    Returns
    -------
    imgsTrain : [str]
        The list of training images to train on
    imgsTest : [str]
        The list of testing images to test on

    '''
    imgsTrain = []
    imgsTest = []
    # masksTrain = []
    # masksTest = []
    
    imgsTrain, imgsTest = sklearn.model_selection.train_test_split(imgs, test_size=0.3, random_state=seed)
    
    return imgsTrain, imgsTest#, masksTrain, masksTest

def stats(yPred, yTrue):
    '''
    Calculates the scores to determine how accurate the models predictions are

    Parameters
    ----------
    yPred : pandas.DataFrame[[[int]]]
        The predicted labels of the dataset
    yTrue : pandas.DataFrame[[[int]]]
        The ground truth labels of the dataset

    Returns
    -------
    scores : [float]
        A list with the calculated performance metrics from the predictions
            []
    statStr : str
        A string summerizing the data from scores

    '''
    scores = []
    statStr = ""
    
    return scores, statStr

#Create Machine Learning Model ##################################################

def createModel(inputSize, outputSize):
    model = None
    
    model = keras.Sequential(layers.Conv2D(3, 20, input_shape=inputSize, activation='relu'))
    
    return model

#Main ###########################################################################

cleanUpDatasets()
dataX, dataY = getData()

#Preprocess - Test
dataXTrain, dataXTest = trainTestSplit(dataX, 123)

dataXRaw = loadAllImgs(dataXTest[:int(len(dataXTest)/10)])
dataYRaw = getYLabels(dataXTest[:int(len(dataXTest)/10)], dataY)

dataYMasks = [m.segmentations.detections for m in dataYRaw]
# dataYMasks = [(m.segmentations.detections[i].mask for i in range(len(m.segmentations.detections))) for m in dataYRaw]

dataXRaw_norm, dataYRaw_norm = preprocessData(dataXRaw, dataYMasks)


#Only run this to check data in fiftyone Viewer, cannot be run in a Notebook editor (like Spyder)
# if __name__ == "__main__":
#     # Ensures that the App processes are safely launched on Windows
#     session = fo.launch_app(dataY,desktop=True)
#     session.wait()

