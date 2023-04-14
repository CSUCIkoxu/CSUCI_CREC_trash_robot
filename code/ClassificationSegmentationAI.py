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
import sklearn
import tensorflow as tf
from tensorflow import keras #Must use this to import these libraries or you get import errors
from tensorflow.keras import layers

DATASET_DIR = "../data"

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
            images.append(os.path.join(path,filename).replace('\\','/'))
    
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
    imageData : pd.DataFrame[[[float]]]
        The pixel data of the image

    '''
    imageData = None
    
    imageData = pd.DataFrame(opencv.imread(filePath))
    
    return imageData

def cleanUpDatasets():
    '''
    Call this function before rerunning code to remove any reference to any lingering
    datasets still in memory without restarting the kernel/console

    Returns
    -------
    None.

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
                    data_path=DATASET_DIR+'/../data',
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
    img : pandas.Dataframe[[[float]]]
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
    
    reimg = tf.image.resize_with_pad(img, (targetH, targetW), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    
    for mask in masks:
        remasks.append(tf.image.resize_with_pad(mask, (targetH, targetW), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR))
    
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
    imgNorm = img.copy()
    
    imgNorm = imgNorm / 255.0
    
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
        The default is (3264,2448).

    Returns
    -------
    imgNew : [pandas.DataFrame[[[float]]]]
        The set of preprocessed images for the AI
    masksNew : [pandas.DataFrame[[int]]
        The resized masks of the associated images for the AI

    '''
    imgNew = []
    masksNew = []
    
    for i, m in imgs, masks:
        img_R, masks_R = resizeImg(i, m, resizeDim[0], resizeDim[1])
        img_N = normalize(img_R)
        
        imgNew.append(img_N)        #Append the resized and normalized image
        masksNew.append(masks_R)    #Append the resized masks
    
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

#Only run this to check data in fiftyone Viewer, cannot be run in a Notebook editor (like Spyder)
# if __name__ == "__main__":
#     # Ensures that the App processes are safely launched on Windows
#     session = fo.launch_app(dataY,desktop=True)
#     session.wait()

