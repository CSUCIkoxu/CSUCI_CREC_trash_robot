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
import tensorflow.keras as keras
import keras.layers as layers

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
    path : string
        The file path to the folder

    Returns
    -------
    images : [string]
        A list with all of the images in the folder

    '''
    images = []
    
    for filename in os.listdir(path):
        
        if filename.lower().endswith(('.jpg','.png','.jpeg')):  #Checks if the file is an image
            images.append(os.path.join(path,filename))
    
    return images

def loadImg(filePath):
    '''
    Loads the image data from the file path

    Parameters
    ----------
    filePath : string
        Path to the file

    Returns
    -------
    imageData : TYPE
        DESCRIPTION.

    '''
    imageData = []
    
    imageData = opencv.imread(filePath)
    
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
            dataX.append(loadImgInfoFromFolder(pathName))
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

def resizeImg(img, masks, targetW, targetH):
    '''
    Resizes a given image and associated mask to the specified Width and Height

    Parameters
    ----------
    img : pandas.Dataframe[[[float]]]
        The image data to resize
    masks : pandas.Dataframe[[[float]]]
        An array of images that contain the mask labels
    targetW : int
        The width you want to resize to
    targetH : int
        The height you want to resize to

    Returns
    -------
    reimg : pandas.Dataframe[[[float]]]
        The resized image of size [targetW,targetH]
    remasks : pandas.Dataframe[[[float]]]
        An array of masks resized to the size [targetW,targetH]

    '''
    reimg = []
    remasks = []
    
    reimg = tf.image.resize(img, (targetW, targetH), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    
    for mask in masks:
        remasks.append(tf.image.resize(mask, (targetW, targetH), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR))
    
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

def normalize_SingelMask(img, masks):
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

def preprocessData():
    return

#Create Machine Learning Model ##################################################

def createModel(inputSize, outputSize):
    model = None
    
    model = keras.Sequential(layers.Conv2D(3, 20, input_shape=inputSize, activation='relu'))
    
    return model

#Main ###########################################################################

cleanUpDatasets()
dataX, dataY = getData()

#Only run this to check data in fiftyone Viewer
# if __name__ == "__main__":
#     # Ensures that the App processes are safely launched on Windows
#     session = fo.launch_app(dataY,desktop=True)
#     session.wait()

