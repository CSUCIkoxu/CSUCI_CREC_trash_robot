# -*- coding: utf-8 -*-
"""
CSUCI Conservation Robotics and Engineering Club
Trash Robot Trash Classification Segmentation AI

@author: Christopher Chang
"""

#Standard Libraries
import os
import cv2
#import json
import fiftyone as fo #Library read json files in the COCO format

#AI Libraries
import pandas as pd
import sklearn
import tensorflow as tf
import tensorflow.keras as keras
import keras.layers as layers

DATASET_DIR = "../data"

#Data Fetching ##################################################################

def loadImgFromFolder(path):
    images = [[],[]]
    
    for filename in os.listdir(path):
        img = cv2.imread(os.path.join(path,filename))
        
        if img is not None:
            # images[0].append(img)
            images[1].append(os.path.join(path,filename))
    
    return images

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
            dataX.append(loadImgFromFolder(pathName))
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

def resizeImg(img, targetW, targetH):
    pass

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

