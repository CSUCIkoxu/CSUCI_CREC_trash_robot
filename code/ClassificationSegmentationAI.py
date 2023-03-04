# -*- coding: utf-8 -*-
"""
CSUCI Conservation Robotics and Engineering Club
Trash Robot Trash Classification Segmentation AI

@author: Christopher Chang
"""

#Standard Libraries
import os
import cv2
import json

#AI Libraries
import pandas as pd
import sklearn
import tensorflow as tf
import tensorflow.keras as keras
import keras.layers as layers

DATASET_DIR = "../dataset"

def loadImgFromFolder(path):
    images = []
    
    for filename in os.listdir(path):
        img = cv2.imread(os.path.join(path,filename))
        
        if img is not None:
            images.append(img)
    
    return images

def getData():
    data = []
    dataX = []
    dataY = []
    
    for filename in os.listdir(DATASET_DIR):
        pathName = os.path.join(DATASET_DIR,filename)
        
        if (os.path.isdir(pathName)):
            dataX.append(loadImgFromFolder(pathName))
        elif (os.path.isfile(pathName)):
            if (filename == "annotations.json" or filename == "annotations_unofficial.json"):
                f = open(pathName)
                jsonData = json.load(f)
                
                
                
    
    data = [dataX, dataY]
    
    return data