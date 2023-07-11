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
ANNOTATION_CATEGORIES = ["Cigarette","Unlabeled litter","Plastic film","Clear plastic bottle","Other plastic",
                         "Other plastic wrapper","Drink can","Plastic bottle cap","Plastic straw","Broken glass",
                         "Styrofoam piece","Disposable plastic cup","Glass bottle",
                         "Pop tab","Other carton","Normal paper","Metal bottle cap",
                         "Plastic lid","Paper cup","Corrugated carton","Aluminium foil",
                         "Single-use carrier bag","Other plastic bottle","Drink carton",
                         "Tissues","Crisp packet","Disposable food container","Plastic utensils"
                         ,"Food Can","Garbage bag","Meal carton","Rope & strings",
                         "Paper bag","Scrap metal","Foam food container","Foam cup",
                         "Magazine paper","Wrapping paper","Egg carton","Aerosol",
                         "Metal lid","Spread tub","Food waste","Shoe","Squeezable tube",
                         "Aluminium blister pack","Glass cup","Other plastic container",
                         "Glass jar","Six pack rings","Toilet tube","Paper straw",
                         "Plastic glooves","Tupperware","Polypropylene bag","Pizza box",    #[sic] gloves is misspelt
                         "Other plastic cup","Battery","Carded blister pack","Plastified paper bag"]
ANNOTATION_SUPER_CATEGORIES = ["Plastic bag & wrapper","Cigarette","Unlabeled litter",
                               "Bottle","Bottle cap","Can","Other plastic","Carton",
                               "Cup","Straw","Paper","Broken glass","Styrofoam piece",
                               "Pop tab","Lid","Plastic container","Aluminium foil",
                               "Plastic utensils","Rope & strings","Paper bag","Scrap metal",
                               "Food Waste","Shoe","Squeezable tube","Blister pack",
                               "Glass jar","Plastic glooves","Battery"]

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
    #(Height, Width)
    bboxCoor = (int((imgDim[0] * bounding_box[1]) + 0.5), int((imgDim[1] * bounding_box[0]) + 0.5))
    bboxDim = (int((imgDim[0] * bounding_box[3]) + 0.5), int((imgDim[1] * bounding_box[2]) + 0.5)) #I could just use the mask size
    
    #Iterate through each classification in the mask and sets it in the appropriate space in the expandded mask
    # print("BBox: {}, Actual: {}".format(bboxDim, mask.shape))
    for i in range(bboxDim[0]):
        for j in range(bboxDim[1]):
            eMask[bboxCoor[0] + i][bboxCoor[1] + j] = mask[i][j]
    
    return eMask

def combineMasks(imgDim, masks, detectionList):
    '''
    Converts the given list of partial masks (i.e. masks confined to a bounding box)
    into a list of full categorical masks (i.e. mask of the entire image for 
                                           each individual category)

    Parameters
    ----------
    imgDim :  [int]
        An array that holds the original image's shape [Width, Height]
        (note: np.shape gives [Height, Width])
    masks : np.Array[[bool]]
        The mask array that stores the segmentation data of the mask within
        the bounding box
    detectionList : [[fo.Detection]]
        A list consisting of the various detections within the associated image
        and mask

    Returns
    -------
    masks_expanded_combined : [numpy.array[[[int]]]]
        The list of full categorical masks of dimension [imgDimH x imgDimW x 60]
        Where 60 is the number of annotation categories in the dataset
        Individual indices are ints of either 0 or 1 

    '''
    #Converts the list of partial segmentations to full image segmentations
    masks_expanded = []
    for i in range(len(masks)):
        expandedMasks = []
        for j in range(len(masks[i])):
            #Creates a tuple (expanded mask, annotation label)
            expandedMasks.append((expandMask(imgDim, detectionList[i][j].bounding_box, masks[i][j]), detectionList[i][j].label))
        masks_expanded.append(expandedMasks)

    #Converts the list of full image segmentations into a full list of categorical annotations, i.e. the masks for all categories, even if they are not present
    masks_expanded_combined = []
    for i in masks_expanded:
        combinedMasks = np.full((len(ANNOTATION_CATEGORIES), imgDim[0], imgDim[1]), False)
        for j in i:
            #Perform a bitwise OR on the proper category of the combined masks array
            combinedMasks[ANNOTATION_CATEGORIES.index(j[1])] = np.bitwise_or(combinedMasks[ANNOTATION_CATEGORIES.index(j[1])], j[0])
        #Transpose mask_expanded_combined to change the format from (60 x imgDimH x imgDimW) to
        #(imgDimH x imgDimW x 60)
        combinedMasks = combinedMasks.transpose(1,2,0)
        
        masks_expanded_combined.append(combinedMasks)
        
    return masks_expanded_combined.astype(float)

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
    '''
    Gets the label information relevant to the provided list of file paths

    Parameters
    ----------
    filePaths : [str]
        A list of file paths you want to retrieve data from
    foDataset : fo.Dataset
        The dataset you want to retrieve data from

    Returns
    -------
    labels : [fo.Sample]
        A list of segmentation samples for each image

    '''
    labels = []
    
    for f in filePaths:
        labels.append(foDataset[os.path.abspath(f)])
        
    return labels

def getDetections(dataYLabels):
    '''
    Gets the detection information from a given list of labels (from getYLabels)

    Parameters
    ----------
    dataYLabels : [fo.Samples]
        The list of segmentation samples you want to get the detection data for

    Returns
    -------
    [[fo.Detection]]
        The list of detection information for each identified segmentation

    '''
    return [m.segmentations.detections for m in dataYLabels]

def getMasks(dataYDetections):
    '''
    Gets the mask data from a given list of detections (from getDetections)

    Parameters
    ----------
    dataYDetections : [[fo.Detection]]
        A 2-d list of all the detections of the given selection of data

    Returns
    -------
    masksList : [[numpy.array[[bool]]]]
        A 2-d list of 2-d masks for each detection

    '''
    masksList = []
    
    for d in dataYDetections:
        masks = []
        for s in d:
            masks.append(s.mask)
    
    return masksList

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
    masks : [fiftyone.detection[[int]]
        The list of detections samples provided by the fiftyone datastructure.
        The list consists of samples from the database, that is:
            [database.segmentations.detections]
        *detections is a list consisting of the detection datastructure
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

def createModel1(inputSize, outputSize):
    '''
    A simple U-Net segmentation model provided by https://www.tensorflow.org/tutorials/images/segmentation
    Intended as a quick and dirty test model rather than an optimized model for trash detection.

    Parameters
    ----------
    inputSize : [int]
        The dimensions of the input of the parameter of the model (usually the image size)
        Syntax:
            [Height, Width, Channels]
    outputSize : [int]
        The dimensions of the output of the model.
        In this case, we would want the output to be the dimensions of the image
        times the number of channels/classification categories in the data
        Syntax:
            [Height, Width, Channels/Categories]

    Returns
    -------
    model : tensorflow.keras.model
        The constructed machine learning model

    '''
    model = None
    
    # Simple U-net Model provided by https://www.tensorflow.org/tutorials/images/segmentation
    from tensorflow_examples.models.pix2pix import pix2pix
    
    base_model = tf.keras.applications.MobileNetV2(input_shape=inputSize, include_top=False)
    # Use the activations of these layers
    layer_names = [
        'block_1_expand_relu',   # 64x64
        'block_3_expand_relu',   # 32x32
        'block_6_expand_relu',   # 16x16
        'block_13_expand_relu',  # 8x8
        'block_16_project',      # 4x4
    ]
    base_model_outputs = [base_model.get_layer(name).output for name in layer_names]
    
    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)
    
    down_stack.trainable = False
    
    inputs = tf.keras.layers.Input(shape=inputSize)

    # Downsampling through the model
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])
    
    # Upsampling
    up_stack = [
        pix2pix.upsample(512, 3),  # 4x4 -> 8x8
        pix2pix.upsample(256, 3),  # 8x8 -> 16x16
        pix2pix.upsample(128, 3),  # 16x16 -> 32x32
        pix2pix.upsample(64, 3),   # 32x32 -> 64x64
    ]

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
      x = up(x)
      concat = tf.keras.layers.Concatenate()
      x = concat([x, skip])
      
    output_channels = 60 #There are 60 categories in the dataset - more info here: http://tacodataset.org/stats
  
    # This is the last layer of the model
    last = tf.keras.layers.Conv2DTranspose(
        filters=output_channels, kernel_size=3, strides=2,
        padding='same')  #64x64 -> 128x128
  
    x = last(x)

    model = keras.Model(inputs=inputs, outputs=x)
    
    return model

def createModel(inputSize, outputSize):
    model = None
    
    model = createModel1(inputSize, outputSize)
    
    return model

#Train Machine Learning Model ###################################################
def batch_loadImg(xxTrain, yData, index, batchSize, imgDim=(3264,2448)):
    '''
    Individual loading function for loading data in batches of batchSize each call.
    The index specifies which section of the data to pull the batch from.
    Preprocesses the data before returning the data for use

    Parameters
    ----------
    xxTrain : [str]
        A list of filepaths to the images for training
    yData : fo.dataset
        The full dataset of label information provided be reading the COCO json
        file using the Fiftyone library
    index : int
        The current slice to gather data from xxTrain
    batchSize : int
        The volume of data to pull into memory
    imgDim : [int], optional
        A list or tuple containing the image resize dimensions. 
        The dimensions should be of the format: [H, W]
        The default is (3264,2448).

    Returns
    -------
    xBatchData : [numpy.array[[[float]]]]
        A list of image data loaded for training
    yBatchData : [numpy.array[[[float]]]]
        A list of categorized masks of the entire associated image

    '''
    xBatchData = None
    yBatchData = None
    
    x_selected = []
    x_dat = []
    y_dat = []
    
    #Fetch the image data for this batch
    for i in range(index*batchSize, (index*batchSize) + batchSize):
        x_selected.append(xxTrain[i])
        x_dat.append(loadImg(xxTrain[i]))
    #Fetch the masks from the database
    y_dat = getYLabels(x_selected, yData)
    y_dat_masks = [m.segmentations.detections for m in y_dat]
    
    #Preprocess the data before sending to fit
    x_dat_norm, y_dat_masks_norm = preprocessData(x_dat, y_dat_masks, resizeDim=imgDim)
    
    #Set xBatchData and combines the masks into a workable format (imgDim x 60)
    xBatchData = combineChannelsInArr(x_dat_norm)
    yBatchData = combineMasks(imgDim, y_dat_masks_norm, y_dat)
    
    return xBatchData, yBatchData

def batchGenerator(xData, yData, batchSize, steps, imgDim=(3264,2448)):
    '''
    A generator object that fetches specified data when called upon

    Parameters
    ----------
    xData : [str]
        A list of filepath strings that lead to an associated image in storage
    yData : fo.dataset
        A fiftyone dataset object that contains all of the label information
    batchSize : int
        The number of samples to pull into memory per batch
    steps : int
        The number of batches there will be when reading the data
    imgDim : [int], optional
        A list or tuple containing the image resize dimensions. 
        The dimensions should be of the format: [H, W]
        The default is (3264,2448).

    Yields
    ------
    ([numpy.array[[[float]]]], [numpy.array[[[float]]]])
        A tuple containing the image data (x) and the mask data (y) in the format
        (x,y)
    '''
    indx = 0
    while True:
        yield batch_loadImg(xData, yData, indx, batchSize, imgDim)
        if indx < steps:
            indx += 1
        else:
            indx = 0
    
def trainSequence(model, xxTrain, xValid, yData, hyperParams=['adam',tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),['accuracy']], batchSize=20, epochNum=30, callbacks=None):
    '''
    Performs a training sequence of the model given a set of hyperparameters.
    This function is intended for performing hold-out cross-validation for optimization
    hyperparameters.

    Parameters
    ----------
    model : keras.Model
        The predesigned model to train on
    xxTrain : [str]
        A list of filepath strings to direct to an image in storage intended to
        be used for validation training
    xValid : [str]
        A list of filepath strings to direct to an image in storage intended to
        be used for validation testing
        If xValid is None, then no validation will be performed
    yData : fo.dataset
        A Fiftyone dataset that contains all label information for every image
        in the database
    hyperParams : [str or keras object], optional
        A list of hyperparameters to use in the compilation and training of the model. 
        The default is ['adam',tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),['accuracy']].
    batchSize : int, optional
        The number of samples to pull into memory for each training batch. 
        The default is 20.
    epochNum : int, optional
        The maximum number of epochs to perform when training the model. If an 
        earlystopping callback is given, it will stop when the callback determines 
        is best. 
        The default is 30.
    callbacks : [keras.callback], optional
        A list of callbacks to use in the training model. The default is None.

    Returns
    -------
    trainedModel : keras.Model
        A trained clone of the model provided
    modelHist : keras.History
        A history object that contains all the training and scoring information 
        during the trainin process

    '''
    trainedModel = None
    modelHist = None
    
    print("Model training on : [{}, {}, {}]".format(hyperParams[0], hyperParams[1], hyperParams[2]))
    
    #Create the current itereaetion of the training model
    trainedModel = keras.models.clone_model(model)
    trainedModel.compile(optimizer=hyperParams[0], loss=hyperParams[1], metrics=hyperParams[2])
    
    #Define the number of steps for training and validation
    stepsPerEpoch_training = len(xxTrain) / batchSize
    stepsPerEpoch_valid = 0
    #Create the batch generators for the data of the training data nad validation data
    trainingBatchGen = batchGenerator(xxTrain, yData, batchSize, stepsPerEpoch_training)
    validationBatchGen = None
    
    if (not xValid is None):
        stepsPerEpoch_valid = len(xValid) / batchSize
        validationBatchGen = batchGenerator(xValid, yData, batchSize, stepsPerEpoch_valid)
    
    #Train the model by iterativeley pulling data using a generator; NOTE: multiprocessing is used to get data from the generators and put it in a queue for the gpu, use if data fetching is slow
    modelHist = trainedModel.fit(x=trainingBatchGen, validation_data=validationBatchGen, epochs=epochNum, steps_per_epoch=stepsPerEpoch_training, validation_steps=stepsPerEpoch_valid, callbacks=callbacks, max_queue_size=5, workers=3, use_multiprocessing=True)
    
    return trainedModel, modelHist

def iterativeTrain(model, xTrain, xTest, yData, yMasks):
    trainedModel = None
    hyperParams = []
    trainingScore = 0
    
    #Validation Split
    xxTrain, xVal = trainTestSplit(xTrain,seed=123)
    
    #Early-stopping
    earlyStopCallback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    callbacks = [earlyStopCallback]
    
    scores = {}
    scoreEpoch = {}
    optimizers = ['adadelta', 'adagrad', 'adam', 'adamax', 'ftrl', 'nadam', 'rmsprop']#, 'SGD']     #SGD just returns Nan's for some reason
    lossFuncs = [tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),'poisson','kl_divergence']
    metrics = [['accuracy']]
    for o in optimizers:
        for l in lossFuncs:
            for m in metrics:
                #Train the model
                _, modelHist = trainSequence(model, xxTrain, xVal, yData, hyperParams=[o, l, m], batchSize=20, epochNum=30, callbacks=callbacks)
                
                #Since we have early stopping, we want to get the epoch at which we had the best score
                minScore = min(modelHist.history["val_" + m[0]])
                for i in range(len(modelHist.history["val_" + m[0]])):
                    if (modelHist.history["val_" + m[0]][i] == minScore):
                            scoreEpoch[(o, l, m[0])] = i
                            break
                            
                scores[(o, l, m[0])] = minScore
        
    #Fetch the best score for the model with which parameters
    trainingScore = min(scores.values())
    for k in scores.keys():
        if (trainingScore == scores[k]):
            hyperParams = k
            break
        
    print("Finished cross validating!")
    
    #Train the final model on the best performing hyperparameters
    print("Reconstructing best model with hyperparameters: [{}, {}, {}]".format(hyperParams[0], hyperParams[1], hyperParams[2]))
    bestModel = model
    bestModel.compile(optimizer=hyperParams[0], 
                         loss=hyperParams[1], 
                         metrics=[hyperParams[2]])
    trainedModel, _ = trainSequence(bestModel, xTrain, None, yData, hyperParams=hyperParams, batchSize=20, epochNum=scoreEpoch[hyperParams], callbacks=callbacks)
    
    return trainedModel, hyperParams, trainingScore

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

# dataYRaw_norm_expanded = []
# for i in range(len(dataYRaw_norm)):
#     expandedMasks = []
#     for j in range(len(dataYRaw_norm[i])):
#         expandedMasks.append(expandMask((3264,2448), dataYMasks[i][j].bounding_box, dataYRaw_norm[i][j]))
#     dataYRaw_norm_expanded.append(expandedMasks)

# dataYRaw_norm_expanded_combined = []
# for i in dataYRaw_norm_expanded:
#     combinedMasks = np.full((3264,2448), False)
#     for j in i:
#         combinedMasks = np.bitwise_or(combinedMasks, j)
#     dataYRaw_norm_expanded_combined.append(combinedMasks)

masksCombined = combineMasks((3264,2448), dataYRaw_norm, dataYMasks)

#Create the machine learning model
model = createModel((3264,2448,3), (3264,2448))


#Only run this to check data in fiftyone Viewer, cannot be run in a Notebook editor (like Spyder)
# if __name__ == "__main__":
#     # Ensures that the App processes are safely launched on Windows
#     session = fo.launch_app(dataY,desktop=True)
#     session.wait()

