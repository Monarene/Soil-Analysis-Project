# -*- coding: utf-8 -*-
"""
Created on Sat May 25 11:38:52 2019

@author: H P ENVY
"""
#importing the neccessary directories
from config import soil_analysis_config as config
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from utilities.preprocessing import AspectAwarePreprocessor
from utilities.io import HDF5DatasetWriter
from imutils import paths
import os
import json
import numpy as np
import cv2

#the data work and flow
#Data extraction and preprocessing
trainPaths = list(paths.list_images(config.IMAGES_PATH))
trainLabels  = [p.split(os.path.sep)[2].split(".")[0] for p in trainPaths]
le  = LabelEncoder()
trainLabels = le.fit_transform(trainLabels)
split  = train_test_split(trainPaths, trainLabels, test_size  = config.NUM_TEST_IMAGES, 
                          stratify = trainLabels, random_state = 42)
(trainPaths, testPaths, trainLabels, testLabels) = split
split  = train_test_split(trainPaths, trainLabels, test_size = config.NUM_VAL_IMAGES,
                          stratify = trainLabels, random_state = 42)
(trainPaths, valPaths, trainLabels, valLabels) = split
(R, G, B) = ([], [], [])
aap  = AspectAwarePreprocessor(256, 256)

dataset = [
        ("train", trainPaths, trainLabels, config.TRAIN_HDF5),
        ("test", testPaths, testLabels, config.TEST_HDF5),
        ("val", valPaths, valLabels, config.VAL_HDF5)]

for (dType, paths, labels, outputPath) in dataset:
    writer = HDF5DatasetWriter((len(paths), 256, 256, 3), outputPath, bufSize  = 20)
    for (path, label) in zip(paths, labels):
        image  = cv2.imread(path)
        image  = aap.preprocess(image)
        
        if dType == "train":
            (b, g, r) = cv2.mean(image)[:3]
            R.append(r)
            B.append(b)
            G.append(g)
        
        writer.add([image], [label])
writer.close()

D = {"R":np.mean(R), "G":np.mean(G), "B":np.mean(B)}
f = open(config.DATASET_MEAN, "w")
f.write(json.dumps(D))
f.close()





