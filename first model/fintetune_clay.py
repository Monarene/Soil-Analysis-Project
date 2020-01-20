# -*- coding: utf-8 -*-
"""
Created on Wed May 15 10:58:34 2019

@author: H P ENVY
"""

#importing the necessary packages
#importing the neccessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from utilities.preprocessing import ImageToArrayPreprocessor, AspectAwarePreprocessor
from utilities.datasets import SimpleDatasetLoader
from utilities.nn.cnn import FCHeadNet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop, SGD, Adam
from keras.applications import VGG16, inception_v3, vgg19
from keras.layers import Input
from keras.models import Model
from keras.layers.core import Dense
from imutils import paths
import numpy as np
import os
import cv2

#the necessary paths for the project
imagePath = r"C:\Users\H P ENVY\Desktop\Data Science\Soil Analysis Propject\dataset\soils"
modelPath = r"C:\Users\H P ENVY\Desktop\Data Science\Soil Analysis Propject\models"

#dealing with the data
imagePaths = list(paths.list_images(imagePath))
classNames = [pt.split(os.path.sep)[-2] for pt in imagePaths]
classNames = [str(x) for x in np.unique(classNames)]
aap = AspectAwarePreprocessor(224,224)
isp = ImageToArrayPreprocessor()

sdl = SimpleDatasetLoader(preprocessors = [aap, isp])
(data, labels) = sdl.load(imagePaths, verbose = 1)
data = data.astype("float") / 255.0

#split the dataset into the required stages
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size = 0.25, random_state = 42)

#Binarize the labels
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

#introduce the baseModel, buiid the headmodel, introduce the class objects
#baseModel = inception_v3.InceptionV3
baseModel = VGG16(weights = "imagenet", include_top = False, input_tensor  = Input(shape = (224,224,3)))
headModel = FCHeadNet.build(baseModel, len(classNames), 256)
model = Model(inputs  = baseModel.input , outputs = headModel)

#freezing the layers in the baseModel and warming them up
for layer in baseModel.layers:
    layer.trainable = True

#warming up the mdoel for some action
optimizer = SGD(lr = 0.001)
model.compile(loss = "categorical_crossentropy",metrics = ['accuracy'], optimizer = optimizer)
model.fit(trainX, trainY, batch_size = 16, epochs =60, validation_data = (testX, testY))





