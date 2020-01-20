# -*- coding: utf-8 -*-
"""
Created on Fri May 17 17:03:04 2019

@author: H P ENVY
"""

#importimg the necessary libraries
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array, load_img
from keras.models import load_model
import numpy as np
import pickle
import os

def Analyze(imgPath):
    #loading the necssary models
    truth = ["Clay Soil", "loamy Soil", "Sandy Soil"]
    model_1 = load_model(r"C:\Users\H P ENVY\Desktop\Data Science\Soil Analysis Propject\models\real_vgg16.h5")
    model_2 = pickle.loads(open(r"C:\Users\H P ENVY\Desktop\Data Science\Soil Analysis Propject\models\first_model.pickle", "rb").read())
    
    #preprocessing the image and predicting
    image  = load_img(imgPath, target_size = (224,224))
    image  = img_to_array(image)
    image  = imagenet_utils.preprocess_input(image)
    image = np.expand_dims(image, axis  = 0)
    features = model_1.predict(image)
    features = features.reshape((1, 512*7*7))
    final_result  = model_2.predict(features)
    return truth[final_result[0]]

print(Analyze(r"C:\Users\H P ENVY\Desktop\Data Science\Soil Analysis Propject\dataset\soils\Loamy soil\pic.jpg"))
    
    
    
    
    
    