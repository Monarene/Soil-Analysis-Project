# -*- coding: utf-8 -*-
"""
Created on Wed May 15 10:17:04 2019

@author: H P ENVY
"""

#code to do augmentation on the clay dataset
#import the necessary packages
from imutils import paths
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import os
import cv2

#the relevant paths
clay_path = r"C:\Users\H P ENVY\Desktop\Data Science\Soil Analysis Propject\dataset\Clay soil"
new_clay_path  = r"C:\Users\H P ENVY\Desktop\Data Science\Soil Analysis Propject\dataset\Clay"

#getting the list of image from the old packages
imagePaths = list(paths.list_images(clay_path))

aug = ImageDataGenerator(rotation_range = 30, width_shift_range = 0.1,
                         height_shift_range = 0.1, shear_range=0.2, zoom_range = 0.2,
                         horizontal_flip = True, fill_mode = "nearest")

for picture in imagePaths:
    image = load_img(picture)
    image = img_to_array(image)
    image = np.expand_dims(image, axis = 0)
    total  = 0
    imageGen = aug.flow(image, batch_size  = 1, save_to_dir = new_clay_path,
                        save_prefix = "new", save_format = "jpg")
    for image in imageGen:
        total += 1
        if total == 1:
            break