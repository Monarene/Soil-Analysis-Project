# -*- coding: utf-8 -*-
"""
Created on Thu May 16 10:57:51 2019

@author: H P ENVY
"""

#importing the necessary libraries
from sklearn.preprocessing import LabelEncoder
from keras.applications import VGG16, imagenet_utils
from keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from utilities.io import HDF5DatasetWriter
from imutils import paths
from keras.models import Model
from keras.models import load_model
import numpy as np
import pickle 
import random
import os
import h5py
random.seed(42)

#the necessary variables
dataset_path = r"C:\Users\H P ENVY\Desktop\Data Science\Soil Analysis Propject\dataset\soils"
output_path  = r'C:\Users\H P ENVY\Desktop\Data Science\Soil Analysis Propject\hdf5 dataset/extracted_features_soil.hdf5'
bs = 10
bufferSize = 20

#dealing with the raw image data
imagePaths = list(paths.list_images(dataset_path))
random.shuffle(imagePaths)
labels = [p.split(os.path.sep)[-2] for p in imagePaths]

le = LabelEncoder()
labels  = le.fit_transform(labels)

#preparing the model for use 
model = VGG16(weights = "imagenet", include_top = False)
model.save(r"C:\Users\H P ENVY\Desktop\Data Science\Soil Analysis Propject\models\real_vgg16.h5")
dataset = HDF5DatasetWriter((len(imagePaths), 512 * 7 * 7), output_path, dataKey = "features",
                            bufSize = bufferSize)
dataset.storeClassLabels(le.classes_)

#looping over all the images, preprocessing them, and storing them in hdf5
for i in np.arange(0, len(imagePaths), bs):
    batchPaths = imagePaths[i: i + bs]
    batchLabels = labels[i: i + bs]
    batchImages = []
    
    for imagePath in batchPaths:
        image = load_img(imagePath, target_size = (224,224))
        image = img_to_array(image)
        image = np.expand_dims(image, axis = 0)
        image = imagenet_utils.preprocess_input(image)
        batchImages.append(image)

    batchImages = np.vstack(batchImages)
    features = model.predict(batchImages, batch_size = bs)
    features = features.reshape((features.shape[0], 512 * 7 * 7))
    dataset.add(features, batchLabels)

dataset.close()

#starting the real machine learning and evaluating the results
db = h5py.File(output_path, "a")
db["targets"] = labels
i = int(db["labels"].shape[0] * 0.75)

#logistic regression model
params  = {'C':[0.1, 0.5, 2.0, 5.0, 10]}
model = GridSearchCV(LogisticRegression(), params, cv = 5 , verbose = 0, n_jobs = -1)
model.fit(db["features"][:i], db["targets"][:i])
print("[INFO] best parameters:{}".format(model.best_params_))

#Gradient boosting classifier model
params = {"learning_rate":[0.01, 0.05, 0.1], "n_estimators":np.arange(100,400,50),
          "max_features":[1,2,3], "max_depth":[1,2,3], "random_state":np.arange(7,100, 7)}
model = GridSearchCV(GradientBoostingClassifier(), params, cv = 5, verbose = 0, n_jobs = -1)
model.fit(db["features"][:i], db["targets"][:i])
print("[INFO] best parameters:{}".format(model.best_params_))

#model evaluation and checking
preds  = model.predict(db["features"][i:])
print(classification_report(db["targets"][i:], preds, target_names  = le.classes_))
print(confusion_matrix(preds, db["targets"][i:]))

#Bringing the model back from rest
vgg = load_model(r"C:\Users\H P ENVY\Desktop\Data Science\Soil Analysis Propject\models\real_vgg16.h5")











