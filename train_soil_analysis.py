# -*- coding: utf-8 -*-
"""
Created on Sat May 25 12:01:30 2019

@author: H P ENVY
"""

#its time to bring out the big guns+
import matplotlib
matplotlib.use("Agg")
from config import soil_analysis_config as config
from utilities.preprocessing import MeanPreprocessor, ImageToArrayPreprocessor, PatchPreprocessor, SimplePreprocessor
from utilities.io import HDF5DatasetGenerator 
from utilities.nn.cnn import AlexNet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import os
import json

#cosntructing the Image data preprocessor and using it
means = json.loads(open(config.DATASET_MEAN).read())
aug = ImageDataGenerator(rotation_range = 20, zoom_range = 0.15, width_shift_range = 0.2, height_shift_range = 0.2,
                         shear_range = 0.15, horizontal_flip = True, fill_mode = "nearest")
iap = ImageToArrayPreprocessor()
pp = PatchPreprocessor(227, 227)
sp = SimplePreprocessor(227, 227)
mp = MeanPreprocessor(means["R"], means["G"], means["B"])

#initialize the dataset generators
#initialize the dataset generators
trainGen  = HDF5DatasetGenerator(config.TRAIN_HDF5, 4, aug=aug, preprocessors =[pp, mp, iap], classes = 3)
valGen = HDF5DatasetGenerator(config.VAL_HDF5, 4, aug=aug, preprocessors=[sp, mp, iap], classes = 3)

#compile the model and do al the deep learning
optimizer = Adam(lr = 1e-3)
model = AlexNet.build(width = 227, height = 227, depth = 3, reg = 0.0002, classes = 3)
model.compile(loss = "binary_crossentropy", optimizer = optimizer, metrics = ["accuracy"])

#construct callbacks
path = os.path.sep.join([config.OUTPUT_PATH, "{}.png".format(os.getpid())])
model.fit_generator(trainGen.generator(), steps_per_epoch = trainGen.numImages // 4,
                   validation_data = valGen.generator(), validation_steps = valGen.numImages // 4,
                   epochs = 10, max_queue_size = 4*2, verbose = 1)

model.save(config.MODEL_PATH, overwrite = True)
trainGen.close()
valGen.close()

