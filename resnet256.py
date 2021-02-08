# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 01:25:07 2021

@author: kubra
"""

import cv2
import numpy as np
import os
from PIL import Image
from PIL import ImageOps
from joblib import Parallel, delayed
import random
from keras import layers, optimizers
from keras import models
import os
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.utils import to_categorical
from keras.preprocessing.image import  ImageDataGenerator, img_to_array, load_img
from keras import backend as K
from keras.optimizers import Adam, SGD
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from keras.models import load_model
from keras.models import Model
import keras
from keras.applications.resnet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator


from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [16, 10]
plt.rcParams['font.size'] = 16


train_path="D:/proje/ICIAR2018_BACH_Challenge/aug256/Train"
test_path='D:/proje/ICIAR2018_BACH_Challenge/patch256_thresh110/'

CATEGORIES = ['Benign', 'InSitu', 'Normal', 'Invasive']
NUM_CATEGORIES = len(CATEGORIES)
NUM_CATEGORIES
train = []
for category_id, category in enumerate(CATEGORIES):
    for file in os.listdir(os.path.join(train_path, category)):
        train.append(['train/{}/{}'.format(category, file), category_id, category])
train = pd.DataFrame(train, columns=['file', 'category_id', 'category'])
train.shape

from keras.applications import ResNet50

def build_model(backbone, lr=1e-4):
    model = models.Sequential()
    model.add(backbone)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.5))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(4, activation='softmax'))
    
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizers.Adam(lr=lr),
        metrics=['accuracy']
    )
    
    return model

resnet = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)

def printHistory(history, title, epochs):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    t = f.suptitle(title, fontsize=12)
    f.subplots_adjust(top=0.85, wspace=0.3)

    epoch_list = list(range(1,epochs+1))
    ax1.plot(epoch_list, history.history['accuracy'], label='Train Accuracy')
    ax1.plot(epoch_list, history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_xticks(np.arange(0, epochs+1, 5))
    ax1.set_ylabel('Accuracy Value')
    ax1.set_xlabel('Epoch')
    ax1.set_title('Accuracy')
    l1 = ax1.legend(loc="best")

    ax2.plot(epoch_list, history.history['loss'], label='Train Loss')
    ax2.plot(epoch_list, history.history['val_loss'], label='Validation Loss')
    ax2.set_xticks(np.arange(0, epochs+1, 5))
    ax2.set_ylabel('Loss Value')
    ax2.set_xlabel('Epoch')
    ax2.set_title('Loss')
    l2 = ax2.legend(loc="best")
    
        
print("Cross validation")
kfold = StratifiedKFold(n_splits=5, shuffle=True)
cvscores = []
iteration = 1

t = train.category_id 
    
for train_index, test_index in kfold.split(np.zeros(len(t)), t):

        print("======================================")
        print("Iteration = ", iteration)

        iteration = iteration + 1


        print("======================================")
        
        model = build_model(resnet ,lr = 1e-4)

        print("======================================")
       
        

        train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
        test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
        train_generator=train_datagen.flow_from_directory(
                                                  train_path,
                                                  target_size=(224,224),
                                                  batch_size=32,
                                                  class_mode='categorical')
        valid_generator=test_datagen.flow_from_directory(
                                                  test_path,
                                                  target_size=(224,224),
                                                  batch_size=32,
                                                  class_mode='categorical')
        for data_batch,labels_batch in train_generator:
            print('data batch shape:',data_batch.shape)
            print('labels batch shape:',labels_batch.shape)
            break

        #Trains the model on data generated batch-by-batch by a Python generator
        history=model.fit_generator(
                                    train_generator,
                                    steps_per_epoch=10610//32,
                                    epochs=25,
                                    validation_data=valid_generator,
                                    validation_steps=3399//32)
        model.save('proje_model_resnet50_256.h5')
        
        scores = model.evaluate_generator(generator=valid_generator, steps=3399//32)
        print("Accuarcy %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)
        
        printHistory(history, "ResNet50", 25)

accuracy = np.mean(cvscores);
std = np.std(cvscores);
print("Accuracy: %.2f%% (+/- %.2f%%)" % (accuracy, std))
