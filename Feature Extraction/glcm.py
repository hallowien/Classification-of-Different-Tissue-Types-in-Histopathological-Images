# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 19:07:04 2021

@author: kubra
"""

import cv2

import numpy as np
from matplotlib import pyplot as plt
plt.style.use('ggplot')
import sys
sys.path.append('../common/')

import os
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from skimage.feature import greycomatrix, greycoprops
from skimage.color import rgb2gray
from skimage import feature

from skimage import img_as_ubyte
from matplotlib import pyplot as plt


train_path = 'D:/proje/ICIAR2018_BACH_Challenge/patch256_thresh110_son/'


def glcm_feature(img):

    img = rgb2gray(img)
    img = img_as_ubyte(img)
    
    bins = np.array([0, 32, 64, 96, 128, 160, 192, 224, 255]) # 8 bits
    inds = np.digitize(img, bins)
    
    max_value = inds.max()+1
 
    matrix_coocurrence = greycomatrix(inds, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=max_value, normed=False, symmetric=False)
    
    props = np.zeros((24))
    
    cont = greycoprops(matrix_coocurrence, 'contrast')
    diss = greycoprops(matrix_coocurrence, 'dissimilarity')
    homog = greycoprops(matrix_coocurrence, 'homogeneity')
    energ = greycoprops(matrix_coocurrence, 'energy')
    correl = greycoprops(matrix_coocurrence, 'correlation')
    asm = greycoprops(matrix_coocurrence, 'ASM')
    
    props[0] = cont[0][0]  #1
    props[1] = cont[0][1]  #2
    props[2] = cont[0][2]  #3
    props[3] = cont[0][3]  #4
    props[4] = diss[0][0]       #5
    props[5] = diss[0][1]       #6
    props[6] = diss[0][2]       #7
    props[7] = diss[0][3]       #8
    props[8] = homog[0][0]      #9
    props[9] = homog[0][1]      #10
    props[10] = homog[0][2]     #11
    props[11] = homog[0][3]     #12
    props[12] = energ[0][0]     #13
    props[13] = energ[0][1]     #14
    props[14] = energ[0][2]     #15
    props[15] = energ[0][3]     #16
    props[16] = correl[0][0]    #17
    props[17] = correl[0][1]    #18
    props[18] = correl[0][2]    #19
    props[19] = correl[0][3]    #20
    props[20] = asm[0][0]       #21
    props[21] = asm[0][1]       #22
    props[22] = asm[0][2]       #23
    props[23] = asm[0][3]       #24

    return props

labels = []
features = []


# filter all the warnings
import warnings
warnings.filterwarnings('ignore')

train_labels = os.listdir(train_path)
train_labels.sort()


# Iterate over the training images:
for typefile in train_labels:
    lesiontype = os.listdir(os.path.join(train_path,typefile))
    current_label = typefile
    for filename in lesiontype:   
        imgpath = os.path.join(train_path,typefile,filename)
        image = cv2.imread(imgpath)
        print(imgpath)
        feature = glcm_feature(image)
        labels.append(current_label)
        features.append(feature)

        

        


print ("[STATUS] completed Global Feature Extraction...")
print ("[STATUS] feature vector size {}".format(np.array(features).shape))
print ("[STATUS] training Labels {}".format(np.array(labels).shape))

print("one feature example:\n")
print(features[1])
print("\n")
len(features),len(labels)

print ("[STATUS] completed Global Feature Extraction...")

# get the overall feature vector size
print ("[STATUS] feature vector size {}".format(np.array(features).shape))

# get the overall training label size
print ("[STATUS] training Labels {}".format(np.array(labels).shape))

targetNames = np.unique(labels)
le = LabelEncoder()
target = le.fit_transform(labels)
print ("[STATUS] training labels encoded...")

# normalize the feature vector in the range (0-1)
scaler = MinMaxScaler(feature_range=(0, 1))
rescaled_features = scaler.fit_transform(features)
print ("[STATUS] feature vector normalized...")

print ("[STATUS] target labels: {}".format(target))
print( "[STATUS] target labels shape: {}".format(target.shape))

rescaled_features= np.nan_to_num(rescaled_features,0)


models = []
models.append(('KNN', KNeighborsClassifier(n_neighbors=7)))
models.append(('RF', RandomForestClassifier(n_estimators=700, random_state=9)))
models.append(('SVM', SVC(C=200,gamma=2)))

results = []
names = []
scoring = "accuracy"

(trainDataGlobal, testDataGlobal, trainLabelsGlobal, testLabelsGlobal) = train_test_split(np.array(rescaled_features),
                                                                                          np.array(target),
                                                                                          test_size=0.2,
                                                                                          random_state=9)
print("\n")
print("************************************************************")
print("K-Fold Results")
print("************************************************************")
print("\n")

for name, model in models:
    kfold = KFold(n_splits=5, random_state=9)
    cv_results = cross_val_score(model, trainDataGlobal, trainLabelsGlobal, cv=kfold, scoring=scoring)
    print(cross_val_score(model, trainDataGlobal,trainLabelsGlobal, cv=kfold, scoring=scoring))
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    
    
print("\n") 
print("************************************************************")
print("************************************************************")
print("\n")

from sklearn import metrics
for name, model in models:
    clf=model
    clf.fit(trainDataGlobal, trainLabelsGlobal)
    y_pred=clf.predict(testDataGlobal)
    msg = "%s: %f " % (name, metrics.accuracy_score(y_pred,testLabelsGlobal))
    print(msg)
    
target_names = ['Benign', 'InSitu', 'Invasive', 'Normal']
print(classification_report(testLabelsGlobal, y_pred, target_names=target_names))

import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix

plot_confusion_matrix(clf, testDataGlobal, testLabelsGlobal)  
plt.show()  