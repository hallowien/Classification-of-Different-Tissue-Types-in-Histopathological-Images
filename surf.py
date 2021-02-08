# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 19:19:41 2021

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
from skimage import feature



train_path = 'D:/proje/ICIAR2018_BACH_Challenge/patch256_thresh110_son/'

output_folder = "output"
#bins for colour histogram
bins = 32
random_seed = 9 



def surf_feature(image):
    gray= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    surf = cv2.xfeatures2d.SURF_create()
            # Computing key points       
    kp, dsc = surf.detectAndCompute(gray, None)  # get keypoint and descriptor
    img = dsc.mean(axis=0)
    return img


labels = []
features = []
hog_images=[]

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
        feature = surf_feature(image)
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