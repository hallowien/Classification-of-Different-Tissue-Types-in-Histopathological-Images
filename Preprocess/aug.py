# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 20:34:01 2021

@author: kubra
"""

import cv2
import numpy as np
import os
from PIL import Image
from PIL import ImageOps
from joblib import Parallel, delayed
import random

centers = np.array([ [122,28,45],
	[231,214,211],
	[209,157,187],
	[171,70,168] 
	])

def create_folder(folder_name):
	if(not os.path.exists(folder_name)):
		os.mkdir(folder_name)
  
		
def augmentation(input_image_name, output_image_name):
	image = Image.open(input_image_name)
	image.save(output_image_name) #save original image

	output_image_name_splitted = os.path.splitext(output_image_name)

	image.save(output_image_name_splitted[0] + output_image_name_splitted[1])

	edited_image = image.rotate(90)
	edited_image.save(output_image_name_splitted[0] + "_90" + output_image_name_splitted[1])

	edited_image = image.rotate(180)
	edited_image.save(output_image_name_splitted[0] + "_180" + output_image_name_splitted[1])

	edited_image = image.rotate(270)
	edited_image.save(output_image_name_splitted[0] + "_270" + output_image_name_splitted[1])

	edited_image = ImageOps.mirror(image)
	edited_image.save(output_image_name_splitted[0] + "_mirror" + output_image_name_splitted[1])

def folder_edit(input_folder_name, output_folder_name, function_name, patch_sizes, function_args, validation_size, random_seed):
	labels = os.listdir(input_folder_name)

	input_train_images = []
	input_validation_images = []
	output_train_images = []
	output_validation_images = []

	if(validation_size > 0):
		output_train_folder_name = output_folder_name + "/Train/"
		output_validation_folder_name = output_folder_name + "/Validation/"
	else:
		output_train_folder_name = output_validation_folder_name = output_folder_name + "/"

	create_folder(output_folder_name)
	create_folder(output_train_folder_name)
	create_folder(output_validation_folder_name)

	for label in labels:
		input_folder_name_label = input_folder_name + "/" + label
		folder = os.listdir(input_folder_name_label)
		random.Random(random_seed).shuffle(folder)

		output_train_folder_name_label = output_train_folder_name + label
		output_validation_folder_name_label = output_validation_folder_name+ label

		create_folder(output_train_folder_name_label)
		create_folder(output_validation_folder_name_label)

		train_size = int(len(folder)*(1.0-validation_size))
		for j in range(0, train_size):
			input_train_images.append(input_folder_name_label + "/" + folder[j])
			output_train_images.append(output_train_folder_name_label + "/" + folder[j])
		for j in range(train_size, len(folder)):
			input_validation_images.append(input_folder_name_label + "/" + folder[j])
			output_validation_images.append(output_validation_folder_name_label + "/" + folder[j])

	if(function_name == "augmentation"):
		Parallel(n_jobs=-1)(delayed(augmentation)(input_train_images[i], output_train_images[i] ) for i in range(0,len(input_train_images)))
		#augmentation is for trainset, not for validation

TRAIN_PATH = 'D:/proje/ICIAR2018_BACH_Challenge/dene/'
PATCH_PATH = 'D:/proje/ICIAR2018_BACH_Challenge/dene256/'
folder_edit(TRAIN_PATH,PATCH_PATH,"augmentation",(256,256),(256,256),0.2,5)