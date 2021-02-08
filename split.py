
import random

from matplotlib import pyplot as plt


plt.style.use('ggplot')

import sys
sys.path.append('../common/')

import os
import shutil


def create_folder(folder_name):
	if(not os.path.exists(folder_name)):
		os.mkdir(folder_name)
        

output_folder_name = "D:/proje/ICIAR2018_BACH_Challenge/split256/"
validation_size = 0.2
input_folder_name = "D:/proje/ICIAR2018_BACH_Challenge/patch256/"
random_seed = 343
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
        input_train_image = (input_folder_name_label + "/" + folder[j])
        output_train_image = (output_train_folder_name_label + "/" + folder[j])
        shutil.move(input_train_image, output_train_image)
    for j in range(train_size, len(folder)):
        input_validation_image = (input_folder_name_label + "/" + folder[j])
        output_validation_image = (output_validation_folder_name_label + "/" + folder[j])
        shutil.move(input_validation_image, output_validation_image)


