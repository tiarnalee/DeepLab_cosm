#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 21 21:33:15 2022

@author: tle19
"""
import os
os.chdir('/home/tle19/Documents/deeplab_cosm')
import numpy as np
import glob
from batchgenerators.utilities.file_and_folder_operations import *
from sklearn.model_selection import train_test_split
import re
import yaml
import shutil

base_folder = '/home/tle19/Documents/deeplab_cosm/datasets/Figaro1k/'
training_validation_path = base_folder + 'GT/images'
classes=['3A', '3B', '3C', '4A', '4B', '4C']

im_paths=glob.glob(base_folder+'GT/images/*')
labs=glob.glob(base_folder+'GT/np_labels/*') # copy labels in numpy format

annot=['straight', 'wavy', 'curly', 'kinky', 'braids', 'dreadlocks', 'short']
labels=[]
#replace labels with values 0-7
for i in labs:
    labels.append(annot[int(np.floor(int(re.findall('\d+', i.split('/')[-1])[0])/150))-1])

# Split into training and test
X_train, X_test, y_train, y_test = train_test_split(im_paths, labels, test_size=0.1, shuffle=True, stratify=labels) #10% testing
# Split into training and val
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=1/4.5, shuffle=True, stratify=y_train) #~20% val, ~70% train 

print(f'There are {len(X_train)} training images and {len(X_val)} test images ({len(X_train)+len(X_val)} total)')

# MAKE FOLDERS
training_folder= base_folder + 'train/image/'
validation_folder= base_folder + 'validation/image/'
label_folder= base_folder + 'labels/'
test_folder= base_folder + 'test/image/'
all_folders=[base_folder, base_folder +'train', base_folder+'validation', base_folder+'test', training_folder, validation_folder, label_folder, test_folder]
_=[os.mkdir(i) for i in all_folders if os.path.isdir(i)==False]

#Copy images to corresponding folders
_=[shutil.copy(i, training_folder) for i in X_train]
_=[shutil.copy(i, validation_folder) for i in X_val]
_=[shutil.copy(i, test_folder) for i in X_test]

train, val, test=[], [], []
#get lost of images in train, val nad test
_=[train.append( {'annotation': i.split('/')[-1][:-8] + '.npy', 'image':  i.split('/')[-1] }) for i in X_train]
_=[val.append( {'annotation': i.split('/')[-1][:-8] + '.npy', 'image':  i.split('/')[-1] }) for i in  X_val]
_=[test.append( {'annotation': i.split('/')[-1][:-8] + '.npy', 'image':  i.split('/')[-1] }) for i in X_test]

#save list to json
json_file={'train': train, 'test': test, 'val':val}
      
save_json(json_file,'/home/tle19/Documents/deeplab_cosm/datasets/Figaro1k/train_val_test.json')

#Copy labels to label folder
#Copy val and test images to training folder. They won't be used, they just have to be present
print('\nCopying test files to training')

_=[shutil.copy(i, label_folder) for i in labs]
_=[shutil.copy(i, training_folder) for i in X_test]
_=[shutil.copy(i, training_folder) for i in X_val]

with open("/home/tle19/Documents/deeplab_cosm/configs/config_default.yml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

config['dataset']['base_path']=base_folder
config['training']['model_last_checkpoint']['enabled']=False
config['image']['base_size']=227
config['image']['crop_size']=227

with open("/home/tle19/Documents/deeplab_cosm/configs/config1k.yml", 'w') as f:
    yaml.dump(config, f)

print('\nAmmended YAML files')

# write training command to .sh file
# with open("/home/tle19/Desktop/deeplab_torch/run_training.sh", 'w') as f:
#     f.write('python main.py -c configs/config1k.yml --train')
