#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 13:20:19 2022

@author: tle19
"""
import os
os.chdir('/home/tle19/Documents/deeplab_cosm')
import torch
import numpy as np

from data_generators.NUMPY_loader1 import np_loader
import cv2
import random
import json
import yaml
from predictors.predictor import Predictor
import sys

#% Predict on test set 

def dice_sim_coef(result, reference):
    result=np.atleast_1d(result.astype(bool))
    reference=np.atleast_1d(reference.astype(bool))
    
    tp=np.count_nonzero(result & reference)
    fp=np.count_nonzero(result & ~reference)
    fn=np.count_nonzero(~result & reference)
    
    try:
        dc=2. *tp/ float(2*tp +fp +fn)
    except ZeroDivisionError:
        dc=0.0
    return dc

with open("/home/tle19/Documents/deeplab_cosm/configs/config1k.yml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    
with open('/home/tle19/Documents/deeplab_cosm/datasets/Figaro1k/train_val_test.json') as json_file:
    dataset = json.load(json_file)
results=[]
base_size = config['image']['base_size']
    #load predictor for task
predictor = Predictor(config, checkpoint_path='/home/tle19/Documents/deeplab_cosm/experiments/last.pth.tar')

for index in range(len(dataset['test'])):
    file_name = '/home/tle19/Documents/deeplab_cosm/datasets/Figaro1k/test/image/' + dataset['test'][index]['image']
    annotation = np.load('/home/tle19/Documents/deeplab_cosm/datasets/Figaro1k/labels/' + dataset['test'][index]['annotation'])

    annotation = cv2.resize(annotation, (base_size, base_size), interpolation=cv2.INTER_NEAREST)
    #find prediction
    _, prediction = predictor.segment_image(file_name)
    
    DSC=dice_sim_coef(annotation[:,:,0], prediction)
    results.append([dataset['test'][index]['image'][:-8], DSC])            
    # np.save('/home/tle19/Documents/deeplab_cosm/results/' + dataset['test'][index]['annotation'], prediction)

